from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


FILM_VARIANTS = {
    "film_additive",
    "dose_gated_film",
    "dose_gated_film_adaptive",
    "dose_gated_film_cell",
    "dose_gated_film_cell_adaptive",
}
DOSE_GATED_VARIANTS = FILM_VARIANTS - {"film_additive"}
ADAPTIVE_SLOPE_VARIANTS = {
    "dose_gated_film_adaptive",
    "dose_gated_film_cell_adaptive",
}
CELL_POTENCY_VARIANTS = {
    "dose_gated_film_cell",
    "dose_gated_film_cell_adaptive",
}


class ResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(width * 2, width),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.block(values)


class StructureConditionedPerturbationModel(nn.Module):
    """Phase-1 structure/context/baseline model with no compound-ID embeddings."""

    def __init__(
        self,
        output_dim: int,
        num_cell_lines: int,
        latent_dim: int = 128,
        dropout: float = 0.15,
        response_basis: torch.Tensor | None = None,
        variant: str = "attention",
        fingerprint_hidden_dim: int = 256,
        descriptor_hidden_dim: int = 48,
        descriptor_input_dim: int = 8,
        fusion_depth: int = 2,
        attention_heads: int = 4,
        fingerprint_dropout: float = 0.0,
        baseline_mode: str = "encoder",
    ) -> None:
        super().__init__()
        if variant not in {
            "attention",
            "chemical_attention",
            "factorized",
            "film_additive",
            "dose_gated_film",
            "dose_gated_film_adaptive",
            "dose_gated_film_cell",
            "dose_gated_film_cell_adaptive",
        }:
            raise ValueError(f"Unknown model variant: {variant}")
        if fusion_depth < 1:
            raise ValueError("fusion_depth must be positive")
        if not 0.0 <= fingerprint_dropout < 1.0:
            raise ValueError("fingerprint_dropout must be in [0, 1)")
        if variant in {"attention", "chemical_attention"} and latent_dim % attention_heads:
            raise ValueError("latent_dim must be divisible by attention_heads")
        if baseline_mode not in {"encoder", "none"}:
            raise ValueError(f"Unknown baseline mode: {baseline_mode}")
        self.variant = variant
        self.fingerprint_dropout = fingerprint_dropout
        self.baseline_mode = baseline_mode
        self.chemical_encoder = nn.Sequential(
            nn.Linear(2048, fingerprint_hidden_dim),
            nn.LayerNorm(fingerprint_hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(fingerprint_hidden_dim, latent_dim),
        )
        self.descriptor_encoder = nn.Sequential(
            nn.Linear(descriptor_input_dim, descriptor_hidden_dim),
            nn.SiLU(),
            nn.Linear(descriptor_hidden_dim, latent_dim),
        )
        self.cell_embedding = nn.Embedding(num_cell_lines + 1, 32, padding_idx=0)
        self.context_encoder = nn.Sequential(
            nn.Linear(32 + 2, 64),
            nn.SiLU(),
            nn.Linear(64, latent_dim),
        )
        self.baseline_encoder = (
            nn.Sequential(
                nn.LayerNorm(output_dim),
                nn.Linear(output_dim, 128),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(128, latent_dim),
            )
            if baseline_mode == "encoder"
            else None
        )
        self.token_norm = nn.LayerNorm(latent_dim)
        if variant in {"attention", "chemical_attention"}:
            self.attention = nn.MultiheadAttention(
                embed_dim=latent_dim,
                num_heads=attention_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.factorized_fusion = None
            self.film_modulator = None
        elif variant == "factorized":
            self.attention = None
            self.factorized_fusion = nn.Sequential(
                nn.Linear(latent_dim * 4, latent_dim * 2),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(latent_dim * 2, latent_dim),
            )
            self.film_modulator = None
        else:
            self.attention = None
            self.factorized_fusion = None
            self.film_modulator = nn.Sequential(
                nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, latent_dim * 2),
            )
        if variant == "dose_gated_film":
            # Keep this exact module layout compatible with existing checkpoints.
            self.potency_predictor = nn.Sequential(
                nn.LayerNorm(latent_dim), nn.Linear(latent_dim, 1), nn.Tanh()
            )
        elif variant in DOSE_GATED_VARIANTS:
            potency_outputs = 2 if variant in ADAPTIVE_SLOPE_VARIANTS else 1
            self.potency_predictor = nn.Sequential(
                nn.LayerNorm(latent_dim), nn.Linear(latent_dim, potency_outputs)
            )
        else:
            self.potency_predictor = None
        self.dose_slope_raw = (
            nn.Parameter(torch.tensor(1.0)) if variant in DOSE_GATED_VARIANTS else None
        )
        self.potency_cell_shift = (
            nn.Embedding(num_cell_lines + 1, 1, padding_idx=0)
            if variant in CELL_POTENCY_VARIANTS
            else None
        )
        if self.potency_cell_shift is not None:
            nn.init.zeros_(self.potency_cell_shift.weight)
        self.fusion = nn.Sequential(
            *[ResidualBlock(latent_dim, dropout) for _ in range(fusion_depth)],
            nn.LayerNorm(latent_dim),
        )
        if response_basis is not None:
            if response_basis.ndim != 2 or response_basis.shape[1] != output_dim:
                raise ValueError("Response basis must have shape (rank, output_dim)")
            self.register_buffer("response_basis", response_basis.float())
            decoder_dim = int(response_basis.shape[0])
        else:
            self.response_basis = None
            decoder_dim = output_dim
        self.decoder = self._make_decoder(latent_dim, decoder_dim, dropout)
        self.interaction_decoder = (
            self._make_decoder(latent_dim, decoder_dim, dropout)
            if variant in FILM_VARIANTS
            else None
        )

    @staticmethod
    def _make_decoder(input_dim: int, output_dim: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, output_dim),
        )

    def forward(
        self,
        fingerprint: torch.Tensor,
        descriptors: torch.Tensor,
        cell_index: torch.Tensor,
        context: torch.Tensor,
        baseline: torch.Tensor,
    ) -> torch.Tensor:
        if self.training and self.fingerprint_dropout > 0.0:
            keep = torch.rand_like(fingerprint) >= self.fingerprint_dropout
            fingerprint = fingerprint * keep
        chemical = self.chemical_encoder(fingerprint) + self.descriptor_encoder(descriptors)
        context_token = self.context_encoder(torch.cat([self.cell_embedding(cell_index), context], dim=-1))
        baseline_token = (
            self.baseline_encoder(baseline)
            if self.baseline_encoder is not None
            else torch.zeros_like(context_token)
        )
        if self.variant in {"attention", "chemical_attention"}:
            tokens = self.token_norm(torch.stack([chemical, context_token, baseline_token], dim=1))
            assert self.attention is not None
            attended, _ = self.attention(tokens, tokens, tokens, need_weights=False)
            if self.variant == "chemical_attention":
                fused = tokens[:, 0] + attended[:, 0]
            else:
                fused = (tokens + attended).mean(dim=1)
        elif self.variant == "factorized":
            biological = context_token + baseline_token
            chemical = self.token_norm(chemical)
            biological = self.token_norm(biological)
            interaction = torch.cat(
                [chemical, biological, chemical * biological, torch.abs(chemical - biological)],
                dim=-1,
            )
            assert self.factorized_fusion is not None
            fused = self.factorized_fusion(interaction)
        else:
            chemical = self.token_norm(chemical)
            if self.variant in DOSE_GATED_VARIANTS:
                assert self.potency_predictor is not None and self.dose_slope_raw is not None
                potency = self.potency_predictor(chemical)
                if self.variant in ADAPTIVE_SLOPE_VARIANTS:
                    midpoint = torch.tanh(potency[:, :1])
                    slope = F.softplus(
                        self.dose_slope_raw + 0.5 * torch.tanh(potency[:, 1:2])
                    ) + 1e-3
                else:
                    midpoint = (
                        potency if self.variant == "dose_gated_film" else torch.tanh(potency)
                    )
                    slope = F.softplus(self.dose_slope_raw) + 1e-3
                if self.potency_cell_shift is not None:
                    midpoint = torch.clamp(
                        midpoint + 0.35 * torch.tanh(self.potency_cell_shift(cell_index)),
                        min=-1.0,
                        max=1.0,
                    )
                dose_gate = torch.sigmoid(slope * (context[:, :1] - midpoint))
                chemical = chemical * dose_gate
            biological = self.token_norm(context_token + baseline_token)
            assert self.film_modulator is not None
            scale, shift = self.film_modulator(biological).chunk(2, dim=-1)
            fused = chemical * (1.0 + 0.5 * torch.tanh(scale)) + shift
        if self.variant in FILM_VARIANTS:
            assert self.interaction_decoder is not None
            decoded = self.decoder(chemical) + self.interaction_decoder(self.fusion(fused))
        else:
            decoded = self.decoder(self.fusion(fused))
        if self.response_basis is not None:
            return decoded @ self.response_basis
        return decoded


def correlation_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    pred = y_pred - y_pred.mean(dim=1, keepdim=True)
    true = y_true - y_true.mean(dim=1, keepdim=True)
    similarity = F.cosine_similarity(pred, true, dim=1, eps=1e-8)
    losses = 1.0 - similarity
    if sample_weight is None:
        return losses.mean()
    if sample_weight.ndim != 1 or sample_weight.shape[0] != y_pred.shape[0]:
        raise ValueError("Sample weight must match the batch dimension")
    return (losses * sample_weight).sum() / sample_weight.sum().clamp_min(1e-8)


def reconstruction_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    loss_type: str = "mse",
    huber_delta: float = 1.0,
    weight: torch.Tensor | None = None,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    if loss_type == "mse":
        elementwise = F.mse_loss(y_pred, y_true, reduction="none")
    elif loss_type == "huber":
        elementwise = F.huber_loss(y_pred, y_true, delta=huber_delta, reduction="none")
    else:
        raise ValueError(f"Unknown reconstruction loss: {loss_type}")
    if weight is not None:
        if weight.ndim != 1 or weight.shape[0] != y_pred.shape[-1]:
            raise ValueError("Reconstruction weight must match the output dimension")
        elementwise = elementwise * weight
    per_sample = elementwise.mean(dim=-1)
    if sample_weight is None:
        return per_sample.mean()
    if sample_weight.ndim != 1 or sample_weight.shape[0] != y_pred.shape[0]:
        raise ValueError("Sample weight must match the batch dimension")
    return (per_sample * sample_weight).sum() / sample_weight.sum().clamp_min(1e-8)


def model_kwargs_from_config(config: dict) -> dict:
    descriptor_set = str(config.get("descriptor_set", "basic"))
    descriptor_dimensions = {"basic": 8, "extended": 23}
    if descriptor_set not in descriptor_dimensions:
        raise ValueError(f"Unknown descriptor set: {descriptor_set}")
    return {
        "latent_dim": int(config["latent_dim"]),
        "dropout": float(config["dropout"]),
        "variant": str(config.get("model_variant", "attention")),
        "fingerprint_hidden_dim": int(config.get("fingerprint_hidden_dim", 256)),
        "descriptor_hidden_dim": int(config.get("descriptor_hidden_dim", 48)),
        "descriptor_input_dim": descriptor_dimensions[descriptor_set],
        "fusion_depth": int(config.get("fusion_depth", 2)),
        "attention_heads": int(config.get("attention_heads", 4)),
        "fingerprint_dropout": float(config.get("fingerprint_dropout", 0.0)),
        "baseline_mode": str(config.get("baseline_mode", "encoder")),
    }
