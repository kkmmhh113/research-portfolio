from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def _mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    dropout: float,
    depth: int = 2,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    current = input_dim
    for _ in range(max(depth - 1, 0)):
        layers.extend(
            [
                nn.Linear(current, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout),
            ]
        )
        current = hidden_dim
    layers.append(nn.Linear(current, output_dim))
    return nn.Sequential(*layers)


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


class ReadoutEncoder(nn.Module):
    def __init__(
        self,
        readout_features: torch.Tensor,
        embedding_dim: int,
        dropout: float,
        use_metadata: bool = True,
    ) -> None:
        super().__init__()
        if readout_features.ndim != 2:
            raise ValueError("Readout features must be a two-dimensional tensor")
        self.register_buffer("features", readout_features.float())
        self.identity = nn.Embedding(readout_features.shape[0], embedding_dim)
        self.metadata_encoder = _mlp(
            readout_features.shape[1], embedding_dim, embedding_dim, dropout, depth=2
        )
        self.norm = nn.LayerNorm(embedding_dim)
        self.use_metadata = bool(use_metadata)

    def forward(self, readout_indices: torch.Tensor | None = None) -> torch.Tensor:
        if readout_indices is None:
            readout_indices = torch.arange(
                self.features.shape[0], device=self.features.device
            )
        metadata = (
            self.metadata_encoder(self.features[readout_indices])
            if self.use_metadata
            else torch.zeros_like(self.identity(readout_indices))
        )
        return self.norm(self.identity(readout_indices) + metadata)


class FourierConditionEncoder(nn.Module):
    """Smooth nonlinear dose/time representation without a monotonicity assumption."""

    def __init__(
        self,
        output_dim: int,
        frequencies: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.frequencies = int(frequencies)
        input_dim = 2 + 4 * self.frequencies + 1
        self.encoder = _mlp(input_dim, output_dim, output_dim, dropout, depth=2)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        continuous = context[:, :2]
        features = [continuous]
        for frequency in range(1, self.frequencies + 1):
            angle = math.pi * frequency * continuous
            features.extend([torch.sin(angle), torch.cos(angle)])
        features.append(context[:, 2:3])
        return self.encoder(torch.cat(features, dim=-1))


class GraphMessageLayer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.self_projection = nn.Linear(hidden_dim, hidden_dim)
        self.bond_projections = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(4)]
        )
        self.output = nn.Sequential(
            nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim)
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        values: torch.Tensor,
        atom_mask: torch.Tensor,
        bond_types: torch.Tensor,
    ) -> torch.Tensor:
        message = self.self_projection(values)
        for index, projection in enumerate(self.bond_projections, start=1):
            adjacency = (bond_types == index).to(values.dtype)
            degree = adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)
            neighbors = torch.bmm(adjacency, values) / degree
            message = message + projection(neighbors)
        result = self.norm(values + self.output(message))
        return result * atom_mask.unsqueeze(-1).to(result.dtype)


class LightweightGraphEncoder(nn.Module):
    """Small typed-message-passing encoder that does not require PyG."""

    def __init__(
        self,
        output_dim: int,
        hidden_dim: int = 128,
        layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        categorical_sizes = (119, 7, 11, 9, 6, 5, 2)
        embedding_dims = (32, 8, 8, 8, 8, 8, 4)
        self.feature_embeddings = nn.ModuleList(
            [
                nn.Embedding(size, dimension)
                for size, dimension in zip(categorical_sizes, embedding_dims)
            ]
        )
        self.input_projection = nn.Linear(sum(embedding_dims), hidden_dim)
        self.layers = nn.ModuleList(
            [GraphMessageLayer(hidden_dim, dropout) for _ in range(layers)]
        )
        self.pool = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        atom_features: torch.Tensor,
        atom_mask: torch.Tensor,
        bond_types: torch.Tensor,
    ) -> torch.Tensor:
        if atom_features.ndim != 3 or atom_features.shape[-1] != 7:
            raise ValueError("atom_features must have shape (batch, atoms, 7)")
        embedded = torch.cat(
            [
                embedding(atom_features[..., index])
                for index, embedding in enumerate(self.feature_embeddings)
            ],
            dim=-1,
        )
        values = self.input_projection(embedded)
        values = values * atom_mask.unsqueeze(-1).to(values.dtype)
        for layer in self.layers:
            values = layer(values, atom_mask, bond_types)
        weights = atom_mask.unsqueeze(-1).to(values.dtype)
        mean_pool = (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        minimum = torch.finfo(values.dtype).min
        max_pool = values.masked_fill(~atom_mask.unsqueeze(-1), minimum).max(dim=1).values
        max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))
        return self.pool(torch.cat([mean_pool, max_pool], dim=-1))


class ChemicalPerturbationEncoder(nn.Module):
    """Standalone P encoder with parameter names compatible with the PRC model."""

    def __init__(
        self,
        descriptor_input_dim: int,
        latent_dim: int,
        hidden_dim: int,
        descriptor_hidden_dim: int,
        dropout: float,
        mode: str = "ecfp",
        graph_message_layers: int = 3,
    ) -> None:
        super().__init__()
        if mode == "ecfp":
            self.chemical_encoder = nn.Sequential(
                nn.Linear(2048, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, latent_dim),
            )
        elif mode == "graph":
            self.chemical_encoder = LightweightGraphEncoder(
                output_dim=latent_dim,
                hidden_dim=hidden_dim,
                layers=graph_message_layers,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unknown chemical encoder mode: {mode}")
        self.descriptor_encoder = nn.Sequential(
            nn.Linear(descriptor_input_dim, descriptor_hidden_dim),
            nn.SiLU(),
            nn.Linear(descriptor_hidden_dim, latent_dim),
        )
        self.chemical_norm = nn.LayerNorm(latent_dim)
        self.mode = mode

    def forward(
        self,
        fingerprint: torch.Tensor,
        descriptors: torch.Tensor,
        atom_features: torch.Tensor | None = None,
        atom_mask: torch.Tensor | None = None,
        bond_types: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.mode == "ecfp":
            chemical = self.chemical_encoder(fingerprint)
        else:
            if atom_features is None or atom_mask is None or bond_types is None:
                raise ValueError("Graph mode requires atom_features, atom_mask, and bond_types")
            chemical = self.chemical_encoder(atom_features, atom_mask, bond_types)
        return self.chemical_norm(chemical + self.descriptor_encoder(descriptors))


class PRCFactorizedModel(nn.Module):
    """Structure-conditioned P/C/R model with an explicit gene readout encoder."""

    def __init__(
        self,
        readout_features: torch.Tensor,
        num_cell_lines: int,
        num_domains: int,
        response_basis: torch.Tensor | None = None,
        descriptor_input_dim: int = 23,
        latent_dim: int = 128,
        readout_dim: int = 96,
        fingerprint_hidden_dim: int = 256,
        baseline_hidden_dim: int = 192,
        descriptor_hidden_dim: int = 48,
        fusion_depth: int = 1,
        dropout: float = 0.15,
        fingerprint_dropout: float = 0.05,
        learned_readout_initial_logit: float = -5.0,
        dose_fourier_frequencies: int = 4,
        readout_attention_heads: int = 4,
        readout_attention_layers: int = 1,
        enable_distribution_head: bool = False,
        enable_response_signature_head: bool = False,
        target_auxiliary_dim: int = 0,
        decoder_mode: str = "query_attention",
        condition_mode: str = "fourier",
        chemical_encoder_mode: str = "ecfp",
        graph_message_layers: int = 3,
        use_cell_conditioning: bool = True,
        use_domain_conditioning: bool = True,
        use_dose_time_conditioning: bool = True,
        use_baseline_conditioning: bool = True,
        use_readout_metadata: bool = True,
    ) -> None:
        super().__init__()
        self.output_dim = int(readout_features.shape[0])
        self.readout_dim = int(readout_dim)
        if response_basis is not None:
            if response_basis.shape != (readout_dim, self.output_dim):
                raise ValueError(
                    "Response basis must have shape (readout_dim, output_dim)"
                )
            response_basis = response_basis.float()
        self.register_buffer("response_basis", response_basis, persistent=False)
        self.fingerprint_dropout = float(fingerprint_dropout)
        self.enable_distribution_head = bool(enable_distribution_head)
        self.enable_response_signature_head = bool(enable_response_signature_head)
        if decoder_mode not in {"query_attention", "gated_factorized"}:
            raise ValueError(f"Unknown decoder mode: {decoder_mode}")
        self.decoder_mode = decoder_mode
        if chemical_encoder_mode == "ecfp":
            self.chemical_encoder = nn.Sequential(
                nn.Linear(2048, fingerprint_hidden_dim),
                nn.LayerNorm(fingerprint_hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(fingerprint_hidden_dim, latent_dim),
            )
        elif chemical_encoder_mode == "graph":
            self.chemical_encoder = LightweightGraphEncoder(
                output_dim=latent_dim,
                hidden_dim=fingerprint_hidden_dim,
                layers=graph_message_layers,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unknown chemical encoder mode: {chemical_encoder_mode}")
        self.chemical_encoder_mode = chemical_encoder_mode
        self.use_cell_conditioning = bool(use_cell_conditioning)
        self.use_domain_conditioning = bool(use_domain_conditioning)
        self.use_dose_time_conditioning = bool(use_dose_time_conditioning)
        self.use_baseline_conditioning = bool(use_baseline_conditioning)
        self.descriptor_encoder = nn.Sequential(
            nn.Linear(descriptor_input_dim, descriptor_hidden_dim),
            nn.SiLU(),
            nn.Linear(descriptor_hidden_dim, latent_dim),
        )
        self.chemical_norm = nn.LayerNorm(latent_dim)

        cell_dim = 32
        domain_dim = 16
        self.cell_embedding = nn.Embedding(num_cell_lines + 1, cell_dim, padding_idx=0)
        self.domain_embedding = nn.Embedding(num_domains, domain_dim)
        if condition_mode == "fourier":
            condition_dim = 32
            self.condition_encoder = FourierConditionEncoder(
                condition_dim, dose_fourier_frequencies, dropout
            )
        elif condition_mode == "raw":
            condition_dim = 3
            self.condition_encoder = nn.Identity()
        else:
            raise ValueError(f"Unknown condition mode: {condition_mode}")
        self.condition_mode = condition_mode
        self.baseline_encoder = nn.Sequential(
            nn.LayerNorm(self.output_dim),
            nn.Linear(self.output_dim, baseline_hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(baseline_hidden_dim, latent_dim),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(cell_dim + domain_dim + condition_dim, 64),
            nn.SiLU(),
            nn.Linear(64, latent_dim),
        )
        self.context_norm = nn.LayerNorm(latent_dim)

        self.potency_predictor = nn.Sequential(
            nn.LayerNorm(latent_dim), nn.Linear(latent_dim, 2)
        )
        self.dose_slope_raw = nn.Parameter(torch.tensor(1.0))
        self.context_film = nn.Sequential(
            nn.LayerNorm(latent_dim), nn.Linear(latent_dim, latent_dim * 2)
        )
        self.fusion = nn.Sequential(
            *[ResidualBlock(latent_dim, dropout) for _ in range(fusion_depth)],
            nn.LayerNorm(latent_dim),
        )
        self.perturbation_decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, readout_dim),
        )
        self.interaction_decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, readout_dim),
        )
        self.readout_encoder = ReadoutEncoder(
            readout_features,
            readout_dim,
            dropout,
            use_metadata=use_readout_metadata,
        )
        if readout_dim % readout_attention_heads != 0:
            raise ValueError("readout_dim must be divisible by readout_attention_heads")
        if readout_attention_layers > 0:
            readout_layer = nn.TransformerEncoderLayer(
                d_model=readout_dim,
                nhead=readout_attention_heads,
                dim_feedforward=readout_dim * 2,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.readout_contextualizer = nn.TransformerEncoder(
                readout_layer,
                num_layers=readout_attention_layers,
                enable_nested_tensor=False,
            )
        else:
            self.readout_contextualizer = nn.Identity()
        self.perturbation_query = nn.Linear(latent_dim, readout_dim)
        self.context_query = nn.Linear(latent_dim, readout_dim)
        self.interaction_query = nn.Linear(latent_dim, readout_dim)
        self.condition_query = nn.Linear(condition_dim, readout_dim)
        self.query_norm = nn.LayerNorm(readout_dim)
        self.readout_predictor = nn.Sequential(
            nn.Linear(readout_dim, readout_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(readout_dim, 1),
        )
        self.distribution_head = (
            nn.Sequential(
                nn.Linear(readout_dim, readout_dim // 2),
                nn.GELU(),
                nn.Linear(readout_dim // 2, 1),
            )
            if self.enable_distribution_head
            else None
        )
        self.response_signature_head = (
            nn.Sequential(
                nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, latent_dim),
                nn.GELU(),
                nn.Linear(latent_dim, self.output_dim),
            )
            if self.enable_response_signature_head
            else None
        )
        self.target_auxiliary_head = (
            nn.Sequential(
                nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, latent_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(latent_dim, int(target_auxiliary_dim)),
            )
            if int(target_auxiliary_dim) > 0
            else None
        )
        self.learned_readout_logit = nn.Parameter(
            torch.tensor(float(learned_readout_initial_logit))
        )
        self.global_readout_bias = nn.Parameter(torch.zeros(self.output_dim))
        self.domain_readout_bias = nn.Parameter(torch.zeros(num_domains, self.output_dim))
        self.domain_readout_scale_raw = nn.Parameter(
            torch.zeros(num_domains, self.output_dim)
        )

    def forward(
        self,
        fingerprint: torch.Tensor,
        descriptors: torch.Tensor,
        cell_index: torch.Tensor,
        domain_index: torch.Tensor,
        context: torch.Tensor,
        baseline: torch.Tensor,
        readout_indices: torch.Tensor | None = None,
        return_aux: bool = False,
        atom_features: torch.Tensor | None = None,
        atom_mask: torch.Tensor | None = None,
        bond_types: torch.Tensor | None = None,
        zero_perturbation: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor | None]:
        if (
            self.chemical_encoder_mode == "ecfp"
            and self.training
            and self.fingerprint_dropout > 0
        ):
            keep = torch.rand_like(fingerprint) >= self.fingerprint_dropout
            fingerprint = fingerprint * keep
        if self.chemical_encoder_mode == "ecfp":
            chemical_features = self.chemical_encoder(fingerprint)
        else:
            if atom_features is None or atom_mask is None or bond_types is None:
                raise ValueError("Graph mode requires atom_features, atom_mask, and bond_types")
            chemical_features = self.chemical_encoder(
                atom_features, atom_mask, bond_types
            )
        perturbation = self.chemical_norm(
            chemical_features + self.descriptor_encoder(descriptors)
        )
        if zero_perturbation:
            perturbation = torch.zeros_like(perturbation)

        condition_input = context
        if not self.use_dose_time_conditioning:
            condition_input = context.clone()
            condition_input[:, :2] = 0.0
        condition_token = self.condition_encoder(condition_input)
        cell_embedding = self.cell_embedding(cell_index)
        if not self.use_cell_conditioning:
            cell_embedding = torch.zeros_like(cell_embedding)
        domain_embedding = self.domain_embedding(domain_index)
        if not self.use_domain_conditioning:
            domain_embedding = torch.zeros_like(domain_embedding)
        context_token = self.context_encoder(
            torch.cat(
                [
                    cell_embedding,
                    domain_embedding,
                    condition_token,
                ],
                dim=-1,
            )
        )
        baseline_gate = context[:, 2:3]
        if not self.use_baseline_conditioning:
            baseline_gate = torch.zeros_like(baseline_gate)
        baseline_token = self.baseline_encoder(baseline) * baseline_gate
        biological = self.context_norm(context_token + baseline_token)

        effective_perturbation = perturbation
        if self.decoder_mode == "gated_factorized":
            potency = self.potency_predictor(perturbation)
            midpoint = torch.tanh(potency[:, :1])
            slope = F.softplus(
                self.dose_slope_raw + 0.5 * torch.tanh(potency[:, 1:2])
            ) + 1e-3
            dose_gate = (
                torch.sigmoid(slope * (context[:, :1] - midpoint))
                if self.use_dose_time_conditioning
                else torch.ones_like(midpoint)
            )
            effective_perturbation = perturbation * dose_gate

        film_scale, film_shift = self.context_film(biological).chunk(2, dim=-1)
        conditioned = effective_perturbation * (
            1.0 + 0.5 * torch.tanh(film_scale)
        ) + film_shift
        interaction = self.fusion(conditioned)
        coefficients = self.perturbation_decoder(
            effective_perturbation
        ) + self.interaction_decoder(interaction)

        all_readouts = self.readout_encoder()
        all_readouts = self.readout_contextualizer(all_readouts.unsqueeze(0)).squeeze(0)
        readouts = all_readouts if readout_indices is None else all_readouts[readout_indices]
        if self.decoder_mode == "gated_factorized":
            query = self.query_norm(
                coefficients[:, None, :] * readouts[None, :, :]
                / math.sqrt(self.readout_dim)
            )
            learned_prediction = coefficients @ readouts.T / math.sqrt(
                self.readout_dim
            )
        else:
            sample_query = (
                self.perturbation_query(perturbation)
                + self.context_query(biological)
                + self.interaction_query(interaction)
                + self.condition_query(condition_token)
            )
            query = self.query_norm(
                sample_query[:, None, :]
                + readouts[None, :, :]
                + sample_query[:, None, :] * readouts[None, :, :]
                / math.sqrt(self.readout_dim)
            )
            learned_prediction = self.readout_predictor(query).squeeze(-1)
        prediction = torch.sigmoid(self.learned_readout_logit) * learned_prediction
        if self.response_basis is not None:
            basis = (
                self.response_basis
                if readout_indices is None
                else self.response_basis[:, readout_indices]
            )
            prediction = prediction + coefficients @ basis
        if readout_indices is None:
            global_bias = self.global_readout_bias
            if self.use_domain_conditioning:
                domain_bias = self.domain_readout_bias[domain_index]
                domain_scale = 1.0 + 0.25 * torch.tanh(
                    self.domain_readout_scale_raw[domain_index]
                )
            else:
                domain_bias = torch.zeros_like(prediction)
                domain_scale = torch.ones_like(prediction)
        else:
            global_bias = self.global_readout_bias[readout_indices]
            if self.use_domain_conditioning:
                domain_bias = self.domain_readout_bias[domain_index][:, readout_indices]
                domain_scale = 1.0 + 0.25 * torch.tanh(
                    self.domain_readout_scale_raw[domain_index][:, readout_indices]
                )
            else:
                domain_bias = torch.zeros_like(prediction)
                domain_scale = torch.ones_like(prediction)
        prediction = prediction * domain_scale + global_bias + domain_bias
        if not return_aux:
            return prediction
        predicted_log_std = (
            self.distribution_head(query).squeeze(-1).clamp(-6.0, 3.0)
            if self.distribution_head is not None
            else None
        )
        response_signature = (
            self.response_signature_head(perturbation)
            if self.response_signature_head is not None
            else None
        )
        return {
            "mean": prediction,
            "log_std": predicted_log_std,
            "response_signature": response_signature,
            "chemical_embedding": perturbation,
            "target_gene_logits": (
                self.target_auxiliary_head(perturbation)
                if self.target_auxiliary_head is not None
                else None
            ),
        }


def masked_reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
    loss_type: str = "huber",
    huber_delta: float = 1.0,
) -> torch.Tensor:
    mask = mask.bool()
    if loss_type == "huber":
        loss = F.huber_loss(prediction, target, reduction="none", delta=huber_delta)
    elif loss_type == "mse":
        loss = F.mse_loss(prediction, target, reduction="none")
    else:
        raise ValueError(f"Unknown reconstruction loss: {loss_type}")
    weights = mask.float()
    if sample_weight is not None:
        weights = weights * sample_weight[:, None]
    return (loss * weights).sum() / weights.sum().clamp_min(1.0)


def masked_correlation_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    mask = mask.bool()
    weights = mask.float()
    counts = weights.sum(dim=1).clamp_min(1.0)
    pred_mean = (prediction * weights).sum(dim=1) / counts
    target_mean = (target * weights).sum(dim=1) / counts
    pred_centered = (prediction - pred_mean[:, None]) * weights
    target_centered = (target - target_mean[:, None]) * weights
    numerator = (pred_centered * target_centered).sum(dim=1)
    denominator = torch.sqrt(
        pred_centered.square().sum(dim=1) * target_centered.square().sum(dim=1)
    ).clamp_min(1e-8)
    loss = 1.0 - numerator / denominator
    valid = mask.sum(dim=1) >= 2
    if sample_weight is None:
        return loss[valid].mean()
    effective_weight = sample_weight[valid]
    return (loss[valid] * effective_weight).sum() / effective_weight.sum().clamp_min(1e-8)


def masked_topk_reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    masked_magnitude = target.abs().masked_fill(~mask.bool(), -float("inf"))
    count = min(int(top_k), target.shape[1])
    indices = masked_magnitude.topk(count, dim=1).indices
    selected_prediction = prediction.gather(1, indices)
    selected_target = target.gather(1, indices)
    selected_mask = mask.gather(1, indices)
    return masked_reconstruction_loss(
        selected_prediction,
        selected_target,
        selected_mask,
        loss_type="huber",
    )


def masked_response_norm_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    weights = mask.float()
    predicted_norm = torch.sqrt((prediction.square() * weights).sum(dim=1) + 1e-8)
    target_norm = torch.sqrt((target.square() * weights).sum(dim=1) + 1e-8)
    return F.smooth_l1_loss(torch.log1p(predicted_norm), torch.log1p(target_norm))


def response_signature_loss(
    prediction: torch.Tensor | None,
    target: torch.Tensor,
    available: torch.Tensor,
) -> torch.Tensor:
    if prediction is None:
        return target.new_zeros(())
    valid = available.bool()
    if not valid.any():
        return target.new_zeros(())
    return (1.0 - F.cosine_similarity(prediction[valid], target[valid], dim=1)).mean()


def distribution_moment_loss(
    predicted_log_std: torch.Tensor | None,
    target_std: torch.Tensor,
    target_std_mask: torch.Tensor,
) -> torch.Tensor:
    if predicted_log_std is None or not target_std_mask.any():
        return target_std.new_zeros(())
    target = torch.log(target_std.clamp_min(1e-5))
    return masked_reconstruction_loss(
        predicted_log_std,
        target,
        target_std_mask,
        loss_type="huber",
    )


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    valid = labels > 0
    if valid.sum() < 2:
        return embeddings.new_zeros(())
    embeddings = F.normalize(embeddings[valid], dim=1)
    labels = labels[valid]
    logits = embeddings @ embeddings.T / temperature
    identity = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    positives = labels[:, None].eq(labels[None, :]) & ~identity
    anchors = positives.any(dim=1)
    if not anchors.any():
        return embeddings.new_zeros(())
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits).masked_fill(identity, 0.0)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-8))
    positive_log_prob = (log_prob * positives).sum(dim=1) / positives.sum(dim=1).clamp_min(1)
    return -positive_log_prob[anchors].mean()


def target_gene_auxiliary_loss(
    logits: torch.Tensor | None,
    targets: torch.Tensor,
    available: torch.Tensor,
    positive_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    if logits is None:
        return targets.new_zeros(())
    valid = available.bool()
    if not valid.any() or logits.shape[1] == 0:
        return targets.new_zeros(())
    if logits.shape != targets.shape:
        raise ValueError("Target-gene logits and labels must have matching shapes")
    return F.binary_cross_entropy_with_logits(
        logits[valid],
        targets[valid],
        pos_weight=positive_weight,
    )
