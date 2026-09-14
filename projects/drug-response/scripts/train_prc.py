#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from my_star.chemistry import descriptor_dimension
from my_star.prc_data import (
    build_context_maps,
    build_stage_datasets,
    collate_prc_batch,
    fit_masked_response_basis,
    fit_shared_response_basis,
    load_prc_observations,
    load_readout_features,
)
from my_star.prc_metrics import (
    calibrate_masked_residual_scale,
    masked_regression_metrics,
)
from my_star.prc_model import (
    PRCFactorizedModel,
    distribution_moment_loss,
    masked_correlation_loss,
    masked_reconstruction_loss,
    masked_response_norm_loss,
    masked_topk_reconstruction_loss,
    response_signature_loss,
    supervised_contrastive_loss,
    target_gene_auxiliary_loss,
)


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(values: np.ndarray | None) -> str | None:
    if values is None:
        return None
    contiguous = np.ascontiguousarray(values)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def load_transfer_state(
    model: nn.Module,
    checkpoint: dict,
    prefixes: list[str] | None,
) -> list[str]:
    if not prefixes:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        return list(checkpoint["model_state_dict"])
    current = model.state_dict()
    selected = {
        key: value
        for key, value in checkpoint["model_state_dict"].items()
        if any(key == prefix or key.startswith(f"{prefix}.") for prefix in prefixes)
        and key in current
        and current[key].shape == value.shape
    }
    if not selected:
        raise ValueError("No checkpoint parameters matched initialize_prefixes")
    model.load_state_dict(selected, strict=False)
    return sorted(selected)


def freeze_parameter_prefixes(model: nn.Module, prefixes: list[str] | None) -> list[str]:
    if not prefixes:
        return []
    frozen = []
    for name, parameter in model.named_parameters():
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes):
            parameter.requires_grad_(False)
            frozen.append(name)
    if not frozen:
        raise ValueError("No model parameters matched freeze_prefixes")
    return sorted(frozen)


def restrict_trainable_parameter_prefixes(
    model: nn.Module, prefixes: list[str] | None
) -> tuple[list[str], list[str]]:
    if not prefixes:
        return [], sorted(name for name, _ in model.named_parameters())
    frozen, trainable = [], []
    for name, parameter in model.named_parameters():
        selected = any(
            name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes
        )
        parameter.requires_grad_(selected)
        (trainable if selected else frozen).append(name)
    if not trainable:
        raise ValueError("No model parameters matched trainable_prefixes")
    return sorted(frozen), sorted(trainable)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


def model_forward(
    model: nn.Module,
    batch: dict,
    return_aux: bool = False,
    zero_perturbation: bool = False,
) -> torch.Tensor | dict[str, torch.Tensor | None]:
    return model(
        batch["fingerprint"],
        batch["descriptors"],
        batch["cell_index"],
        batch["domain_index"],
        batch["context"],
        batch["baseline"],
        return_aux=return_aux,
        atom_features=batch.get("atom_features"),
        atom_mask=batch.get("atom_mask"),
        bond_types=batch.get("bond_types"),
        zero_perturbation=zero_perturbation,
    )


def build_training_sampler(
    dataset,
    balance_domains: bool,
    balance_compounds: bool,
    samples_per_epoch: int | None,
    seed: int,
) -> WeightedRandomSampler | None:
    if not balance_domains and not balance_compounds and samples_per_epoch is None:
        return None
    weights = np.ones(len(dataset), dtype=np.float64)
    if balance_domains:
        labels = np.asarray(dataset.domain_labels)
        count_map = dict(zip(*np.unique(labels, return_counts=True)))
        weights *= np.asarray(
            [1.0 / count_map[label] for label in labels], dtype=np.float64
        )
    if balance_compounds:
        compound_ids = dataset.frame["compound_id"].astype(str).to_numpy()
        count_map = dict(zip(*np.unique(compound_ids, return_counts=True)))
        weights *= np.asarray(
            [1.0 / count_map[value] for value in compound_ids], dtype=np.float64
        )
    weights /= weights.mean()
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        torch.from_numpy(weights),
        num_samples=int(samples_per_epoch or len(dataset)),
        replacement=True,
        generator=generator,
    )


def optimizer_parameter_groups(
    model: nn.Module,
    base_learning_rate: float,
    prefix_multipliers: dict[str, float] | None,
) -> list[dict]:
    if not prefix_multipliers:
        return [
            {
                "params": [
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ],
                "lr": base_learning_rate,
                "group_name": "default",
            }
        ]
    normalized = {str(key): float(value) for key, value in prefix_multipliers.items()}
    if any(not np.isfinite(value) or value <= 0.0 for value in normalized.values()):
        raise ValueError("Learning-rate multipliers must be finite and positive")
    grouped: dict[tuple[str, float], list[nn.Parameter]] = {}
    sorted_prefixes = sorted(normalized, key=len, reverse=True)
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        matches = [
            prefix
            for prefix in sorted_prefixes
            if name == prefix or name.startswith(f"{prefix}.")
        ]
        prefix = matches[0] if matches else "default"
        multiplier = normalized[prefix] if matches else 1.0
        grouped.setdefault((prefix, multiplier), []).append(parameter)
    missing = [
        prefix
        for prefix in normalized
        if not any(
            name == prefix or name.startswith(f"{prefix}.")
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        )
    ]
    if missing:
        raise ValueError(
            f"Learning-rate multiplier prefixes matched no trainable parameters: {missing}"
        )
    ordered = sorted(
        grouped.items(),
        key=lambda item: (item[0][0] != "default", item[0][0]),
    )
    return [
        {
            "params": parameters,
            "lr": base_learning_rate * multiplier,
            "group_name": prefix,
        }
        for (prefix, multiplier), parameters in ordered
    ]


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    scaler,
    top_k: int,
    loss_type: str,
    huber_delta: float,
    correlation_weight: float,
    calibration_metric: str,
    calibration_maximum: float,
    calibration_steps: int,
    zero_perturbation: bool = False,
    fixed_residual_scale: float | None = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    model.eval()
    scale = torch.as_tensor(scaler.scale, device=device)
    mean = torch.as_tensor(scaler.mean, device=device)
    predictions, targets, masks, priors = [], [], [], []
    sample_ids, compound_ids = [], []
    losses = []
    for batch in loader:
        batch = move_batch(batch, device)
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            prediction = model_forward(
                model, batch, zero_perturbation=zero_perturbation
            )
            point_loss = masked_reconstruction_loss(
                prediction,
                batch["target"],
                batch["target_mask"],
                loss_type=loss_type,
                huber_delta=huber_delta,
            )
            prediction_raw = prediction * scale + mean + batch["context_prior"]
            corr_loss = masked_correlation_loss(
                prediction_raw, batch["target_raw"], batch["target_mask"]
            )
            losses.append(float((point_loss + correlation_weight * corr_loss).item()))
        predictions.append(prediction_raw.float().cpu().numpy())
        targets.append(batch["target_raw"].float().cpu().numpy())
        masks.append(batch["target_mask"].cpu().numpy())
        priors.append(batch["context_prior"].float().cpu().numpy())
        sample_ids.extend(batch["sample_id"])
        compound_ids.extend(batch["compound_id"])

    y_pred = np.vstack(predictions)
    y_true = np.vstack(targets)
    mask = np.vstack(masks).astype(bool)
    prior = np.vstack(priors)
    uncalibrated_metrics = masked_regression_metrics(y_true, y_pred, mask, top_k=top_k)
    if fixed_residual_scale is None:
        residual_scale, metrics = calibrate_masked_residual_scale(
            y_true,
            y_pred,
            prior,
            mask,
            top_k=top_k,
            metric=calibration_metric,
            maximum=calibration_maximum,
            steps=calibration_steps,
        )
    else:
        residual_scale = float(fixed_residual_scale)
        fixed_prediction = prior + residual_scale * (y_pred - prior)
        metrics = masked_regression_metrics(
            y_true, fixed_prediction, mask, top_k=top_k
        )
        metrics["residual_scale"] = residual_scale
    calibrated = prior + residual_scale * (y_pred - prior)
    metrics["standardized_loss"] = float(np.mean(losses))
    prior_metrics = masked_regression_metrics(y_true, prior, mask, top_k=top_k)
    output = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "compound_id": compound_ids,
            "target_delta_y": list(y_true.astype(np.float32)),
            "target_mask": list(mask),
            "context_prior_delta_y": list(prior.astype(np.float32)),
            "uncalibrated_predicted_delta_y": list(y_pred.astype(np.float32)),
            "predicted_delta_y": list(calibrated.astype(np.float32)),
        }
    )
    return {
        **metrics,
        **{f"uncalibrated_{k}": v for k, v in uncalibrated_metrics.items()},
        **{f"context_prior_{k}": v for k, v in prior_metrics.items()},
    }, output


def compound_derangement(compound_ids: list[str], seed: int) -> dict[str, str]:
    values = sorted(set(compound_ids))
    if len(values) < 2:
        raise ValueError("Chemical permutation requires at least two compounds")
    rng = np.random.default_rng(seed)
    shuffled = values.copy()
    rng.shuffle(shuffled)
    if any(left == right for left, right in zip(values, shuffled)):
        shift = int(rng.integers(1, len(values)))
        shuffled = values[shift:] + values[:shift]
    if any(left == right for left, right in zip(values, shuffled)):
        raise RuntimeError("Failed to construct a chemical derangement")
    return dict(zip(values, shuffled))


def checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR,
    epoch: int,
    best_score: float,
    stale_epochs: int,
    history: list[dict],
    config: dict,
    stage_name: str,
    stage_config: dict,
    scaler,
    prior,
    cell_map: dict[str, int],
    domain_map: dict[str, int],
    dataset_path: Path,
    split_path: Path,
    residual_scale: float,
    response_basis_sha256: str | None,
    response_basis_source: dict | None,
    mechanism_supervision_sha256: str | None,
    target_gene_map: dict[str, int],
) -> dict:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "best_validation_score": best_score,
        "stale_epochs": stale_epochs,
        "history": history,
        "config": config,
        "stage_name": stage_name,
        "stage_config": stage_config,
        "target_scaler": scaler.to_dict(),
        "context_prior": prior.to_dict(),
        "response_basis": (
            model.response_basis.detach().cpu()
            if getattr(model, "response_basis", None) is not None
            else None
        ),
        "response_basis_sha256": response_basis_sha256,
        "response_basis_source": response_basis_source,
        "cell_map": cell_map,
        "domain_map": domain_map,
        "data_sha256": file_sha256(dataset_path),
        "split_sha256": file_sha256(split_path),
        "mechanism_supervision_sha256": mechanism_supervision_sha256,
        "target_gene_map": target_gene_map,
        "residual_scale": residual_scale,
    }


def train_stage(
    config: dict,
    stage_name: str,
    run_name: str,
    initialize_from: Path | None = None,
    split_override: Path | None = None,
    epochs_override: int | None = None,
) -> Path:
    stage = dict(config[stage_name])
    if split_override is not None:
        stage["split_path"] = str(split_override)
    if epochs_override is not None:
        stage["epochs"] = int(epochs_override)
    seed = int(stage.get("seed", config["seed"]))
    set_seed(seed)
    if bool(config.get("require_cuda", True)) and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Phase 5 training; run this stage on the GPU host")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        memory_fraction = float(config.get("cuda_memory_fraction", 1.0))
        if not 0.0 < memory_fraction <= 1.0:
            raise ValueError("cuda_memory_fraction must be in (0, 1]")
        torch.cuda.set_per_process_memory_fraction(memory_fraction, device=0)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    dataset_path = project_path(stage["dataset_path"])
    split_path = project_path(stage["split_path"])
    panel_path = project_path(config["gene_panel_path"])
    observations = load_prc_observations(dataset_path)
    mechanism_path_value = config.get("mechanism_supervision_path")
    mechanism_path = (
        project_path(mechanism_path_value) if mechanism_path_value else None
    )
    mechanism_supervision = (
        pd.read_parquet(mechanism_path) if mechanism_path is not None else None
    )
    mechanism_supervision_sha256 = (
        file_sha256(mechanism_path) if mechanism_path is not None else None
    )
    split_manifest = json.loads(split_path.read_text(encoding="utf-8"))
    fixed_epoch_training = bool(stage.get("fixed_epoch_training", False))
    context_paths = [project_path(path) for path in config["context_dataset_paths"]]
    cell_map, domain_map = build_context_maps(context_paths)
    external_readout_path = config.get("readout_feature_path")
    readout_features, panel = load_readout_features(
        panel_path,
        project_path(external_readout_path) if external_readout_path else None,
    )
    descriptor_set = str(config.get("descriptor_set", "extended"))
    datasets, scaler, prior = build_stage_datasets(
        observations,
        split_manifest,
        cell_map,
        domain_map,
        target_scaling=str(stage.get("target_scaling", "global")),
        descriptor_set=descriptor_set,
        fingerprint_radius=int(config.get("fingerprint_radius", 2)),
        fingerprint_use_counts=bool(config.get("fingerprint_use_counts", True)),
        fingerprint_include_chirality=bool(
            config.get("fingerprint_include_chirality", True)
        ),
        include_validation=not fixed_epoch_training,
        mechanism_supervision=mechanism_supervision,
        target_auxiliary_label_column=str(
            stage.get(
                "target_auxiliary_label_column",
                "specific_target_gene_symbols",
            )
        ),
        target_auxiliary_min_compounds=int(
            stage.get("target_auxiliary_min_compounds", 1)
        ),
        chemical_encoder_mode=str(config.get("chemical_encoder_mode", "ecfp")),
    )
    response_rank = int(stage.get("response_rank", config.get("readout_dim", 64)))
    shared_basis = config.get("shared_response_basis")
    response_basis_source = None
    if shared_basis:
        basis_dataset_path = project_path(shared_basis["dataset_path"])
        basis_split_path = project_path(shared_basis["split_path"])
        basis_observations = load_prc_observations(basis_dataset_path)
        basis_split = json.loads(basis_split_path.read_text(encoding="utf-8"))
        basis_rank = int(shared_basis.get("rank", response_rank))
        if basis_rank != response_rank:
            raise ValueError("Shared response-basis rank must match the active stage rank")
        response_basis_array = fit_shared_response_basis(
            basis_observations,
            basis_split,
            target_scaling=str(shared_basis.get("target_scaling", "global")),
            rank=basis_rank,
        )
        response_basis_source = {
            "dataset_path": str(basis_dataset_path.relative_to(ROOT)),
            "split_path": str(basis_split_path.relative_to(ROOT)),
            "dataset_sha256": file_sha256(basis_dataset_path),
            "split_sha256": file_sha256(basis_split_path),
            "target_scaling": str(shared_basis.get("target_scaling", "global")),
        }
    else:
        response_basis_array = fit_masked_response_basis(datasets["train"], response_rank)
    response_basis_sha256 = array_sha256(response_basis_array)
    response_basis = (
        torch.from_numpy(response_basis_array) if response_basis_array is not None else None
    )
    batch_size = int(stage["batch_size"])
    num_workers = int(stage.get("num_workers", 0))
    train_sampler = build_training_sampler(
        datasets["train"],
        balance_domains=bool(stage.get("balance_domains", False)),
        balance_compounds=bool(stage.get("balance_compounds", False)),
        samples_per_epoch=(
            int(stage["samples_per_epoch"])
            if stage.get("samples_per_epoch") is not None
            else None
        ),
        seed=seed,
    )
    loaders = {
        name: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=name == "train" and train_sampler is None,
            sampler=train_sampler if name == "train" else None,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=num_workers > 0,
            drop_last=False,
            collate_fn=collate_prc_batch,
        )
        for name, dataset in datasets.items()
    }
    model_overrides = stage.get("model_overrides", {})
    if not isinstance(model_overrides, dict):
        raise ValueError("model_overrides must be a mapping")
    allowed_model_overrides = {
        "condition_mode",
        "use_cell_conditioning",
        "use_domain_conditioning",
        "use_dose_time_conditioning",
        "use_baseline_conditioning",
        "use_readout_metadata",
    }
    unknown_model_overrides = set(model_overrides) - allowed_model_overrides
    if unknown_model_overrides:
        raise ValueError(
            f"Unknown stage model overrides: {sorted(unknown_model_overrides)}"
        )

    def model_value(name: str, default):
        return model_overrides.get(name, config.get(name, default))

    model = PRCFactorizedModel(
        readout_features=torch.from_numpy(readout_features),
        num_cell_lines=len(cell_map),
        num_domains=len(domain_map),
        response_basis=response_basis,
        descriptor_input_dim=descriptor_dimension(descriptor_set),
        latent_dim=int(config.get("latent_dim", 128)),
        readout_dim=int(config.get("readout_dim", 96)),
        fingerprint_hidden_dim=int(config.get("fingerprint_hidden_dim", 256)),
        baseline_hidden_dim=int(config.get("baseline_hidden_dim", 192)),
        descriptor_hidden_dim=int(config.get("descriptor_hidden_dim", 48)),
        fusion_depth=int(config.get("fusion_depth", 1)),
        dropout=float(config.get("dropout", 0.15)),
        fingerprint_dropout=float(config.get("fingerprint_dropout", 0.05)),
        learned_readout_initial_logit=float(
            config.get("learned_readout_initial_logit", -5.0)
        ),
        dose_fourier_frequencies=int(config.get("dose_fourier_frequencies", 4)),
        readout_attention_heads=int(config.get("readout_attention_heads", 4)),
        readout_attention_layers=int(config.get("readout_attention_layers", 1)),
        enable_distribution_head=bool(config.get("enable_distribution_head", False)),
        enable_response_signature_head=bool(
            config.get("enable_response_signature_head", False)
        ),
        target_auxiliary_dim=len(datasets["train"].target_gene_map),
        decoder_mode=str(config.get("decoder_mode", "query_attention")),
        condition_mode=str(model_value("condition_mode", "fourier")),
        chemical_encoder_mode=str(config.get("chemical_encoder_mode", "ecfp")),
        graph_message_layers=int(config.get("graph_message_layers", 3)),
        use_cell_conditioning=bool(model_value("use_cell_conditioning", True)),
        use_domain_conditioning=bool(
            model_value("use_domain_conditioning", True)
        ),
        use_dose_time_conditioning=bool(
            model_value("use_dose_time_conditioning", True)
        ),
        use_baseline_conditioning=bool(
            model_value("use_baseline_conditioning", True)
        ),
        use_readout_metadata=bool(model_value("use_readout_metadata", True)),
    ).to(device)
    freeze_prefixes = stage.get("freeze_prefixes")
    if freeze_prefixes is not None and not isinstance(freeze_prefixes, list):
        raise ValueError("freeze_prefixes must be a list")
    trainable_prefixes = stage.get("trainable_prefixes")
    if trainable_prefixes is not None and not isinstance(trainable_prefixes, list):
        raise ValueError("trainable_prefixes must be a list")
    if freeze_prefixes and trainable_prefixes:
        raise ValueError("freeze_prefixes and trainable_prefixes are mutually exclusive")
    if trainable_prefixes:
        frozen_parameters, explicitly_trainable = restrict_trainable_parameter_prefixes(
            model, trainable_prefixes
        )
    else:
        frozen_parameters = freeze_parameter_prefixes(model, freeze_prefixes)
        explicitly_trainable = sorted(
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        )
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise ValueError("Training requires at least one trainable parameter")
    base_learning_rate = float(stage["learning_rate"])
    learning_rate_multipliers = stage.get("learning_rate_multipliers")
    if learning_rate_multipliers is not None and not isinstance(
        learning_rate_multipliers, dict
    ):
        raise ValueError("learning_rate_multipliers must be a mapping")
    parameter_groups = optimizer_parameter_groups(
        model,
        base_learning_rate,
        learning_rate_multipliers,
    )
    optimizer = AdamW(
        parameter_groups,
        lr=base_learning_rate,
        weight_decay=float(stage["weight_decay"]),
    )
    epochs = int(stage["epochs"])
    scheduler_epochs = int(stage.get("scheduler_epochs", epochs))
    if scheduler_epochs < epochs:
        raise ValueError("scheduler_epochs must be greater than or equal to epochs")
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=scheduler_epochs,
        eta_min=float(stage.get("minimum_learning_rate", 1e-6)),
    )
    output_dir = project_path(config.get("output_dir", "artifacts")) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    last_path = output_dir / "last_checkpoint.pt"
    best_path = output_dir / "best_model.pt"
    completed_path = output_dir / "metrics.json"
    if completed_path.exists() and best_path.exists():
        print(f"stage_already_complete={output_dir}", flush=True)
        return output_dir

    start_epoch = 1
    history: list[dict] = []
    best_score = -float("inf")
    stale_epochs = 0
    if last_path.exists():
        checkpoint = torch.load(last_path, map_location=device, weights_only=True)
        if checkpoint["data_sha256"] != file_sha256(dataset_path):
            raise ValueError("Resume checkpoint dataset hash mismatch")
        if checkpoint["split_sha256"] != file_sha256(split_path):
            raise ValueError("Resume checkpoint split hash mismatch")
        if checkpoint.get("response_basis_sha256") != response_basis_sha256:
            raise ValueError("Resume checkpoint response-basis hash mismatch")
        if checkpoint.get("mechanism_supervision_sha256") != mechanism_supervision_sha256:
            raise ValueError("Resume checkpoint mechanism-supervision hash mismatch")
        if checkpoint.get("target_gene_map", {}) != datasets["train"].target_gene_map:
            raise ValueError("Resume checkpoint target-gene vocabulary mismatch")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        history = list(checkpoint["history"])
        best_score = float(checkpoint["best_validation_score"])
        stale_epochs = int(checkpoint["stale_epochs"])
        print(f"resumed={last_path} start_epoch={start_epoch}", flush=True)
    elif initialize_from is not None:
        checkpoint = torch.load(initialize_from, map_location=device, weights_only=True)
        initialize_prefixes = stage.get("initialize_prefixes")
        if initialize_prefixes is not None and not isinstance(initialize_prefixes, list):
            raise ValueError("initialize_prefixes must be a list of module prefixes")
        if "cell_map" in checkpoint and "domain_map" in checkpoint:
            if checkpoint["cell_map"] != cell_map or checkpoint["domain_map"] != domain_map:
                raise ValueError("Pretraining checkpoint context vocabulary mismatch")
        elif checkpoint.get("checkpoint_type") == "chemical_pretraining":
            chemical_prefixes = {
                "chemical_encoder",
                "descriptor_encoder",
                "chemical_norm",
            }
            if not initialize_prefixes or not set(initialize_prefixes).issubset(
                chemical_prefixes
            ):
                raise ValueError(
                    "Chemical-only checkpoints may initialize only P encoder modules"
                )
        else:
            raise ValueError("Transfer checkpoint has no context vocabulary")
        if not initialize_prefixes and checkpoint.get("response_basis_sha256") != response_basis_sha256:
            raise ValueError(
                "Transfer checkpoint uses a different response basis; coefficient axes are incompatible"
            )
        loaded_keys = load_transfer_state(model, checkpoint, initialize_prefixes)
        print(
            f"initialized_from={initialize_from} loaded_parameters={len(loaded_keys)} "
            f"prefixes={initialize_prefixes or ['*']}",
            flush=True,
        )

    if start_epoch == 1 and bool(
        config.get("reset_training_rng_after_initialization", False)
    ):
        set_seed(seed + 1_000_003)

    use_amp = device.type == "cuda"
    grad_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    scale_tensor = torch.as_tensor(scaler.scale, device=device)
    mean_tensor = torch.as_tensor(scaler.mean, device=device)
    correlation_weight = float(stage.get("correlation_loss_weight", 0.3))
    loss_type = str(stage.get("reconstruction_loss", "huber"))
    huber_delta = float(stage.get("huber_delta", 1.0))
    top_k = int(config.get("top_deg_count", 50))
    selection_metric = str(stage.get("selection_metric", "mean_sample_pearson"))
    patience = int(stage.get("early_stopping_patience", epochs))
    use_sample_weights = bool(stage.get("use_sample_weights", False))
    topk_loss_weight = float(stage.get("topk_loss_weight", 0.0))
    topk_loss_count = int(stage.get("topk_loss_count", top_k))
    response_norm_loss_weight = float(stage.get("response_norm_loss_weight", 0.0))
    signature_loss_weight = float(stage.get("signature_loss_weight", 0.0))
    distribution_loss_weight = float(stage.get("distribution_loss_weight", 0.0))
    pathway_contrastive_weight = float(
        stage.get("pathway_contrastive_weight", 0.0)
    )
    target_auxiliary_weight = float(stage.get("target_auxiliary_weight", 0.0))
    if target_auxiliary_weight > 0 and not datasets["train"].target_gene_map:
        raise ValueError(
            "target_auxiliary_weight is positive but no train-only target vocabulary exists"
        )
    target_positive_weight = torch.as_tensor(
        datasets["train"].target_positive_weights, device=device
    )
    contrastive_temperature = float(stage.get("contrastive_temperature", 0.1))
    started = time.time()
    print(
        json.dumps(
            {
                "stage": stage_name,
                "device": str(device),
                "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                "train_rows": len(datasets["train"]),
                "validation_rows": (
                    len(datasets["validation"]) if "validation" in datasets else 0
                ),
                "fixed_epoch_training": fixed_epoch_training,
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "trainable_parameters": sum(
                    parameter.numel() for parameter in trainable_parameters
                ),
                "frozen_parameter_tensors": len(frozen_parameters),
                "explicitly_trainable_parameter_tensors": len(explicitly_trainable),
                "start_epoch": start_epoch,
                "epochs": epochs,
                "scheduler_epochs": scheduler_epochs,
                "target_auxiliary_genes": len(datasets["train"].target_gene_map),
                "target_auxiliary_compounds": datasets[
                    "train"
                ].target_label_compound_count,
                "target_auxiliary_weight": target_auxiliary_weight,
                "target_auxiliary_label_column": str(
                    stage.get(
                        "target_auxiliary_label_column",
                        "specific_target_gene_symbols",
                    )
                ),
                "target_auxiliary_min_compounds": int(
                    stage.get("target_auxiliary_min_compounds", 1)
                ),
                "balance_compounds": bool(stage.get("balance_compounds", False)),
                "model_overrides": model_overrides,
                "learning_rate_groups": {
                    str(group.get("group_name", index)): float(group["lr"])
                    for index, group in enumerate(optimizer.param_groups)
                },
            }
        ),
        flush=True,
    )

    for epoch in range(start_epoch, epochs + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model.train()
        epoch_losses, point_losses, correlation_losses = [], [], []
        topk_losses, norm_losses, signature_losses = [], [], []
        distribution_losses, contrastive_losses, target_auxiliary_losses = [], [], []
        for batch in loaders["train"]:
            batch = move_batch(batch, device)
            loss_sample_weight = batch["sample_weight"] if use_sample_weights else None
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                output = model_forward(model, batch, return_aux=True)
                if not isinstance(output, dict):
                    raise TypeError("Auxiliary model forward must return a dictionary")
                prediction = output["mean"]
                if not isinstance(prediction, torch.Tensor):
                    raise TypeError("Model mean prediction must be a tensor")
                point_loss = masked_reconstruction_loss(
                    prediction,
                    batch["target"],
                    batch["target_mask"],
                    sample_weight=loss_sample_weight,
                    loss_type=loss_type,
                    huber_delta=huber_delta,
                )
                prediction_raw = (
                    prediction * scale_tensor
                    + mean_tensor
                    + batch["context_prior"]
                )
                corr_loss = masked_correlation_loss(
                    prediction_raw,
                    batch["target_raw"],
                    batch["target_mask"],
                    sample_weight=loss_sample_weight,
                )
                topk_loss = masked_topk_reconstruction_loss(
                    prediction,
                    batch["target"],
                    batch["target_mask"],
                    top_k=topk_loss_count,
                )
                norm_loss = masked_response_norm_loss(
                    prediction,
                    batch["target"],
                    batch["target_mask"],
                )
                signature_loss = response_signature_loss(
                    output["response_signature"],
                    batch["response_signature"],
                    batch["response_signature_available"],
                )
                moment_loss = distribution_moment_loss(
                    output["log_std"],
                    batch["target_std"],
                    batch["target_std_mask"],
                )
                chemical_embedding = output["chemical_embedding"]
                if not isinstance(chemical_embedding, torch.Tensor):
                    raise TypeError("Chemical embedding must be a tensor")
                contrastive_loss = supervised_contrastive_loss(
                    chemical_embedding,
                    batch["pathway_index"],
                    temperature=contrastive_temperature,
                )
                target_aux_loss = target_gene_auxiliary_loss(
                    output["target_gene_logits"],
                    batch["target_gene_labels"],
                    batch["target_gene_labels_available"],
                    positive_weight=target_positive_weight,
                )
                loss = (
                    point_loss
                    + correlation_weight * corr_loss
                    + topk_loss_weight * topk_loss
                    + response_norm_loss_weight * norm_loss
                    + signature_loss_weight * signature_loss
                    + distribution_loss_weight * moment_loss
                    + pathway_contrastive_weight * contrastive_loss
                    + target_auxiliary_weight * target_aux_loss
                )
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), float(stage.get("max_grad_norm", 5.0)))
            grad_scaler.step(optimizer)
            grad_scaler.update()
            epoch_losses.append(float(loss.item()))
            point_losses.append(float(point_loss.item()))
            correlation_losses.append(float(corr_loss.item()))
            topk_losses.append(float(topk_loss.item()))
            norm_losses.append(float(norm_loss.item()))
            signature_losses.append(float(signature_loss.item()))
            distribution_losses.append(float(moment_loss.item()))
            contrastive_losses.append(float(contrastive_loss.item()))
            target_auxiliary_losses.append(float(target_aux_loss.item()))
        scheduler.step()

        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(epoch_losses)),
            "train_point_loss": float(np.mean(point_losses)),
            "train_correlation_loss": float(np.mean(correlation_losses)),
            "train_topk_loss": float(np.mean(topk_losses)),
            "train_response_norm_loss": float(np.mean(norm_losses)),
            "train_signature_loss": float(np.mean(signature_losses)),
            "train_distribution_loss": float(np.mean(distribution_losses)),
            "train_pathway_contrastive_loss": float(np.mean(contrastive_losses)),
            "train_target_auxiliary_loss": float(np.mean(target_auxiliary_losses)),
            "learning_rate": float(max(scheduler.get_last_lr())),
            "learning_rates": {
                str(group.get("group_name", index)): float(rate)
                for index, (group, rate) in enumerate(
                    zip(optimizer.param_groups, scheduler.get_last_lr())
                )
            },
            "cuda_peak_allocated_gb": (
                float(torch.cuda.max_memory_allocated(device) / 1024**3)
                if device.type == "cuda"
                else 0.0
            ),
        }
        if fixed_epoch_training:
            metrics = None
            predictions = None
        else:
            metrics, predictions = evaluate(
                model,
                loaders["validation"],
                device,
                scaler,
                top_k,
                loss_type,
                huber_delta,
                correlation_weight,
                str(stage.get("calibration_metric", selection_metric)),
                float(stage.get("calibration_maximum", 1.5)),
                int(stage.get("calibration_steps", 21)),
            )
            record.update(
                {f"validation_{key}": value for key, value in metrics.items()}
            )
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if fixed_epoch_training:
            improved = epoch == epochs
            stale_epochs = 0
            residual_scale = float(stage.get("fixed_member_residual_scale", 1.0))
        else:
            assert metrics is not None
            selection_value = float(metrics[selection_metric])
            improved = selection_value > best_score + float(
                stage.get("minimum_improvement", 1e-5)
            )
            if improved:
                best_score = selection_value
                stale_epochs = 0
            else:
                stale_epochs += 1
            residual_scale = float(metrics["residual_scale"])
        payload = checkpoint_payload(
            model,
            optimizer,
            scheduler,
            epoch,
            best_score,
            stale_epochs,
            history,
            config,
            stage_name,
            stage,
            scaler,
            prior,
            cell_map,
            domain_map,
            dataset_path,
            split_path,
            residual_scale,
            response_basis_sha256,
            response_basis_source,
            mechanism_supervision_sha256,
            datasets["train"].target_gene_map,
        )
        torch.save(payload, last_path)
        if improved:
            torch.save(payload, best_path)
            if predictions is not None:
                predictions.to_parquet(
                    output_dir / "validation_predictions.parquet", index=False
                )
        if not fixed_epoch_training and stale_epochs >= patience:
            print(f"early_stopping epoch={epoch}", flush=True)
            break

    best = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(best["model_state_dict"], strict=True)
    if fixed_epoch_training:
        final_metrics = None
        final_predictions = None
    else:
        final_metrics, final_predictions = evaluate(
            model,
            loaders["validation"],
            device,
            scaler,
            top_k,
            loss_type,
            huber_delta,
            correlation_weight,
            str(stage.get("calibration_metric", selection_metric)),
            float(stage.get("calibration_maximum", 1.5)),
            int(stage.get("calibration_steps", 21)),
        )
    drug_input_ablations = None
    if not fixed_epoch_training and bool(
        stage.get("evaluate_drug_input_ablations", False)
    ):
        if num_workers != 0:
            raise ValueError("Drug-input ablations require num_workers=0")
        assert final_metrics is not None
        calibration_scale = float(final_metrics["residual_scale"])
        common_evaluate_args = (
            model,
            loaders["validation"],
            device,
            scaler,
            top_k,
            loss_type,
            huber_delta,
            correlation_weight,
            str(stage.get("calibration_metric", selection_metric)),
            float(stage.get("calibration_maximum", 1.5)),
            int(stage.get("calibration_steps", 21)),
        )
        zero_metrics, _ = evaluate(
            *common_evaluate_args,
            zero_perturbation=True,
            fixed_residual_scale=calibration_scale,
        )
        validation_dataset = datasets["validation"]
        permutation = compound_derangement(
            validation_dataset.frame["compound_id"].astype(str).tolist(),
            seed + 8000,
        )
        validation_dataset.set_chemical_permutation(permutation)
        try:
            shuffled_metrics, _ = evaluate(
                *common_evaluate_args,
                fixed_residual_scale=calibration_scale,
            )
        finally:
            validation_dataset.set_chemical_permutation(None)
        drug_input_ablations = {
            "calibration_scale_fixed_to_model": calibration_scale,
            "zero_P": zero_metrics,
            "compound_shuffled_P": shuffled_metrics,
            "model_minus_zero_P_pearson": float(
                final_metrics["mean_sample_pearson"]
                - zero_metrics["mean_sample_pearson"]
            ),
            "model_minus_shuffled_P_pearson": float(
                final_metrics["mean_sample_pearson"]
                - shuffled_metrics["mean_sample_pearson"]
            ),
            "permutation_fixed_points": sum(
                left == right for left, right in permutation.items()
            ),
        }
    summary = {
        "run_name": run_name,
        "stage": stage_name,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "frozen_parameter_names": frozen_parameters,
        "explicitly_trainable_parameter_names": explicitly_trainable,
        "best_epoch": int(best["epoch"]),
        "fixed_epoch_training": fixed_epoch_training,
        "checkpoint_selection": (
            "final_fixed_epoch" if fixed_epoch_training else "validation_metric"
        ),
        "elapsed_seconds": time.time() - started,
        "dataset_sha256": file_sha256(dataset_path),
        "split_sha256": file_sha256(split_path),
        "gene_panel_sha256": file_sha256(panel_path),
        "gene_count": len(panel),
        "response_basis_sha256": response_basis_sha256,
        "response_basis_source": response_basis_source,
        "mechanism_supervision_sha256": mechanism_supervision_sha256,
        "target_auxiliary_gene_count": len(datasets["train"].target_gene_map),
        "target_auxiliary_compound_count": datasets[
            "train"
        ].target_label_compound_count,
        "validation": final_metrics,
        "drug_input_ablations": drug_input_ablations,
        "test_evaluated": False,
    }
    if final_predictions is not None:
        final_predictions.to_parquet(
            output_dir / "validation_predictions.parquet", index=False
        )
    (output_dir / "history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "config.json").write_text(
        json.dumps({**config, "active_stage": stage}, indent=2) + "\n",
        encoding="utf-8",
    )
    completed_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/prc_phase5.yaml")
    parser.add_argument(
        "--stage",
        required=True,
        help="Name of a stage section in the configuration file",
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--initialize-from", type=Path)
    parser.add_argument("--split", type=Path)
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()
    config_path = project_path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    train_stage(
        config,
        args.stage,
        args.run_name,
        initialize_from=project_path(args.initialize_from) if args.initialize_from else None,
        split_override=project_path(args.split) if args.split else None,
        epochs_override=args.epochs,
    )


if __name__ == "__main__":
    main()
