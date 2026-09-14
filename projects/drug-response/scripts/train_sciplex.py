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
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from my_star.data import build_datasets, fit_response_basis, load_manifest, load_observations
from my_star.chemistry import morgan_fingerprint
from my_star.metrics import calibrate_residual_scale, regression_metrics
from my_star.model import (
    StructureConditionedPerturbationModel,
    correlation_loss,
    model_kwargs_from_config,
    reconstruction_loss,
)


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def move_batch(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def model_forward(model: nn.Module, batch: dict) -> torch.Tensor:
    return model(
        batch["fingerprint"],
        batch["descriptors"],
        batch["cell_index"],
        batch["context"],
        batch["baseline"],
    )


def fit_gene_loss_weights(
    observations: pd.DataFrame,
    train_ids: set[str],
    mode: str,
    noise_floor: float,
    maximum: float,
) -> np.ndarray | None:
    if mode == "none":
        return None
    if mode != "replicate_reliability":
        raise ValueError(f"Unknown gene loss weighting mode: {mode}")
    if noise_floor <= 0.0 or maximum < 1.0:
        raise ValueError("Invalid gene reliability weighting parameters")
    train = observations[observations["compound_id"].isin(train_ids)]
    noise = np.stack(train["delta_y_replicate_std"].map(np.asarray)).astype(np.float32)
    median_noise = np.median(noise, axis=0)
    weights = 1.0 / (median_noise**2 + noise_floor**2)
    weights /= weights.mean()
    weights = np.clip(weights, 1.0 / maximum, maximum)
    weights /= weights.mean()
    return weights.astype(np.float32)


def sample_reliability_weights(
    replicate_pearson: torch.Tensor,
    mode: str,
    floor: float,
    power: float,
) -> torch.Tensor | None:
    if mode == "none":
        return None
    if mode != "replicate_reliability":
        raise ValueError(f"Unknown sample loss weighting mode: {mode}")
    if floor <= 0.0 or power <= 0.0:
        raise ValueError("Sample reliability floor and power must be positive")
    return torch.clamp(replicate_pearson, min=0.0).pow(power) + floor


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    scaler,
    top_k: int,
    correlation_weight: float,
    residual_scale: float = 1.0,
    loss_type: str = "mse",
    huber_delta: float = 1.0,
) -> tuple[dict[str, float], pd.DataFrame]:
    model.eval()
    predictions, targets, priors, sample_ids, compound_ids = [], [], [], [], []
    losses = []
    scale_tensor = torch.as_tensor(scaler.scale, device=device)
    mean_tensor = torch.as_tensor(scaler.mean, device=device)
    for batch in loader:
        batch = move_batch(batch, device)
        prediction = model_forward(model, batch)
        point_loss = reconstruction_loss(prediction, batch["target"], loss_type, huber_delta)
        prediction_raw = prediction * scale_tensor + mean_tensor + batch["context_prior"]
        loss = point_loss + correlation_weight * correlation_loss(prediction_raw, batch["target_raw"])
        losses.append(float(loss.item()))
        predictions.append(prediction.cpu().numpy())
        targets.append(batch["target_raw"].cpu().numpy())
        priors.append(batch["context_prior"].cpu().numpy())
        sample_ids.extend(batch["sample_id"])
        compound_ids.extend(batch["compound_id"])
    prediction_scaled = np.vstack(predictions)
    target_raw = np.vstack(targets)
    prior_raw = np.vstack(priors)
    prediction_uncalibrated = scaler.inverse_transform(prediction_scaled) + prior_raw
    prediction_raw = prior_raw + residual_scale * (prediction_uncalibrated - prior_raw)
    metrics = regression_metrics(target_raw, prediction_raw, top_k=top_k)
    metrics["standardized_loss"] = float(np.mean(losses))
    output = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "compound_id": compound_ids,
            "target_delta_y": list(target_raw.astype(np.float32)),
            "context_prior_delta_y": list(prior_raw.astype(np.float32)),
            "uncalibrated_predicted_delta_y": list(
                prediction_uncalibrated.astype(np.float32)
            ),
            "predicted_delta_y": list(prediction_raw.astype(np.float32)),
        }
    )
    return metrics, output


def benchmark_baselines(
    observations: pd.DataFrame,
    manifest: dict,
    top_k: int,
    evaluation_split: str,
) -> dict[str, dict[str, float] | float]:
    train_ids = set(manifest["splits"]["train"])
    evaluation_ids = set(manifest["splits"][evaluation_split])
    train = observations[observations.compound_id.isin(train_ids)].copy()
    evaluation = observations[observations.compound_id.isin(evaluation_ids)].copy()
    y_true = np.stack(evaluation["delta_y"].map(np.asarray)).astype(np.float32)
    train_targets = np.stack(train["delta_y"].map(np.asarray)).astype(np.float32)

    zero_prediction = np.zeros_like(y_true)
    train_mean = np.broadcast_to(train_targets.mean(axis=0), y_true.shape)

    context_columns = ["cell_line", "dose_nM", "duration_hours"]
    context_means = {
        key: np.stack(group["delta_y"].map(np.asarray)).mean(axis=0)
        for key, group in train.groupby(context_columns)
    }
    context_prediction = np.stack(
        [
            context_means.get(
                (row.cell_line, row.dose_nM, row.duration_hours),
                train_targets.mean(axis=0),
            )
            for row in evaluation.itertuples()
        ]
    )

    train_structures = train[["compound_id", "canonical_smiles"]].drop_duplicates().reset_index(drop=True)
    evaluation_structures = evaluation[["compound_id", "canonical_smiles"]].drop_duplicates().reset_index(drop=True)
    train_fingerprints = np.stack(train_structures.canonical_smiles.map(morgan_fingerprint)).astype(bool)
    nearest: dict[str, tuple[str, float]] = {}
    for row in evaluation_structures.itertuples():
        fingerprint = morgan_fingerprint(row.canonical_smiles).astype(bool)
        intersection = np.logical_and(train_fingerprints, fingerprint).sum(axis=1)
        union = np.logical_or(train_fingerprints, fingerprint).sum(axis=1)
        similarities = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union > 0)
        best = int(np.argmax(similarities))
        nearest[row.compound_id] = (str(train_structures.iloc[best].compound_id), float(similarities[best]))

    exact_context = {
        (row.compound_id, row.cell_line, row.dose_nM, row.duration_hours): np.asarray(row.delta_y)
        for row in train.itertuples()
    }
    compound_means = {
        compound_id: np.stack(group["delta_y"].map(np.asarray)).mean(axis=0)
        for compound_id, group in train.groupby("compound_id")
    }
    nearest_predictions = []
    for row in evaluation.itertuples():
        neighbor, _ = nearest[row.compound_id]
        nearest_predictions.append(
            exact_context.get(
                (neighbor, row.cell_line, row.dose_nM, row.duration_hours),
                compound_means[neighbor],
            )
        )
    nearest_prediction = np.stack(nearest_predictions)
    return {
        "zero_change": regression_metrics(y_true, zero_prediction, top_k),
        "train_mean": regression_metrics(y_true, train_mean, top_k),
        "context_mean": regression_metrics(y_true, context_prediction, top_k),
        "nearest_structure": regression_metrics(y_true, nearest_prediction, top_k),
        "nearest_structure_mean_tanimoto": float(np.mean([value[1] for value in nearest.values()])),
    }


def calibrate_predictions(
    predictions: pd.DataFrame,
    top_k: int,
    metric: str,
    maximum: float,
    steps: int,
) -> tuple[float, dict[str, float], pd.DataFrame]:
    y_true = np.stack(predictions["target_delta_y"].map(np.asarray))
    y_pred = np.stack(predictions["predicted_delta_y"].map(np.asarray))
    prior = np.stack(predictions["context_prior_delta_y"].map(np.asarray))
    scale, metrics = calibrate_residual_scale(
        y_true,
        y_pred,
        prior,
        top_k=top_k,
        metric=metric,
        maximum=maximum,
        steps=steps,
    )
    calibrated = predictions.copy()
    calibrated_values = prior + scale * (y_pred - prior)
    calibrated["predicted_delta_y"] = list(calibrated_values.astype(np.float32))
    metrics["residual_scale"] = scale
    return scale, metrics, calibrated


def train(config: dict, run_name: str, epochs_override: int | None, evaluate_test: bool) -> Path:
    seed = int(config["seed"])
    set_seed(seed)
    device = choose_device()
    observations = load_observations(project_path(config["dataset_path"]))
    split_path = project_path(config["split_path"])
    dataset_path = project_path(config["dataset_path"])
    gene_panel_path = project_path(config["gene_panel_path"])
    gene_panel = pd.read_csv(gene_panel_path)
    manifest = load_manifest(split_path)
    train_ids = set(manifest["splits"]["train"])
    datasets, scaler, cell_map, context_prior = build_datasets(
        observations,
        manifest,
        target_scaling=str(config.get("target_scaling", "per_gene")),
        fingerprint_radius=int(config.get("fingerprint_radius", 2)),
        fingerprint_use_counts=bool(config.get("fingerprint_use_counts", False)),
        fingerprint_include_chirality=bool(
            config.get("fingerprint_include_chirality", False)
        ),
        descriptor_set=str(config.get("descriptor_set", "basic")),
    )
    response_basis_array = fit_response_basis(datasets["train"], config.get("response_rank"))
    response_basis = (
        torch.from_numpy(response_basis_array) if response_basis_array is not None else None
    )

    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=int(config["batch_size"]),
            shuffle=True,
            drop_last=False,
            num_workers=int(config["num_workers"]),
            pin_memory=device.type == "cuda",
        ),
        "validation": DataLoader(
            datasets["validation"],
            batch_size=int(config["batch_size"]),
            shuffle=False,
            num_workers=int(config["num_workers"]),
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=int(config["batch_size"]),
            shuffle=False,
            num_workers=int(config["num_workers"]),
        ),
    }
    output_dim = len(scaler.mean)
    if len(gene_panel) != output_dim:
        raise ValueError("Gene panel length does not match response dimension")
    model = StructureConditionedPerturbationModel(
        output_dim=output_dim,
        num_cell_lines=len(cell_map),
        response_basis=response_basis,
        **model_kwargs_from_config(config),
    ).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    epochs = int(epochs_override or config["epochs"])
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    use_amp = device.type == "cuda"
    grad_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    correlation_weight = float(config["correlation_loss_weight"])
    loss_type = str(config.get("reconstruction_loss", "mse"))
    huber_delta = float(config.get("huber_delta", 1.0))
    calibration_metric = str(config.get("calibration_metric", "mean_sample_pearson"))
    calibration_maximum = float(config.get("calibration_maximum", 1.5))
    calibration_steps = int(config.get("calibration_steps", 31))
    selection_metric = str(config.get("selection_metric", "mean_sample_pearson"))
    minimize_selection = selection_metric in {"standardized_loss", "rmse", "mae"}
    target_scale_tensor = torch.as_tensor(scaler.scale, device=device)
    target_mean_tensor = torch.as_tensor(scaler.mean, device=device)
    gene_loss_weights_array = fit_gene_loss_weights(
        observations,
        train_ids,
        str(config.get("gene_loss_weighting", "none")),
        float(config.get("gene_noise_floor", 0.05)),
        float(config.get("gene_weight_maximum", 4.0)),
    )
    gene_loss_weights = (
        torch.as_tensor(gene_loss_weights_array, device=device)
        if gene_loss_weights_array is not None
        else None
    )
    sample_weighting = str(config.get("sample_loss_weighting", "none"))
    sample_weight_floor = float(config.get("sample_weight_floor", 0.25))
    sample_weight_power = float(config.get("sample_weight_power", 1.0))

    run_dir = project_path(config["output_dir"]) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "best_model.pt"
    history = []
    best_validation_score = float("inf") if minimize_selection else -float("inf")
    stale_epochs = 0
    started = time.time()
    print(
        f"device={device} train={len(datasets['train'])} validation={len(datasets['validation'])} "
        f"test={len(datasets['test'])} parameters={sum(p.numel() for p in model.parameters()):,}"
    )

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        train_point_losses = []
        train_correlation_losses = []
        for batch in loaders["train"]:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                prediction = model_forward(model, batch)
                sample_weights = sample_reliability_weights(
                    batch["replicate_pearson"],
                    sample_weighting,
                    sample_weight_floor,
                    sample_weight_power,
                )
                point_loss = reconstruction_loss(
                    prediction,
                    batch["target"],
                    loss_type,
                    huber_delta,
                    gene_loss_weights,
                    sample_weights,
                )
                prediction_raw = (
                    prediction * target_scale_tensor
                    + target_mean_tensor
                    + batch["context_prior"]
                )
                corr = correlation_loss(prediction_raw, batch["target_raw"], sample_weights)
                loss = float(config["gene_loss_weight"]) * point_loss + correlation_weight * corr
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            train_losses.append(float(loss.item()))
            train_point_losses.append(float(point_loss.item()))
            train_correlation_losses.append(float(corr.item()))
        scheduler.step()

        validation_metrics, validation_predictions = evaluate(
            model,
            loaders["validation"],
            device,
            scaler,
            int(config["top_deg_count"]),
            correlation_weight,
            loss_type=loss_type,
            huber_delta=huber_delta,
        )
        residual_scale, calibrated_metrics, _ = calibrate_predictions(
            validation_predictions,
            int(config["top_deg_count"]),
            calibration_metric,
            calibration_maximum,
            calibration_steps,
        )
        calibrated_metrics["standardized_loss"] = validation_metrics["standardized_loss"]
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "train_point_loss": float(np.mean(train_point_losses)),
            "train_correlation_loss": float(np.mean(train_correlation_losses)),
            "learning_rate": float(scheduler.get_last_lr()[0]),
            **{f"validation_{key}": value for key, value in calibrated_metrics.items()},
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True))

        selection_value = float(calibrated_metrics[selection_metric])
        improved = (
            selection_value < best_validation_score - 1e-5
            if minimize_selection
            else selection_value > best_validation_score + 1e-5
        )
        if improved:
            best_validation_score = selection_value
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "best_validation_score": best_validation_score,
                    "selection_metric": selection_metric,
                    "residual_scale": residual_scale,
                    "response_basis": response_basis,
                    "config": config,
                    "cell_map": cell_map,
                    "target_scaler": scaler.to_dict(),
                    "gene_loss_weights": (
                        gene_loss_weights_array.tolist()
                        if gene_loss_weights_array is not None
                        else None
                    ),
                    "context_prior": context_prior.to_dict(),
                    "split_manifest": manifest,
                    "output_dim": output_dim,
                    "gene_panel": gene_panel["ensembl_id"].astype(str).tolist(),
                    "data_sha256": file_sha256(dataset_path),
                    "split_sha256": file_sha256(split_path),
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
        if stale_epochs >= int(config["early_stopping_patience"]):
            print(f"early_stopping epoch={epoch}")
            break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    residual_scale = float(checkpoint.get("residual_scale", 1.0))
    validation_metrics, validation_predictions = evaluate(
        model,
        loaders["validation"],
        device,
        scaler,
        int(config["top_deg_count"]),
        correlation_weight,
        residual_scale,
        loss_type,
        huber_delta,
    )
    summary = {
        "run_name": run_name,
        "device": str(device),
        "best_epoch": int(checkpoint["epoch"]),
        "elapsed_seconds": time.time() - started,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "split_strategy": manifest["strategy"],
        "selection_metric": selection_metric,
        "residual_scale": residual_scale,
        "validation": validation_metrics,
        "baselines_on_validation": benchmark_baselines(
            observations,
            manifest,
            int(config["top_deg_count"]),
            "validation",
        ),
    }
    if evaluate_test:
        test_metrics, test_predictions = evaluate(
            model,
            loaders["test"],
            device,
            scaler,
            int(config["top_deg_count"]),
            correlation_weight,
            residual_scale,
            loss_type,
            huber_delta,
        )
        summary["test"] = test_metrics
        summary["baselines_on_test"] = benchmark_baselines(
            observations,
            manifest,
            int(config["top_deg_count"]),
            "test",
        )
        test_predictions.to_parquet(run_dir / "test_predictions.parquet", index=False)
    (run_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    validation_predictions.to_parquet(run_dir / "validation_predictions.parquet", index=False)
    print(json.dumps(summary, indent=2))
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/sciplex_phase1.yaml")
    parser.add_argument("--run-name", default="sciplex_phase1")
    parser.add_argument("--epochs", type=int)
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Evaluate the held-out test split after model selection",
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    train(config, args.run_name, args.epochs, args.evaluate_test)


if __name__ == "__main__":
    main()
