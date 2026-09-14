#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from my_star.data import build_datasets, load_manifest, load_observations
from my_star.metrics import regression_metrics
from my_star.model import StructureConditionedPerturbationModel, model_kwargs_from_config
from scripts.train_sciplex import (
    benchmark_baselines,
    choose_device,
    evaluate,
    file_sha256,
    project_path,
)


def verify_frozen_candidate(
    manifest: dict,
    config_path: Path,
    dataset_path: Path,
    split_path: Path,
) -> None:
    if manifest.get("status") != "frozen_before_test":
        raise RuntimeError("Candidate manifest is not frozen_before_test")
    if manifest.get("test_evaluations_before_freeze") != 0:
        raise RuntimeError("Candidate manifest records a pre-freeze test evaluation")
    expected = {
        "config": (config_path, manifest.get("config", {}).get("sha256")),
        "dataset": (dataset_path, manifest.get("inputs", {}).get("dataset", {}).get("sha256")),
        "split": (split_path, manifest.get("inputs", {}).get("split", {}).get("sha256")),
    }
    for name, (path, digest) in expected.items():
        if not digest or file_sha256(path) != digest:
            raise RuntimeError(f"Frozen candidate {name} hash mismatch: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ensemble-summary", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    summary_path = (
        args.ensemble_summary
        if args.ensemble_summary.is_absolute()
        else ROOT / args.ensemble_summary
    )
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    candidate_path = (
        args.candidate_manifest
        if args.candidate_manifest.is_absolute()
        else ROOT / args.candidate_manifest
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    ensemble_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    candidate_manifest = json.loads(candidate_path.read_text(encoding="utf-8"))

    dataset_path = project_path(config["dataset_path"])
    split_path = project_path(config["split_path"])
    verify_frozen_candidate(candidate_manifest, config_path, dataset_path, split_path)
    observations = load_observations(dataset_path)
    manifest = load_manifest(split_path)
    datasets, scaler, cell_map, _ = build_datasets(
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
    device = choose_device()
    loader = DataLoader(
        datasets["test"],
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config["num_workers"]),
    )

    frames = []
    individual_metrics = []
    for run in ensemble_summary["individual_runs"]:
        relative = Path(str(run["checkpoint"]).replace("\\", "/"))
        checkpoint_path = ROOT / relative
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if checkpoint["data_sha256"] != file_sha256(dataset_path):
            raise RuntimeError(f"Dataset hash mismatch for {checkpoint_path}")
        if checkpoint["split_sha256"] != file_sha256(split_path):
            raise RuntimeError(f"Split hash mismatch for {checkpoint_path}")
        model = StructureConditionedPerturbationModel(
            output_dim=int(checkpoint["output_dim"]),
            num_cell_lines=len(cell_map),
            response_basis=checkpoint.get("response_basis"),
            **model_kwargs_from_config(checkpoint["config"]),
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        metrics, predictions = evaluate(
            model,
            loader,
            device,
            scaler,
            int(config["top_deg_count"]),
            float(config["correlation_loss_weight"]),
            float(checkpoint.get("residual_scale", 1.0)),
            str(checkpoint["config"].get("reconstruction_loss", "mse")),
            float(checkpoint["config"].get("huber_delta", 1.0)),
        )
        individual_metrics.append(
            {
                "run_name": run["run_name"],
                "seed": run["seed"],
                **metrics,
            }
        )
        frames.append(predictions)
        del model

    reference_ids = frames[0]["sample_id"].tolist()
    if any(frame["sample_id"].tolist() != reference_ids for frame in frames[1:]):
        raise RuntimeError("Seed checkpoints produced different test sample order")
    y_true = np.stack(frames[0]["target_delta_y"].map(np.asarray))
    prior = np.stack(frames[0]["context_prior_delta_y"].map(np.asarray))
    mean_prediction = np.mean(
        [
            np.stack(frame["uncalibrated_predicted_delta_y"].map(np.asarray))
            for frame in frames
        ],
        axis=0,
    )
    ensemble_scale = float(ensemble_summary["ensemble"]["residual_scale"])
    prediction = prior + ensemble_scale * (mean_prediction - prior)
    metrics = regression_metrics(y_true, prediction, top_k=int(config["top_deg_count"]))

    output = frames[0].copy()
    output["predicted_delta_y"] = list(prediction.astype(np.float32))
    output_dir.mkdir(parents=True, exist_ok=True)
    output.to_parquet(output_dir / "test_predictions.parquet", index=False)
    result = {
        "device": str(device),
        "checkpoint_count": len(frames),
        "ensemble_residual_scale_from_validation": ensemble_scale,
        "test": metrics,
        "individual_test_runs": individual_metrics,
        "baselines_on_test": benchmark_baselines(
            observations,
            manifest,
            int(config["top_deg_count"]),
            "test",
        ),
        "data_sha256": file_sha256(dataset_path),
        "split_sha256": file_sha256(split_path),
        "candidate_manifest_sha256": file_sha256(candidate_path),
    }
    (output_dir / "test_metrics.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
