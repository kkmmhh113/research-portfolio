#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from my_star.prc_metrics import calibrate_masked_residual_scale


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path).sort_values("sample_id").reset_index(drop=True)
    required = {
        "sample_id",
        "target_delta_y",
        "target_mask",
        "context_prior_delta_y",
        "uncalibrated_predicted_delta_y",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Prediction file is missing columns: {sorted(missing)}")
    return frame


def _aligned_arrays(
    member_paths: Mapping[str, Path],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    if len(member_paths) < 2:
        raise ValueError("At least two ensemble members are required")
    frames = {name: load_predictions(path) for name, path in member_paths.items()}
    reference_name = next(iter(frames))
    reference = frames[reference_name]
    target = np.stack(reference["target_delta_y"]).astype(np.float32)
    mask = np.stack(reference["target_mask"]).astype(bool)
    prior = np.stack(reference["context_prior_delta_y"]).astype(np.float32)
    predictions: dict[str, np.ndarray] = {}
    for name, frame in frames.items():
        if not reference["sample_id"].equals(frame["sample_id"]):
            raise ValueError(
                f"Ensemble prediction files have different sample IDs: {reference_name}, {name}"
            )
        member_target = np.stack(frame["target_delta_y"]).astype(np.float32)
        member_mask = np.stack(frame["target_mask"]).astype(bool)
        member_prior = np.stack(frame["context_prior_delta_y"]).astype(np.float32)
        if not np.allclose(target, member_target) or not np.array_equal(mask, member_mask):
            raise ValueError(f"Ensemble member {name} was evaluated against different targets")
        if not np.allclose(prior, member_prior):
            raise ValueError(f"Ensemble member {name} uses a different context prior")
        predictions[name] = np.stack(
            frame["uncalibrated_predicted_delta_y"]
        ).astype(np.float32)
    return reference, target, mask, prior, predictions


def _pairwise_error_correlations(
    target: np.ndarray,
    prior: np.ndarray,
    mask: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    top_k: int,
) -> dict[str, float]:
    calibrated_errors: dict[str, np.ndarray] = {}
    for name, prediction in predictions.items():
        scale, _ = calibrate_masked_residual_scale(
            target,
            prediction,
            prior,
            mask,
            top_k=top_k,
            metric="mean_sample_pearson",
            maximum=1.5,
            steps=61,
        )
        calibrated = prior + scale * (prediction - prior)
        calibrated_errors[name] = (target - calibrated)[mask]
    names = list(predictions)
    correlations: dict[str, float] = {}
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            left_error = calibrated_errors[left]
            right_error = calibrated_errors[right]
            denominator = float(left_error.std() * right_error.std())
            correlations[f"{left}__{right}"] = (
                float(np.corrcoef(left_error, right_error)[0, 1])
                if denominator > 1e-12
                else 0.0
            )
    return correlations


def evaluate_members(
    member_paths: Mapping[str, Path],
    weights: Mapping[str, float],
    top_k: int,
) -> tuple[dict, pd.DataFrame]:
    if set(member_paths) != set(weights):
        raise ValueError("Ensemble member and weight names must match")
    numeric_weights = {name: float(value) for name, value in weights.items()}
    if any(not np.isfinite(value) or value < 0.0 for value in numeric_weights.values()):
        raise ValueError("Ensemble weights must be finite and non-negative")
    if not np.isclose(sum(numeric_weights.values()), 1.0, atol=1e-8):
        raise ValueError("Ensemble weights must sum to one")

    reference, target, mask, prior, predictions = _aligned_arrays(member_paths)
    uncalibrated = np.zeros_like(target)
    for name, prediction in predictions.items():
        uncalibrated += numeric_weights[name] * prediction
    residual_scale, metrics = calibrate_masked_residual_scale(
        target,
        uncalibrated,
        prior,
        mask,
        top_k=top_k,
        metric="mean_sample_pearson",
        maximum=1.5,
        steps=61,
    )
    calibrated = prior + residual_scale * (uncalibrated - prior)
    output = pd.DataFrame(
        {
            "sample_id": reference["sample_id"],
            "target_delta_y": list(target),
            "target_mask": list(mask),
            "context_prior_delta_y": list(prior),
            "uncalibrated_predicted_delta_y": list(uncalibrated),
            "predicted_delta_y": list(calibrated.astype(np.float32)),
        }
    )
    member_metadata = {
        name: {
            "path": str(member_paths[name]),
            "sha256": file_sha256(member_paths[name]),
            "weight": numeric_weights[name],
        }
        for name in member_paths
    }
    return {
        "members": member_metadata,
        "residual_scale": residual_scale,
        **metrics,
        "pairwise_calibrated_error_correlation": _pairwise_error_correlations(
            target, prior, mask, predictions, top_k
        ),
        "test_evaluated": False,
    }, output


def evaluate_pair(
    scratch_path: Path,
    transfer_path: Path,
    scratch_weight: float,
    top_k: int,
) -> tuple[dict, pd.DataFrame]:
    metrics, output = evaluate_members(
        {"scratch": scratch_path, "transfer": transfer_path},
        {"scratch": scratch_weight, "transfer": 1.0 - scratch_weight},
        top_k,
    )
    pair_metrics = {
        "scratch_path": str(scratch_path),
        "scratch_sha256": metrics["members"]["scratch"]["sha256"],
        "transfer_path": str(transfer_path),
        "transfer_sha256": metrics["members"]["transfer"]["sha256"],
        "scratch_weight": scratch_weight,
        "transfer_weight": 1.0 - scratch_weight,
        **{key: value for key, value in metrics.items() if key != "members"},
    }
    return pair_metrics, output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--transfer", type=Path, required=True)
    parser.add_argument("--scratch-weight", type=float, default=0.55)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not 0.0 <= args.scratch_weight <= 1.0:
        raise ValueError("scratch-weight must be between zero and one")
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics, predictions = evaluate_pair(
        args.scratch if args.scratch.is_absolute() else ROOT / args.scratch,
        args.transfer if args.transfer.is_absolute() else ROOT / args.transfer,
        args.scratch_weight,
        args.top_k,
    )
    predictions.to_parquet(output_dir / "validation_predictions.parquet", index=False)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
