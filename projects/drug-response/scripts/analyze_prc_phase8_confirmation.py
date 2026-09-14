#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from my_star.prc_metrics import masked_regression_metrics
from scripts.evaluate_prc_validation_ensemble import evaluate_members, load_predictions


CONFIRMATION = ROOT / "artifacts/prc_phase8_confirmation_summary.json"
PROTOCOL = ROOT / "artifacts/prc_phase8_protocol_lock.json"
PHASE6 = ROOT / "artifacts/prc_phase6_final_analysis_summary.json"
PHASE6_PROGRESSIVE = (
    ROOT / "artifacts/prc_phase6_progressive_final_analysis_summary.json"
)
SCIPLEX = ROOT / "data/processed/prc_sciplex3_common977_development.parquet"
READOUT_MAPPING = ROOT / "data/processed/lincs_phase2_common977_mapping.csv"
OUTPUT = ROOT / "artifacts/prc_phase8_confirmation_analysis.json"

PHASE6_PREDICTIONS = {
    fold: ROOT
    / f"artifacts/prc_phase6_final_fixed_ensemble_fold{fold}/validation_predictions.parquet"
    for fold in range(3)
}
METRIC_NAMES = (
    "mean_sample_pearson",
    "median_sample_pearson",
    "rmse",
    "mae",
    "top_50_deg_recall",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aggregate_metrics(rows: list[dict]) -> dict[str, dict[str, float]]:
    return {
        name: {
            "mean": float(np.mean([row[name] for row in rows])),
            "population_sd": float(np.std([row[name] for row in rows])),
        }
        for name in METRIC_NAMES
    }


def samplewise_masked_pearson(
    target: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    values = np.zeros(len(target), dtype=np.float64)
    for index, (target_row, prediction_row, mask_row) in enumerate(
        zip(target, prediction, mask)
    ):
        observed = mask_row.astype(bool)
        target_values = target_row[observed]
        prediction_values = prediction_row[observed]
        if len(target_values) < 2:
            continue
        target_values = target_values - target_values.mean()
        prediction_values = prediction_values - prediction_values.mean()
        denominator = np.sqrt(
            np.square(target_values).sum() * np.square(prediction_values).sum()
        )
        if denominator > 1e-12:
            values[index] = float(
                np.dot(target_values, prediction_values) / denominator
            )
    return values


def compound_block_bootstrap(
    differences_by_compound: dict[str, list[float]],
    iterations: int,
    seed: int = 2026,
) -> dict[str, float | int]:
    if not differences_by_compound:
        raise ValueError("Compound-block bootstrap requires compound groups")
    groups = [
        np.asarray(values, dtype=np.float64)
        for _, values in sorted(differences_by_compound.items())
    ]
    sums = np.asarray([values.sum() for values in groups], dtype=np.float64)
    counts = np.asarray([len(values) for values in groups], dtype=np.float64)
    rng = np.random.default_rng(seed)
    estimates = np.empty(iterations, dtype=np.float64)
    group_count = len(groups)
    for start in range(0, iterations, 500):
        batch_size = min(500, iterations - start)
        indices = rng.integers(0, group_count, size=(batch_size, group_count))
        estimates[start : start + batch_size] = sums[indices].sum(axis=1) / counts[
            indices
        ].sum(axis=1)
    observed = float(sums.sum() / counts.sum())
    return {
        "observed_mean_difference": observed,
        "bootstrap_mean_difference": float(estimates.mean()),
        "ci95_lower": float(np.quantile(estimates, 0.025)),
        "ci95_upper": float(np.quantile(estimates, 0.975)),
        "probability_difference_above_zero": float(np.mean(estimates > 0.0)),
        "compound_blocks": group_count,
        "iterations": iterations,
        "seed": seed,
    }


def _arrays(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.stack(frame["target_delta_y"]).astype(np.float32),
        np.stack(frame["predicted_delta_y"]).astype(np.float32),
        np.stack(frame["target_mask"]).astype(bool),
    )


def _metric_comparison(
    target: np.ndarray,
    phase8_prediction: np.ndarray,
    phase6_prediction: np.ndarray,
    mask: np.ndarray,
) -> dict:
    phase8 = masked_regression_metrics(target, phase8_prediction, mask, top_k=50)
    phase6 = masked_regression_metrics(target, phase6_prediction, mask, top_k=50)
    return {
        "phase8": phase8,
        "phase6_fixed": phase6,
        "phase8_minus_phase6_pearson": float(
            phase8["mean_sample_pearson"] - phase6["mean_sample_pearson"]
        ),
        "rows": int(len(target)),
        "observed_values": int(mask.sum()),
    }


def row_subgroups(
    metadata: pd.DataFrame,
    target: np.ndarray,
    phase8_prediction: np.ndarray,
    phase6_prediction: np.ndarray,
    mask: np.ndarray,
) -> dict[str, dict]:
    output: dict[str, dict] = {}
    for column in ("dataset_domain", "cell_line", "dose_nM", "duration_hours"):
        groups = {}
        values = metadata[column].astype(str).to_numpy()
        for value in sorted(set(values)):
            selected = values == value
            groups[value] = _metric_comparison(
                target[selected],
                phase8_prediction[selected],
                phase6_prediction[selected],
                mask[selected],
            )
        output[column] = {
            "estimable_multiple_groups": len(groups) > 1,
            "groups": groups,
        }
    return output


def readout_subgroups(
    mapping: pd.DataFrame,
    target: np.ndarray,
    phase8_prediction: np.ndarray,
    phase6_prediction: np.ndarray,
    mask: np.ndarray,
) -> dict[str, dict]:
    mapping = mapping.sort_values("common_panel_index").reset_index(drop=True)
    if mapping["common_panel_index"].tolist() != list(range(target.shape[1])):
        raise ValueError("LINCS readout mapping does not match the common gene panel")
    available = mapping["lincs_available"].astype(bool).to_numpy()
    landmark = mapping["lincs_is_landmark"].astype(bool).to_numpy()
    groups = {
        "lincs_measured_landmark": landmark,
        "lincs_inferred": available & ~landmark,
        "not_available_in_lincs": ~available,
    }
    output = {}
    for name, gene_mask in groups.items():
        subgroup_mask = mask & gene_mask[None, :]
        comparison = _metric_comparison(
            target,
            phase8_prediction,
            phase6_prediction,
            subgroup_mask,
        )
        comparison["readouts"] = int(gene_mask.sum())
        output[name] = comparison
    return output


def _validate_confirmation(confirmation: dict, protocol: dict) -> list[int]:
    expected_seeds = sorted(protocol["selection_rules"]["confirmation_seeds"])
    observed_seeds = sorted(confirmation["seeds"])
    if observed_seeds != expected_seeds:
        raise ValueError(
            f"Confirmation seeds differ from protocol: {observed_seeds} != {expected_seeds}"
        )
    expected_pairs = {
        (fold, seed) for fold in range(3) for seed in expected_seeds
    }
    observed_pairs = {
        (int(run["fold"]), int(run["seed"])) for run in confirmation["runs"]
    }
    if observed_pairs != expected_pairs or len(confirmation["runs"]) != 9:
        raise ValueError("Confirmation does not contain exactly three folds by three seeds")
    if confirmation.get("locked_tests_evaluated") is not False:
        raise RuntimeError("Confirmation reports locked-test access")
    return expected_seeds


def main() -> None:
    confirmation = json.loads(CONFIRMATION.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    phase6 = json.loads(PHASE6.read_text(encoding="utf-8"))
    progressive = json.loads(PHASE6_PROGRESSIVE.read_text(encoding="utf-8"))
    seeds = _validate_confirmation(confirmation, protocol)
    metadata = pd.read_parquet(
        SCIPLEX,
        columns=[
            "sample_id",
            "compound_id",
            "dataset_domain",
            "cell_line",
            "dose_nM",
            "duration_hours",
        ],
    )
    if metadata["sample_id"].duplicated().any():
        raise ValueError("SciPlex sample IDs must be unique")
    mapping = pd.read_csv(READOUT_MAPPING)

    fold_results = []
    differences_by_compound: dict[str, list[float]] = defaultdict(list)
    selected_epochs = {"anchor": [], "finetuning": []}
    for fold in range(3):
        runs = sorted(
            (run for run in confirmation["runs"] if int(run["fold"]) == fold),
            key=lambda run: int(run["seed"]),
        )
        member_paths = {
            f"seed{run['seed']}": ROOT / run["predictions"] for run in runs
        }
        weights = {name: 1.0 / len(member_paths) for name in member_paths}
        metrics, predictions = evaluate_members(member_paths, weights, top_k=50)
        output_dir = ROOT / f"artifacts/prc_phase8_equal_seed_ensemble_fold{fold}"
        output_dir.mkdir(parents=True, exist_ok=True)

        fold_metadata = predictions[["sample_id"]].merge(
            metadata,
            on="sample_id",
            how="left",
            validate="one_to_one",
        )
        if fold_metadata["compound_id"].isna().any():
            raise ValueError(f"Fold {fold} predictions have missing SciPlex metadata")
        predictions.insert(1, "compound_id", fold_metadata["compound_id"].astype(str))
        predictions.to_parquet(
            output_dir / "validation_predictions.parquet", index=False
        )
        (output_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
        )

        phase6_predictions = load_predictions(PHASE6_PREDICTIONS[fold])
        if not predictions["sample_id"].equals(phase6_predictions["sample_id"]):
            raise ValueError(f"Fold {fold} Phase 8 and Phase 6 sample IDs differ")
        target, phase8_prediction, mask = _arrays(predictions)
        phase6_target, phase6_prediction, phase6_mask = _arrays(phase6_predictions)
        if not np.allclose(target, phase6_target) or not np.array_equal(
            mask, phase6_mask
        ):
            raise ValueError(f"Fold {fold} Phase 8 and Phase 6 targets differ")

        phase8_sample_pearson = samplewise_masked_pearson(
            target, phase8_prediction, mask
        )
        phase6_sample_pearson = samplewise_masked_pearson(
            target, phase6_prediction, mask
        )
        for compound_id, difference in zip(
            fold_metadata["compound_id"].astype(str),
            phase8_sample_pearson - phase6_sample_pearson,
        ):
            differences_by_compound[compound_id].append(float(difference))

        phase6_fold = next(
            row for row in phase6["folds"] if int(row["fold"]) == fold
        )
        progressive_fold = next(
            row for row in progressive["folds"] if int(row["fold"]) == fold
        )
        fold_results.append(
            {
                "fold": fold,
                "seeds": seeds,
                "weights": weights,
                "ensemble": metrics,
                "phase6_fixed_reference": {
                    name: phase6_fold[name] for name in METRIC_NAMES
                },
                "phase6_progressive_reference": {
                    name: progressive_fold[name] for name in METRIC_NAMES
                },
                "gain_over_phase6_fixed": float(
                    metrics["mean_sample_pearson"]
                    - phase6_fold["mean_sample_pearson"]
                ),
                "gain_over_phase6_progressive": float(
                    metrics["mean_sample_pearson"]
                    - progressive_fold["mean_sample_pearson"]
                ),
                "row_subgroups": row_subgroups(
                    fold_metadata,
                    target,
                    phase8_prediction,
                    phase6_prediction,
                    mask,
                ),
                "readout_subgroups": readout_subgroups(
                    mapping,
                    target,
                    phase8_prediction,
                    phase6_prediction,
                    mask,
                ),
                "predictions": str(
                    (output_dir / "validation_predictions.parquet").relative_to(ROOT)
                ),
            }
        )
        for run in runs:
            anchor_metrics = json.loads(
                (
                    ROOT / "artifacts" / run["anchor_run"] / "metrics.json"
                ).read_text(encoding="utf-8")
            )
            finetuning_metrics = json.loads(
                (
                    ROOT / "artifacts" / run["finetuning_run"] / "metrics.json"
                ).read_text(encoding="utf-8")
            )
            selected_epochs["anchor"].append(int(anchor_metrics["best_epoch"]))
            selected_epochs["finetuning"].append(
                int(finetuning_metrics["best_epoch"])
            )

    aggregate = aggregate_metrics([row["ensemble"] for row in fold_results])
    fixed_reference = phase6["aggregate"]
    progressive_reference = progressive["aggregate"]
    primary_gain = float(
        aggregate["mean_sample_pearson"]["mean"]
        - fixed_reference["mean_sample_pearson"]["mean"]
    )
    progressive_gain = float(
        aggregate["mean_sample_pearson"]["mean"]
        - progressive_reference["mean_sample_pearson"]["mean"]
    )
    bootstrap_iterations = int(
        protocol["selection_rules"].get("bootstrap_iterations", 10_000)
    )
    bootstrap = compound_block_bootstrap(
        differences_by_compound, iterations=bootstrap_iterations
    )

    zero_gaps = [
        float(run["drug_input_ablations"]["model_minus_zero_P_pearson"])
        for run in confirmation["runs"]
    ]
    shuffled_gaps = [
        float(run["drug_input_ablations"]["model_minus_shuffled_P_pearson"])
        for run in confirmation["runs"]
    ]
    minimum_gain = float(
        protocol["selection_rules"]["fold0_minimum_absolute_gain_over_control"]
    )
    fold_gains = [float(row["gain_over_phase6_fixed"]) for row in fold_results]
    rmse_relative_change = float(
        aggregate["rmse"]["mean"] / fixed_reference["rmse"]["mean"] - 1.0
    )
    mae_relative_change = float(
        aggregate["mae"]["mean"] / fixed_reference["mae"]["mean"] - 1.0
    )
    manual_review_gates = {
        "exact_predeclared_fold_seed_grid": True,
        "locked_test_still_unopened": not (
            ROOT / "artifacts/lincs_phase2_locked_test_access_receipt.json"
        ).exists(),
        "mean_gain_at_least_protocol_minimum": primary_gain >= minimum_gain,
        "every_fold_gain_at_least_protocol_minimum": min(fold_gains)
        >= minimum_gain,
        "compound_bootstrap_ci95_lower_above_zero": bootstrap["ci95_lower"] > 0.0,
        "mean_zero_P_gap_at_least_protocol_minimum": float(np.mean(zero_gaps))
        >= minimum_gain,
        "zero_P_gap_positive_in_at_least_8_of_9_runs": sum(
            value > 0.0 for value in zero_gaps
        )
        >= 8,
        "mean_shuffled_P_gap_at_least_protocol_minimum": float(
            np.mean(shuffled_gaps)
        )
        >= minimum_gain,
        "shuffled_P_gap_positive_in_all_runs": all(
            value > 0.0 for value in shuffled_gaps
        ),
        "top50_recall_not_lower_than_phase6": aggregate["top_50_deg_recall"][
            "mean"
        ]
        >= fixed_reference["top_50_deg_recall"]["mean"],
        "rmse_relative_degradation_at_most_1_percent": rmse_relative_change <= 0.01,
        "mae_relative_degradation_at_most_1_percent": mae_relative_change <= 0.01,
    }
    passed = all(manual_review_gates.values())
    fixed_epochs = {
        name: {
            "observed_best_epochs": values,
            "median_fixed_epoch": int(np.median(values)),
            "minimum": min(values),
            "maximum": max(values),
        }
        for name, values in selected_epochs.items()
    }
    result = {
        "status": "passed_manual_external_gate" if passed else "failed_manual_external_gate",
        "candidate": {
            "variant": confirmation["selected_variant"],
            "overrides": confirmation["selected_overrides"],
            "encoder": confirmation["encoder"],
            "ensemble_rule": "equal_weight_mean_of_three_predeclared_seeds",
            "seeds": seeds,
            "weight_per_seed": 1.0 / len(seeds),
            "weight_search_performed": False,
        },
        "folds": fold_results,
        "aggregate_equal_seed_ensemble": aggregate,
        "phase6_fixed_reference": fixed_reference,
        "phase6_progressive_reference": progressive_reference,
        "gain_over_phase6_fixed": primary_gain,
        "gain_over_phase6_progressive": progressive_gain,
        "secondary_metric_relative_change_vs_phase6_fixed": {
            "rmse": rmse_relative_change,
            "mae": mae_relative_change,
            "top_50_deg_recall_absolute": float(
                aggregate["top_50_deg_recall"]["mean"]
                - fixed_reference["top_50_deg_recall"]["mean"]
            ),
        },
        "paired_compound_block_bootstrap_vs_phase6_fixed": bootstrap,
        "P_input_evidence_across_9_runs": {
            "zero_P_gap_mean": float(np.mean(zero_gaps)),
            "zero_P_gap_population_sd": float(np.std(zero_gaps)),
            "zero_P_positive_runs": sum(value > 0.0 for value in zero_gaps),
            "zero_P_gaps": zero_gaps,
            "shuffled_P_gap_mean": float(np.mean(shuffled_gaps)),
            "shuffled_P_gap_population_sd": float(np.std(shuffled_gaps)),
            "shuffled_P_positive_runs": sum(value > 0.0 for value in shuffled_gaps),
            "shuffled_P_gaps": shuffled_gaps,
        },
        "fixed_epoch_retraining_plan": fixed_epochs,
        "manual_review_gates": manual_review_gates,
        "manual_gate_note": (
            "Equal-weight seed ensembling and these manual-review criteria were added "
            "after internal confirmation but before any locked-test response access. "
            "No seed weights were optimized."
        ),
        "recommended_next_action": (
            "freeze_full_development_retraining_recipe"
            if passed
            else "keep_locked_test_sealed_and_return_to_development_optimization"
        ),
        "inputs": {
            "confirmation": {
                "path": str(CONFIRMATION.relative_to(ROOT)),
                "sha256": file_sha256(CONFIRMATION),
            },
            "protocol": {
                "path": str(PROTOCOL.relative_to(ROOT)),
                "sha256": file_sha256(PROTOCOL),
            },
            "phase6_reference": {
                "path": str(PHASE6.relative_to(ROOT)),
                "sha256": file_sha256(PHASE6),
            },
        },
        "locked_tests_evaluated": False,
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
