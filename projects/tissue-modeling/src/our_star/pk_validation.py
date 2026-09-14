"""External PK validation helpers.

Validation observations must be kept separate from parameter fitting.  This
module never changes model parameters; it only aligns frozen predictions with
an observed concentration-time table and computes prespecified diagnostics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_OBSERVED_COLUMNS = {"time_h", "plasma_parent_mg_l"}
PREDICTED_MEDIAN_COLUMN = "plasma_parent_mg_l_p50"


def validate_pk_prediction(
    cohort_trajectory: pd.DataFrame,
    observed: pd.DataFrame,
    *,
    minimum_concentration_mg_l: float = 1e-8,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    missing_observed = REQUIRED_OBSERVED_COLUMNS - set(observed.columns)
    if missing_observed:
        raise ValueError(f"observed table is missing columns: {sorted(missing_observed)}")
    if PREDICTED_MEDIAN_COLUMN not in cohort_trajectory:
        raise ValueError(
            f"prediction table is missing required column {PREDICTED_MEDIAN_COLUMN!r}"
        )
    if observed["time_h"].duplicated().any():
        raise ValueError("observed time_h values must be unique")
    observed_times_unordered = observed["time_h"].to_numpy(dtype=np.float64)
    observed_concentrations_unordered = observed["plasma_parent_mg_l"].to_numpy(
        dtype=np.float64
    )
    if len(observed) < 2:
        raise ValueError("at least two observed time-concentration pairs are required")
    if not np.all(np.isfinite(observed_times_unordered)):
        raise ValueError("observed time_h values must be finite")
    if not np.all(np.isfinite(observed_concentrations_unordered)):
        raise ValueError("observed concentrations must be finite; missing rows must be excluded")
    if np.any(observed_concentrations_unordered < 0):
        raise ValueError("observed concentrations must be nonnegative")

    prediction_times = cohort_trajectory["time_h"].to_numpy(dtype=np.float64)
    prediction_values = cohort_trajectory[PREDICTED_MEDIAN_COLUMN].to_numpy(
        dtype=np.float64
    )
    if not np.all(np.isfinite(prediction_times)) or not np.all(
        np.isfinite(prediction_values)
    ):
        raise ValueError("prediction times and concentrations must be finite")
    order = np.argsort(prediction_times)
    prediction_times = prediction_times[order]
    prediction_values = prediction_values[order]

    observed_sorted = observed.sort_values("time_h").copy()
    observed_times = observed_sorted["time_h"].to_numpy(dtype=np.float64)
    if observed_times.min() < prediction_times.min() or observed_times.max() > prediction_times.max():
        raise ValueError("observed times fall outside the simulated time interval")
    interpolated = np.interp(observed_times, prediction_times, prediction_values)
    observed_values = observed_sorted["plasma_parent_mg_l"].to_numpy(dtype=np.float64)

    comparison = observed_sorted.copy()
    comparison["predicted_median_mg_l"] = interpolated
    comparison["residual_mg_l"] = interpolated - observed_values

    valid_log = (observed_values > minimum_concentration_mg_l) & (
        interpolated > minimum_concentration_mg_l
    )
    ratio = np.full_like(interpolated, np.nan)
    ratio[valid_log] = interpolated[valid_log] / observed_values[valid_log]
    comparison["predicted_observed_ratio"] = ratio

    residual = interpolated - observed_values
    log_ratio = np.log10(ratio[valid_log])
    observed_peak_index = int(np.argmax(observed_values))
    predicted_peak_index = int(np.argmax(interpolated))
    observed_auc = float(np.trapezoid(observed_values, observed_times))
    predicted_auc = float(np.trapezoid(interpolated, observed_times))
    metrics = {
        "observation_count": int(len(comparison)),
        "log_evaluable_count": int(valid_log.sum()),
        "rmse_mg_l": float(np.sqrt(np.mean(residual**2))),
        "mae_mg_l": float(np.mean(np.abs(residual))),
        "mean_fold_error": float(10 ** np.mean(log_ratio)) if len(log_ratio) else float("nan"),
        "absolute_average_fold_error": (
            float(10 ** np.mean(np.abs(log_ratio))) if len(log_ratio) else float("nan")
        ),
        "fraction_within_1_5_fold": (
            float(np.mean((ratio[valid_log] >= 1 / 1.5) & (ratio[valid_log] <= 1.5)))
            if valid_log.any()
            else float("nan")
        ),
        "fraction_within_2_fold": (
            float(np.mean((ratio[valid_log] >= 0.5) & (ratio[valid_log] <= 2.0)))
            if valid_log.any()
            else float("nan")
        ),
        "observed_cmax_mg_l": float(observed_values[observed_peak_index]),
        "predicted_cmax_at_observed_times_mg_l": float(
            interpolated[predicted_peak_index]
        ),
        "cmax_predicted_observed_ratio": (
            float(interpolated[predicted_peak_index] / observed_values[observed_peak_index])
            if observed_values[observed_peak_index] > minimum_concentration_mg_l
            else float("nan")
        ),
        "observed_tmax_h": float(observed_times[observed_peak_index]),
        "predicted_tmax_at_observed_times_h": float(
            observed_times[predicted_peak_index]
        ),
        "observed_auc_last_mg_h_l": observed_auc,
        "predicted_auc_at_observed_times_mg_h_l": predicted_auc,
        "auc_predicted_observed_ratio": (
            float(predicted_auc / observed_auc)
            if observed_auc > minimum_concentration_mg_l
            else float("nan")
        ),
        "validation_role": "external_comparison_only_no_parameter_fitting",
    }
    return comparison, metrics


def validate_pk_files(
    prediction_path: str | Path,
    observed_path: str | Path,
    output_directory: str | Path,
) -> dict[str, Any]:
    prediction = pd.read_csv(prediction_path)
    observed = pd.read_csv(observed_path)
    comparison, metrics = validate_pk_prediction(prediction, observed)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output / "pk_validation_comparison.csv", index=False)
    with (output / "pk_validation_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True, allow_nan=True)
    return metrics
