"""Prespecified interval coverage, benchmark gates, and failure classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GateResult:
    name: str
    value: float
    criterion: str
    passed: bool


def attach_pointwise_interval(
    comparison: pd.DataFrame,
    prediction: pd.DataFrame,
    *,
    lower_column: str,
    upper_column: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Interpolate a frozen prediction interval onto comparison times."""

    required_comparison = {"time_h", "plasma_parent_mg_l"}
    missing = required_comparison - set(comparison)
    if missing:
        raise ValueError(f"comparison table is missing {sorted(missing)}")
    missing_prediction = {"time_h", lower_column, upper_column} - set(prediction)
    if missing_prediction:
        raise ValueError(f"prediction table is missing {sorted(missing_prediction)}")
    times = prediction["time_h"].to_numpy(float)
    lower = prediction[lower_column].to_numpy(float)
    upper = prediction[upper_column].to_numpy(float)
    if np.any(~np.isfinite(times)) or np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper)):
        raise ValueError("prediction intervals must be finite")
    order = np.argsort(times)
    times, lower, upper = times[order], lower[order], upper[order]
    if np.any(np.diff(times) <= 0.0) or np.any(lower > upper):
        raise ValueError("prediction interval times must be unique and lower <= upper")
    observed_times = comparison["time_h"].to_numpy(float)
    if observed_times.min() < times.min() or observed_times.max() > times.max():
        raise ValueError("observations fall outside the prediction interval")
    aligned = comparison.copy()
    aligned["predicted_p05_mg_l"] = np.interp(observed_times, times, lower)
    aligned["predicted_p95_mg_l"] = np.interp(observed_times, times, upper)
    observed = aligned["plasma_parent_mg_l"].to_numpy(float)
    covered = (
        (observed >= aligned["predicted_p05_mg_l"].to_numpy(float))
        & (observed <= aligned["predicted_p95_mg_l"].to_numpy(float))
    )
    aligned["inside_prediction_interval"] = covered
    return aligned, {
        "prediction_interval_level": 0.90,
        "pointwise_covered_count": int(covered.sum()),
        "pointwise_observation_count": int(covered.size),
        "pointwise_interval_coverage": float(np.mean(covered)),
    }


def evaluate_prespecified_gates(
    metrics: Mapping[str, Any],
    interval_metrics: Mapping[str, Any],
    *,
    maximum_mass_balance_error_mg: float,
    criteria: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate every machine-readable gate without dropping failures."""

    auc_bounds = tuple(map(float, criteria["auc_predicted_observed_ratio"]))
    cmax_bounds = tuple(map(float, criteria["cmax_predicted_observed_ratio"]))
    values = {
        "absolute_average_fold_error": float(metrics["absolute_average_fold_error"]),
        "fraction_within_2_fold": float(metrics["fraction_within_2_fold"]),
        "auc_predicted_observed_ratio": float(metrics["auc_predicted_observed_ratio"]),
        "cmax_predicted_observed_ratio": float(metrics["cmax_predicted_observed_ratio"]),
        "absolute_tmax_error_h": abs(
            float(metrics["predicted_tmax_at_observed_times_h"])
            - float(metrics["observed_tmax_h"])
        ),
        "pointwise_interval_coverage": float(interval_metrics["pointwise_interval_coverage"]),
        "maximum_mass_balance_error_mg": float(maximum_mass_balance_error_mg),
    }
    tests = {
        "absolute_average_fold_error": (
            values["absolute_average_fold_error"]
            <= float(criteria["maximum_absolute_average_fold_error"]),
            f"<= {float(criteria['maximum_absolute_average_fold_error'])}",
        ),
        "fraction_within_2_fold": (
            values["fraction_within_2_fold"]
            >= float(criteria["minimum_fraction_within_2_fold"]),
            f">= {float(criteria['minimum_fraction_within_2_fold'])}",
        ),
        "auc_predicted_observed_ratio": (
            auc_bounds[0] <= values["auc_predicted_observed_ratio"] <= auc_bounds[1],
            f"in [{auc_bounds[0]}, {auc_bounds[1]}]",
        ),
        "cmax_predicted_observed_ratio": (
            cmax_bounds[0] <= values["cmax_predicted_observed_ratio"] <= cmax_bounds[1],
            f"in [{cmax_bounds[0]}, {cmax_bounds[1]}]",
        ),
        "absolute_tmax_error_h": (
            values["absolute_tmax_error_h"]
            <= float(criteria["maximum_absolute_tmax_error_h"]),
            f"<= {float(criteria['maximum_absolute_tmax_error_h'])} h",
        ),
        "pointwise_interval_coverage": (
            values["pointwise_interval_coverage"]
            >= float(criteria["minimum_pointwise_interval_coverage"]),
            f">= {float(criteria['minimum_pointwise_interval_coverage'])}",
        ),
        "maximum_mass_balance_error_mg": (
            values["maximum_mass_balance_error_mg"]
            < float(criteria["maximum_mass_balance_error_mg"]),
            f"< {float(criteria['maximum_mass_balance_error_mg'])} mg",
        ),
    }
    gates = {
        name: {
            "value": values[name],
            "criterion": criterion,
            "passed": bool(passed),
        }
        for name, (passed, criterion) in tests.items()
    }
    failed = [name for name, record in gates.items() if not record["passed"]]
    return {"all_gates_passed": not failed, "failed_gates": failed, "gates": gates}


def classify_failure_mechanisms(
    gate_report: Mapping[str, Any],
    *,
    analyte_role: str = "parent",
) -> list[dict[str, str]]:
    """Map prespecified gate failures to mechanism classes, without rescue fitting."""

    failed = set(gate_report.get("failed_gates", ()))
    classes: list[dict[str, str]] = []
    if "maximum_mass_balance_error_mg" in failed:
        classes.append(
            {
                "failure_class": "implementation_mass_conservation",
                "required_mechanism_or_action": "repair_before_scientific_interpretation",
            }
        )
    if "absolute_tmax_error_h" in failed:
        classes.append(
            {
                "failure_class": "absorption_timing",
                "required_mechanism_or_action": "dissolution_gastric_emptying_or_transit_evidence",
            }
        )
    if "auc_predicted_observed_ratio" in failed:
        classes.append(
            {
                "failure_class": "clearance_or_bioavailability",
                "required_mechanism_or_action": "independent_hepatic_renal_or_transporter_clearance",
            }
        )
    if "cmax_predicted_observed_ratio" in failed:
        classes.append(
            {
                "failure_class": "distribution_or_absorption_amplitude",
                "required_mechanism_or_action": "validated_partition_or_absorption_predictor",
            }
        )
    if "pointwise_interval_coverage" in failed:
        classes.append(
            {
                "failure_class": "population_variability_or_structural_shape",
                "required_mechanism_or_action": "correlated_variability_or_report_unresolved_shape",
            }
        )
    if analyte_role == "metabolite" and failed:
        classes.append(
            {
                "failure_class": "parent_metabolite_structure",
                "required_mechanism_or_action": "explicit_molar_reaction_and_metabolite_specific_disposition",
            }
        )
    if not classes:
        classes.append(
            {
                "failure_class": "none_under_prespecified_research_gates",
                "required_mechanism_or_action": "no_post_hoc_change",
            }
        )
    return classes
