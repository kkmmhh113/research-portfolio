from __future__ import annotations

import pandas as pd

from our_star.evaluation import (
    attach_pointwise_interval,
    classify_failure_mechanisms,
    evaluate_prespecified_gates,
)


def test_interval_alignment_gates_and_failure_taxonomy() -> None:
    comparison = pd.DataFrame(
        {"time_h": [0.0, 1.0, 2.0], "plasma_parent_mg_l": [0.0, 1.0, 0.5]}
    )
    prediction = pd.DataFrame(
        {
            "time_h": [0.0, 1.0, 2.0],
            "lo": [0.0, 0.8, 0.4],
            "hi": [0.0, 1.2, 0.6],
        }
    )
    aligned, coverage = attach_pointwise_interval(
        comparison, prediction, lower_column="lo", upper_column="hi"
    )
    assert aligned["inside_prediction_interval"].all()
    assert coverage["pointwise_interval_coverage"] == 1.0
    metrics = {
        "absolute_average_fold_error": 1.5,
        "fraction_within_2_fold": 0.9,
        "auc_predicted_observed_ratio": 0.4,
        "cmax_predicted_observed_ratio": 1.0,
        "predicted_tmax_at_observed_times_h": 2.5,
        "observed_tmax_h": 1.0,
    }
    criteria = {
        "maximum_absolute_average_fold_error": 2.0,
        "minimum_fraction_within_2_fold": 0.8,
        "auc_predicted_observed_ratio": [0.5, 2.0],
        "cmax_predicted_observed_ratio": [0.5, 2.0],
        "maximum_absolute_tmax_error_h": 1.0,
        "minimum_pointwise_interval_coverage": 0.7,
        "maximum_mass_balance_error_mg": 1e-5,
    }
    report = evaluate_prespecified_gates(
        metrics,
        coverage,
        maximum_mass_balance_error_mg=1e-8,
        criteria=criteria,
    )
    assert not report["all_gates_passed"]
    assert set(report["failed_gates"]) == {
        "auc_predicted_observed_ratio",
        "absolute_tmax_error_h",
    }
    classes = classify_failure_mechanisms(report, analyte_role="metabolite")
    labels = {record["failure_class"] for record in classes}
    assert "absorption_timing" in labels
    assert "clearance_or_bioavailability" in labels
    assert "parent_metabolite_structure" in labels
