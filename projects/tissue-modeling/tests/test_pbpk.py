from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from our_star.pbpk import DoseEvent, State
from our_star.pk_validation import validate_pk_prediction
from our_star.virtual_population import reference_patient, sample_population
from our_star.virtual_trial import (
    load_drug,
    load_population_specification,
    simulate_patient,
)


def test_single_dose_conserves_mass() -> None:
    drug = load_drug("configs/virtual_phase1/synthetic_probe.yaml")
    result = simulate_patient(
        reference_patient(),
        drug,
        (DoseEvent(time_h=0.0, amount_mg=100.0),),
        duration_h=48.0,
        output_interval_h=0.5,
    )
    assert result.summary["max_abs_mass_balance_error_mg"] < 1e-5
    assert result.states_mg.min() >= 0
    assert result.summary["cmax_mg_l"] > 0
    assert result.summary["auc_last_mg_h_l"] > 0


def test_multiple_dose_discontinuities_are_accounted_for() -> None:
    drug = load_drug("configs/virtual_phase1/synthetic_probe.yaml")
    result = simulate_patient(
        reference_patient(),
        drug,
        (
            DoseEvent(time_h=0.0, amount_mg=50.0),
            DoseEvent(time_h=12.0, amount_mg=50.0),
        ),
        duration_h=24.0,
        output_interval_h=1.0,
    )
    at_second_dose = result.trajectory.loc[result.trajectory.time_h == 12.0].iloc[0]
    assert at_second_dose.delivered_dose_mg == 100.0
    assert result.summary["max_abs_mass_balance_error_mg"] < 1e-5


def test_functional_impairment_has_expected_direction() -> None:
    drug = load_drug("configs/virtual_phase1/synthetic_probe.yaml")
    dose = (DoseEvent(time_h=0.0, amount_mg=100.0),)
    healthy = simulate_patient(reference_patient(), drug, dose, duration_h=48.0)
    hepatic = simulate_patient(
        reference_patient(liver_function_fraction=0.2),
        drug,
        dose,
        duration_h=48.0,
    )
    renal = simulate_patient(
        reference_patient(renal_function_fraction=0.1),
        drug,
        dose,
        duration_h=48.0,
    )
    assert hepatic.summary["auc_last_mg_h_l"] > healthy.summary["auc_last_mg_h_l"]
    assert renal.summary["final_urine_parent_mg"] < healthy.summary["final_urine_parent_mg"]


def test_population_is_reproducible_and_flow_balanced() -> None:
    specification = load_population_specification(
        "configs/virtual_phase1/healthy_adults.yaml"
    )
    first = sample_population(specification, count=8, seed=17)
    second = sample_population(specification, count=8, seed=17)
    assert first == second
    assert len({patient.patient_id for patient in first}) == 8
    for patient in first:
        outgoing = (
            patient.portal_flow_l_h
            + patient.hepatic_artery_flow_l_h
            + patient.renal_flow_l_h
            + patient.rest_flow_l_h
        )
        assert np.isclose(outgoing, patient.cardiac_output_l_h)
        assert patient.rest_flow_l_h > 0


def test_pk_validation_recovers_exact_prediction() -> None:
    prediction = pd.DataFrame(
        {
            "time_h": [0.0, 1.0, 2.0, 4.0],
            "plasma_parent_mg_l_p50": [0.0, 1.0, 0.5, 0.1],
        }
    )
    observed = pd.DataFrame(
        {
            "time_h": [1.0, 2.0, 4.0],
            "plasma_parent_mg_l": [1.0, 0.5, 0.1],
        }
    )
    comparison, metrics = validate_pk_prediction(prediction, observed)
    assert len(comparison) == 3
    assert metrics["rmse_mg_l"] == 0.0
    assert np.isclose(metrics["absolute_average_fold_error"], 1.0)
    assert metrics["fraction_within_1_5_fold"] == 1.0
    assert np.isclose(metrics["cmax_predicted_observed_ratio"], 1.0)
    assert np.isclose(metrics["auc_predicted_observed_ratio"], 1.0)


def test_pk_validation_rejects_missing_observations() -> None:
    prediction = pd.DataFrame(
        {
            "time_h": [0.0, 1.0, 2.0],
            "plasma_parent_mg_l_p50": [0.0, 1.0, 0.5],
        }
    )
    observed = pd.DataFrame(
        {
            "time_h": [1.0, 2.0],
            "plasma_parent_mg_l": [1.0, float("nan")],
        }
    )
    with pytest.raises(ValueError, match="must be finite"):
        validate_pk_prediction(prediction, observed)


def test_drug_configuration_is_structure_grounded_and_provenanced() -> None:
    drug = load_drug("configs/virtual_phase1/synthetic_probe.yaml")
    assert drug.smiles
    assert "Synthetic demonstration" in drug.parameter_provenance
    assert np.isclose(sum(drug.normalized_zone_activity), 1.0)
