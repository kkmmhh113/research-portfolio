from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from our_star.core import AmountUnit, ChemicalSpecies
from our_star.benchmark_lock import evaluate_acceptance_gates
from our_star.models import LegacyPBPKModel, SegmentedPBPKModel
from our_star.pbpk import DoseEvent, STATE_COUNT, legacy_pbpk_rhs, pbpk_rhs
from our_star.virtual_population import reference_patient
from our_star.virtual_trial import (
    SolverSettings,
    load_drug,
    load_formulation,
    simulate_patient,
)


ROOT = Path(__file__).resolve().parents[1]


def test_species_registry_unit_conversion_round_trip() -> None:
    caffeine = ChemicalSpecies(
        species_id="caffeine",
        name="caffeine",
        molecular_weight_g_mol=194.19,
        structure="Cn1c(=O)c2c(ncn2C)n(C)c1=O",
    )
    amount_umol = caffeine.convert_amount(194.19, AmountUnit.MG, AmountUnit.UMOL)
    assert np.isclose(amount_umol, 1000.0)
    assert np.isclose(
        caffeine.convert_amount(amount_umol, AmountUnit.UMOL, AmountUnit.MG),
        194.19,
    )


def test_modular_rhs_matches_independent_monolithic_reference() -> None:
    drug = load_drug(ROOT / "configs/virtual_phase1/synthetic_probe.yaml")
    patient = reference_patient()
    random = np.random.default_rng(20260807)
    for _ in range(20):
        state = random.uniform(0.0, 100.0, size=STATE_COUNT)
        expected = legacy_pbpk_rhs(3.7, state, patient, drug)
        actual = pbpk_rhs(3.7, state, patient, drug)
        np.testing.assert_allclose(actual, expected, rtol=5e-14, atol=3e-12)
        assert abs(float(actual.sum())) < 1e-10


def test_modular_trajectory_matches_frozen_legacy_summary() -> None:
    golden = json.loads(
        (ROOT / "tests/golden/legacy_v0_1_reference.json").read_text(encoding="utf-8")
    )
    drug = load_drug(ROOT / golden["drug_config"])
    result = simulate_patient(
        reference_patient(),
        drug,
        (DoseEvent(0.0, golden["dose_mg"]),),
        duration_h=golden["duration_h"],
        output_interval_h=golden["output_interval_h"],
        model=LegacyPBPKModel(drug),
    )
    for metric, expected in golden["healthy"].items():
        assert np.isclose(result.summary[metric], expected, rtol=2e-9, atol=2e-9)


def _segmented_result(formulation, *, route: str = "oral"):
    drug = load_drug(ROOT / "configs/virtual_phase1/synthetic_probe.yaml")
    return simulate_patient(
        reference_patient(),
        drug,
        (DoseEvent(0.0, 100.0, route=route),),
        duration_h=12.0,
        output_interval_h=0.1,
        solver=SolverSettings(max_step_h=0.05),
        model=SegmentedPBPKModel(drug, formulation),
    )


def test_segmented_gi_conserves_mass_and_never_reports_negative_amounts() -> None:
    formulation = load_formulation(
        ROOT / "configs/formulations/synthetic_probe_ir.yaml"
    )
    result = _segmented_result(formulation)
    assert result.states_mg.min() >= 0.0
    assert result.summary["max_abs_mass_balance_error_mg"] < 1e-5
    assert result.model_id == "our_star.segmented_gi_pbpk.v0.2"


def test_segmented_gi_parameter_changes_have_expected_directions() -> None:
    formulation = load_formulation(
        ROOT / "configs/formulations/synthetic_probe_ir.yaml"
    )
    slow_emptying = _segmented_result(
        replace(formulation, gastric_emptying_rate_h=0.4)
    )
    fast_emptying = _segmented_result(
        replace(formulation, gastric_emptying_rate_h=8.0)
    )
    slow_dissolution = _segmented_result(
        replace(formulation, dissolution_rate_h=0.2)
    )
    fast_dissolution = _segmented_result(
        replace(formulation, dissolution_rate_h=20.0)
    )
    assert fast_emptying.summary["tmax_h"] < slow_emptying.summary["tmax_h"]
    assert fast_emptying.summary["cmax_mg_l"] > slow_emptying.summary["cmax_mg_l"]
    assert fast_dissolution.summary["cmax_mg_l"] > slow_dissolution.summary["cmax_mg_l"]


def test_iv_prediction_is_independent_of_segmented_gi_parameters() -> None:
    formulation = load_formulation(
        ROOT / "configs/formulations/synthetic_probe_ir.yaml"
    )
    slow = _segmented_result(
        replace(
            formulation,
            gastric_emptying_rate_h=0.4,
            dissolution_rate_h=0.2,
        ),
        route="iv_bolus",
    )
    fast = _segmented_result(
        replace(
            formulation,
            gastric_emptying_rate_h=8.0,
            dissolution_rate_h=20.0,
        ),
        route="iv_bolus",
    )
    np.testing.assert_allclose(slow.states_mg, fast.states_mg, rtol=0.0, atol=0.0)


def test_prespecified_benchmark_gates_fail_closed() -> None:
    criteria = {
        "aafe": {"operator": "maximum", "value": 2.0},
        "within_twofold": {"operator": "minimum", "value": 0.8},
        "auc_ratio": {
            "operator": "inclusive_range",
            "lower": 0.5,
            "upper": 2.0,
        },
    }
    results, passed = evaluate_acceptance_gates(
        {"aafe": 1.9, "within_twofold": 0.8, "auc_ratio": 2.01}, criteria
    )
    assert not passed
    assert results["aafe"]["passed"]
    assert results["within_twofold"]["passed"]
    assert not results["auc_ratio"]["passed"]

    nan_results, nan_passed = evaluate_acceptance_gates(
        {"aafe": float("nan"), "within_twofold": 1.0, "auc_ratio": 1.0},
        criteria,
    )
    assert not nan_passed
    assert not nan_results["aafe"]["passed"]
