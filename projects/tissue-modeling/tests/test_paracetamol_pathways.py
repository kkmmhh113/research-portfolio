from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from our_star.chemistry.kinetics import evaluate_competing_pathways
from our_star.models.paracetamol_pathways import (
    CYP_REACTION,
    PARACETAMOL,
    PARACETAMOL_GLUCURONIDE,
    PARACETAMOL_OXIDATIVE_CONJUGATE,
    PARACETAMOL_SULFATE,
    SULT_REACTION,
    UGT_REACTION,
    ParacetamolPathwayPBPKModel,
    ParacetamolPathwayParameters,
    build_paracetamol_tier_a_network,
)
from our_star.models.paracetamol_pathways import _ParacetamolLiverModule
from our_star.pbpk import DoseEvent
from our_star.virtual_population import reference_patient
from our_star.virtual_trial import (
    SolverSettings,
    load_drug,
    load_formulation,
    load_yaml,
    simulate_patient,
)


ROOT = Path(__file__).resolve().parents[1]


def _components(*, mechanism_factors=None):
    drug = load_drug(
        ROOT / "configs/v04/drugs/paracetamol_tier_a_scaffold.yaml"
    )
    formulation = load_formulation(
        ROOT
        / "configs/v04/formulations/paracetamol_ir_tier_a_scaffold.yaml"
    )
    raw = load_yaml(
        ROOT / "configs/v04/reactions/paracetamol_tier_a_scaffold.yaml"
    )
    parameters = ParacetamolPathwayParameters.from_mapping(raw["pathway_model"])
    model = ParacetamolPathwayPBPKModel(
        drug,
        formulation,
        parameters,
        mechanism_factors=mechanism_factors,
    )
    return drug, formulation, parameters, model


def _simulate(*, mechanism_factors=None, route="oral", patient=None):
    drug, _formulation, _parameters, model = _components(
        mechanism_factors=mechanism_factors
    )
    return simulate_patient(
        patient or reference_patient(),
        drug,
        (DoseEvent(0.0, 1000.0, route),),
        duration_h=24.0,
        output_interval_h=0.25,
        solver=SolverSettings(max_step_h=0.05),
        model=model,
    )


def test_paracetamol_open_reactions_close_mass_and_elements() -> None:
    network = build_paracetamol_tier_a_network()
    assert set(network.species.ids) == {
        PARACETAMOL,
        PARACETAMOL_GLUCURONIDE,
        PARACETAMOL_SULFATE,
        PARACETAMOL_OXIDATIVE_CONJUGATE,
    }
    assert {item.reaction_id for item in network.balance_diagnostics} == {
        UGT_REACTION,
        SULT_REACTION,
        CYP_REACTION,
    }
    assert all(item.elemental_balance_checked for item in network.balance_diagnostics)
    assert max(
        abs(item.closure_error_mg_per_umol)
        for item in network.balance_diagnostics
    ) < 1e-12
    assert all(
        max((abs(value) for value in item.elemental_residuals.values()), default=0.0)
        < 1e-12
        for item in network.balance_diagnostics
    )


def test_dose_boundary_converts_mg_to_umol_and_both_ledgers_equal_dose() -> None:
    drug, _formulation, _parameters, model = _components()
    dosed = model.apply_doses(model.initial_state(), (DoseEvent(0.0, 1000.0),))
    report = model.concentration_record(dosed, reference_patient(), drug)
    assert model.state_amount_unit == "umol"
    assert report["tracked_mass_mg"] == pytest.approx(1000.0, abs=1e-10)
    assert report["augmented_accounted_mass_mg"] == pytest.approx(
        1000.0, abs=1e-10
    )
    assert report["external_import_mass_mg"] == 0.0
    assert report["external_export_mass_mg"] == 0.0


def test_sult_fraction_falls_as_lower_km_pathway_saturates() -> None:
    _drug, _formulation, parameters, model = _components()
    liver = next(
        module
        for module in model.assembler.modules
        if isinstance(module, _ParacetamolLiverModule)
    )
    low = evaluate_competing_pathways(1.0, liver._pathways_by_zone[0])
    high = evaluate_competing_pathways(10_000.0, liver._pathways_by_zone[0])
    low_fraction = low[SULT_REACTION] / sum(low.values())
    high_fraction = high[SULT_REACTION] / sum(high.values())
    assert parameters.sult.km_umol_l < parameters.ugt.km_umol_l
    assert high_fraction < low_fraction


def test_tier_a_simulation_forms_all_products_and_closes_both_ledgers() -> None:
    result = _simulate()
    trajectory = result.trajectory
    assert trajectory[f"plasma_{PARACETAMOL_GLUCURONIDE}_mg_l"].max() > 0.0
    assert trajectory[f"plasma_{PARACETAMOL_SULFATE}_mg_l"].max() > 0.0
    assert (
        trajectory[f"plasma_{PARACETAMOL_OXIDATIVE_CONJUGATE}_mg_l"].max()
        > 0.0
    )
    assert result.summary["max_abs_mass_balance_error_mg"] < 1.0e-5
    assert trajectory["augmented_minus_moiety_mass_mg"].abs().max() < 1.0e-5
    assert trajectory["reaction_static_max_abs_closure_mg_per_umol"].max() < 1e-12
    assert result.states_mg.min() >= 0.0


@pytest.mark.parametrize(
    ("mechanism_factors", "species_id"),
    (
        ({"ugt_activity": 0.0}, PARACETAMOL_GLUCURONIDE),
        ({"sult_activity": 0.0}, PARACETAMOL_SULFATE),
        ({"cyp2e1_activity": 0.0}, PARACETAMOL_OXIDATIVE_CONJUGATE),
    ),
)
def test_disabling_one_pathway_eliminates_only_its_product(
    mechanism_factors, species_id
) -> None:
    result = _simulate(mechanism_factors=mechanism_factors)
    assert result.trajectory[f"total_{species_id}_umol"].max() < 1.0e-10
    other_products = {
        PARACETAMOL_GLUCURONIDE,
        PARACETAMOL_SULFATE,
        PARACETAMOL_OXIDATIVE_CONJUGATE,
    } - {species_id}
    assert all(
        result.trajectory[f"total_{other}_umol"].max() > 0.0
        for other in other_products
    )
    assert result.summary["max_abs_mass_balance_error_mg"] < 1.0e-5


def test_iv_result_is_independent_of_empty_gi_transport_parameters() -> None:
    drug, formulation, parameters, baseline_model = _components()
    changed_formulation = replace(
        formulation,
        dissolution_rate_h=formulation.dissolution_rate_h * 10.0,
        gastric_emptying_rate_h=formulation.gastric_emptying_rate_h * 0.1,
        segment_transit_rate_h=tuple(
            value * 2.0 for value in formulation.segment_transit_rate_h
        ),
        segment_absorption_rate_h=tuple(
            value * 0.5 for value in formulation.segment_absorption_rate_h
        ),
    )
    changed_model = ParacetamolPathwayPBPKModel(
        drug, changed_formulation, parameters
    )
    kwargs = dict(
        duration_h=6.0,
        output_interval_h=0.25,
        solver=SolverSettings(max_step_h=0.05),
    )
    baseline = simulate_patient(
        reference_patient(),
        drug,
        (DoseEvent(0.0, 1000.0, "iv_bolus"),),
        model=baseline_model,
        **kwargs,
    )
    changed = simulate_patient(
        reference_patient(),
        drug,
        (DoseEvent(0.0, 1000.0, "iv_bolus"),),
        model=changed_model,
        **kwargs,
    )
    assert np.allclose(
        baseline.trajectory["plasma_parent_mg_l"],
        changed.trajectory["plasma_parent_mg_l"],
        rtol=0.0,
        atol=1e-10,
    )


def test_renal_impairment_reduces_early_cumulative_metabolite_urine() -> None:
    normal = _simulate()
    impaired_patient = replace(
        reference_patient(),
        patient_id="renal_impairment_directionality",
        renal_function_fraction=0.25,
    )
    impaired = _simulate(patient=impaired_patient)
    # This is a local mechanism directionality check, not a claim about the
    # eventual recovery fraction: reduced parent renal loss can increase later
    # hepatic formation. At the frozen early horizon, the renal-scale effect
    # on already formed conjugates must point downward.
    normal_8h = normal.trajectory.loc[
        normal.trajectory["time_h"] == 8.0, "urine_metabolite_mg"
    ].iloc[0]
    impaired_8h = impaired.trajectory.loc[
        impaired.trajectory["time_h"] == 8.0, "urine_metabolite_mg"
    ].iloc[0]
    assert impaired_8h < normal_8h
    assert impaired.summary["max_abs_mass_balance_error_mg"] < 1.0e-5


def test_v04_pathway_config_is_explicitly_nonclinical_scaffold() -> None:
    raw = load_yaml(
        ROOT / "configs/v04/reactions/paracetamol_tier_a_scaffold.yaml"
    )["pathway_model"]
    assert raw["evidence_role"] == "mixed_independent_and_synthetic_scaffold"
    assert "SYNTHETIC PLACEHOLDER" in raw["pathways"]["sult"][
        "parameter_provenance"
    ]
    synthetic = load_yaml(
        ROOT / "configs/v04/reactions/synthetic_conjugation_engine_check.yaml"
    )
    assert synthetic["synthetic_open_reaction"]["model_status"] == (
        "synthetic_engine_check_not_a_drug"
    )
