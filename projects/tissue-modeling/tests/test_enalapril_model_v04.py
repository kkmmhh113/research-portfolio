from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from our_star.mechanisms import SaturableACEBinding
from our_star.models.enalapril import (
    EnalaprilMechanismPBPKModel,
    load_enalapril_mechanism_parameters,
)
from our_star.pbpk import DoseEvent
from our_star.population.adapters import materialize_patient_physiology
from our_star.population.specification import VirtualSubject
from our_star.prediction.pbpk import (
    TrajectoryEndpoint,
    build_pbpk_simulation_callback,
)
from our_star.virtual_population import reference_patient
from our_star.virtual_trial import (
    SolverSettings,
    load_drug,
    load_formulation,
    simulate_patient,
)


ROOT = Path(__file__).resolve().parents[1]


def _inputs(*, ace: bool = False):
    drug = load_drug(ROOT / "configs/v03/drugs/enalapril_zero_tuning.yaml")
    formulation = load_formulation(
        ROOT / "configs/v03/formulations/enalapril_ir_zero_tuning.yaml"
    )
    mechanism = load_enalapril_mechanism_parameters(
        ROOT / "configs/v04/mechanisms/enalapril_formation.yaml",
        ROOT / "configs/v04/mechanisms/enalaprilat_renal.yaml",
        ace_path=ROOT / "configs/v04/mechanisms/enalaprilat_ace_binding.yaml",
    )
    if ace:
        mechanism = replace(
            mechanism,
            ace_binding=SaturableACEBinding(
                bmax_umol=1.0,
                kon_l_umol_h=0.5,
                koff_h=0.2,
                provenance="synthetic ACE integration fixture only",
            ),
        )
    return drug, formulation, mechanism


def _simulate(*, factors=None, ace=False, route="oral"):
    drug, formulation, mechanism = _inputs(ace=ace)
    model = EnalaprilMechanismPBPKModel(
        drug,
        formulation,
        mechanism,
        mechanism_factors=factors,
    )
    return model, simulate_patient(
        reference_patient(),
        drug,
        (DoseEvent(0.0, 7.64, route),),
        duration_h=24.0,
        output_interval_h=0.25,
        solver=SolverSettings(max_step_h=0.05),
        model=model,
    )


def test_distributed_enalapril_model_forms_named_metabolite_and_closes_ledgers() -> None:
    model, result = _simulate()
    assert model.model_id == "our_star.enalapril_multispecies.m2.v0.4"
    assert result.trajectory["plasma_enalaprilat_mg_l"].max() > 0.0
    assert result.trajectory["liver_zone1_parent_mg_l"].max() > 0.0
    assert result.summary["max_abs_mass_balance_error_mg"] < 1.0e-5
    assert result.trajectory["reaction_static_max_abs_closure_mg_per_umol"].max() < 1.0e-9
    assert np.allclose(
        result.trajectory["augmented_accounted_mass_mg"],
        result.trajectory["drug_equivalent_mass_mg"],
        rtol=0.0,
        atol=1.0e-9,
    )
    assert np.all(result.states_mg >= -1.0e-8)


def test_filtration_and_secretion_sinks_are_separate_and_ablatable() -> None:
    _normal_model, normal = _simulate()
    _off_model, secretion_off = _simulate(factors={"enalaprilat_secretion": 0.0})
    normal_final = normal.trajectory.iloc[-1]
    off_final = secretion_off.trajectory.iloc[-1]
    assert normal_final["urine_enalaprilat_filtration_umol"] > 0.0
    assert normal_final["urine_enalaprilat_secretion_umol"] > 0.0
    assert off_final["urine_enalaprilat_secretion_umol"] == pytest.approx(
        0.0, abs=1.0e-12
    )
    assert off_final["urine_enalaprilat_umol"] < normal_final["urine_enalaprilat_umol"]
    assert secretion_off.summary["max_abs_mass_balance_error_mg"] < 1.0e-5


def test_optional_ace_binding_is_mass_conserving_and_visible_to_observation() -> None:
    model, result = _simulate(ace=True)
    assert model.model_id == "our_star.enalapril_multispecies.m3.v0.4"
    assert result.trajectory["ace_bound_enalaprilat_umol"].max() > 0.0
    assert np.all(
        result.trajectory["observation_enalaprilat_mg_l"]
        >= result.trajectory["plasma_enalaprilat_mg_l"] - 1.0e-14
    )
    assert result.summary["max_abs_mass_balance_error_mg"] < 1.0e-5


def test_iv_simulation_is_independent_of_oral_formulation_rates() -> None:
    drug, formulation, mechanism = _inputs()
    alternative = replace(
        formulation,
        dissolution_rate_h=formulation.dissolution_rate_h * 10.0,
        gastric_emptying_rate_h=formulation.gastric_emptying_rate_h * 0.2,
    )
    shared = dict(
        duration_h=8.0,
        output_interval_h=0.25,
        solver=SolverSettings(max_step_h=0.05),
    )
    first = simulate_patient(
        reference_patient(),
        drug,
        (DoseEvent(0.0, 7.64, "iv_bolus"),),
        model=EnalaprilMechanismPBPKModel(drug, formulation, mechanism),
        **shared,
    )
    second = simulate_patient(
        reference_patient(),
        drug,
        (DoseEvent(0.0, 7.64, "iv_bolus"),),
        model=EnalaprilMechanismPBPKModel(drug, alternative, mechanism),
        **shared,
    )
    assert np.allclose(first.states_mg, second.states_mg, rtol=0.0, atol=1.0e-12)


def test_population_adapter_and_pbpk_callback_keep_mechanisms_separate() -> None:
    subject = VirtualSubject(
        subject_id="synthetic_subject_0001",
        population_id="synthetic_v04",
        draw_index=0,
        sampling_seed=42,
        latent_effects={"weight": 0.0, "ces1": 0.0},
        anatomy_modifiers={"body_weight_kg": 80.0},
        mechanism_modifiers={"ces1_activity": 0.8},
    )
    patient = materialize_patient_physiology(subject, reference_patient())
    assert patient.body_weight_kg == 80.0
    assert not hasattr(patient, "ces1_activity")

    drug, formulation, mechanism = _inputs()
    callback = build_pbpk_simulation_callback(
        reference_patient=reference_patient(),
        drug=drug,
        doses=(DoseEvent(0.0, 7.64, "oral"),),
        model_factory=lambda factors: EnalaprilMechanismPBPKModel(
            drug,
            formulation,
            mechanism,
            mechanism_factors=factors,
        ),
        endpoints=(
            TrajectoryEndpoint(
                quantity="enalaprilat_cmax_mg_l",
                column="observation_enalaprilat_mg_l",
                statistic="cmax",
            ),
            TrajectoryEndpoint(
                quantity="enalapril_auc_mg_h_l",
                column="plasma_parent_mg_l",
                statistic="auc_last",
            ),
        ),
        duration_h=8.0,
        output_interval_h=0.5,
        solver=SolverSettings(max_step_h=0.1),
    )
    values = callback(subject, 0, np.random.default_rng(1))
    assert set(values) == {"enalaprilat_cmax_mg_l", "enalapril_auc_mg_h_l"}
    assert all(np.isfinite(value) and value > 0.0 for value in values.values())


def test_cardiac_output_adapter_rejects_inconsistent_flow_components() -> None:
    subject = VirtualSubject(
        subject_id="bad_flow_subject",
        population_id="synthetic_v04",
        draw_index=0,
        sampling_seed=1,
        latent_effects={"flow": 0.0},
        anatomy_modifiers={
            "cardiac_output_l_h": 300.0,
            "portal_flow_l_h": 70.0,
            "hepatic_artery_flow_l_h": 20.0,
            "renal_flow_l_h": 70.0,
            "rest_flow_l_h": 100.0,
        },
        mechanism_modifiers={},
    )
    with pytest.raises(ValueError, match="cardiac output"):
        materialize_patient_physiology(subject, reference_patient())
