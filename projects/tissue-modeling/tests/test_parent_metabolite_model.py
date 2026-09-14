from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from our_star.models import (
    NamedMetaboliteParameters,
    ReactionCoupledSegmentedPBPKModel,
    SegmentedPBPKModel,
)
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


def _model():
    drug = load_drug(ROOT / "configs/v03/drugs/enalapril_zero_tuning.yaml")
    formulation = load_formulation(
        ROOT / "configs/v03/formulations/enalapril_ir_zero_tuning.yaml"
    )
    raw = load_yaml(ROOT / "configs/v03/reactions/enalapril_to_enalaprilat.yaml")
    metabolite = NamedMetaboliteParameters.from_mapping(raw["metabolite"])
    return drug, formulation, ReactionCoupledSegmentedPBPKModel(
        drug, formulation, metabolite
    )


def test_explicit_enalaprilat_identity_and_reaction_mass_close() -> None:
    drug, _formulation, model = _model()
    assert model.state_registry.spec("central_metabolite").species_id == "enalaprilat"
    assert not model.state_registry.species["enalaprilat"].mass_surrogate
    derivatives, diagnostic = model.reaction_accounting(10.0)
    assert derivatives.species_umol_h["parent"] == pytest.approx(
        -10.0 * 1000.0 / drug.molecular_weight_g_mol
    )
    assert diagnostic.closure_error_mg_h == pytest.approx(0.0, abs=1e-14)


def test_reaction_coupled_simulation_forms_named_metabolite_and_conserves_dose() -> None:
    drug, _formulation, model = _model()
    result = simulate_patient(
        reference_patient(),
        drug,
        (DoseEvent(0.0, 15.28, "oral"),),
        duration_h=24.0,
        output_interval_h=0.1,
        solver=SolverSettings(max_step_h=0.1),
        model=model,
    )
    assert result.trajectory["plasma_enalaprilat_mg_l"].max() > 0.0
    assert result.summary["max_abs_mass_balance_error_mg"] < 1.0e-5
    assert np.allclose(
        result.trajectory["reaction_static_closure_error_mg_per_umol"], 0.0, atol=1e-14
    )


def test_explicit_adapter_preserves_parent_dynamics_of_mass_yield_model() -> None:
    drug, formulation, explicit = _model()
    patient = reference_patient()
    dose = (DoseEvent(0.0, 15.28, "oral"),)
    shared = dict(
        duration_h=12.0,
        output_interval_h=0.2,
        solver=SolverSettings(max_step_h=0.1),
    )
    named = simulate_patient(patient, drug, dose, model=explicit, **shared)
    lumped = simulate_patient(
        patient, drug, dose, model=SegmentedPBPKModel(drug, formulation), **shared
    )
    assert np.allclose(named.states_mg, lumped.states_mg, rtol=1e-9, atol=1e-11)


def test_one_to_one_network_rejects_inconsistent_mass_yield() -> None:
    drug, formulation, model = _model()
    from dataclasses import replace

    inconsistent = replace(drug, metabolite_mass_yield=1.0)
    with pytest.raises(ValueError, match="metabolite_mass_yield"):
        ReactionCoupledSegmentedPBPKModel(inconsistent, formulation, model.metabolite)
