from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from our_star.core.species import AmountUnit
from our_star.models.amount_simulation import simulate_amount_patient
from our_star.models.legacy import LegacyPBPKModel
from our_star.models.paracetamol_pathways import (
    ParacetamolPathwayPBPKModel,
    ParacetamolPathwayParameters,
)
from our_star.pbpk import DoseEvent
from our_star.virtual_population import reference_patient
from our_star.virtual_trial import SolverSettings, load_drug, load_formulation, load_yaml


ROOT = Path(__file__).resolve().parents[1]


def _molar_model() -> tuple[object, ParacetamolPathwayPBPKModel]:
    drug = load_drug(ROOT / "configs/v04/drugs/paracetamol_tier_a_scaffold.yaml")
    formulation = load_formulation(
        ROOT / "configs/v04/formulations/paracetamol_ir_tier_a_scaffold.yaml"
    )
    pathway = ParacetamolPathwayParameters.from_mapping(
        load_yaml(ROOT / "configs/v04/reactions/paracetamol_tier_a_scaffold.yaml")[
            "pathway_model"
        ]
    )
    return drug, ParacetamolPathwayPBPKModel(drug, formulation, pathway)


def test_amount_simulation_exposes_read_only_umol_states() -> None:
    drug, model = _molar_model()
    result = simulate_amount_patient(
        reference_patient(),
        drug,
        (DoseEvent(0.0, 1000.0, "oral"),),
        model=model,
        duration_h=1.0,
        output_interval_h=0.25,
        solver=SolverSettings(max_step_h=0.05),
    )
    assert result.state_amount_unit == AmountUnit.UMOL
    assert result.states_umol.shape[0] == result.times_h.size
    assert np.all(result.states_umol >= 0.0)
    assert not hasattr(result, "states_mg")
    with pytest.raises(ValueError, match="read-only"):
        result.states_umol[0, 0] = 1.0


def test_amount_simulation_rejects_legacy_mass_model_before_running() -> None:
    drug, _model = _molar_model()
    with pytest.raises(TypeError, match="MultiSpeciesPBPKModel"):
        simulate_amount_patient(
            reference_patient(),
            drug,
            (DoseEvent(0.0, 1000.0, "oral"),),
            model=LegacyPBPKModel(drug),  # type: ignore[arg-type]
            duration_h=1.0,
        )
