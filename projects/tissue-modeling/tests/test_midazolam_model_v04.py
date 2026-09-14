from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from our_star.models.amount_simulation import simulate_amount_patient
from our_star.models.midazolam import (
    GLUCURONIDE,
    HYDROXY,
    MINOR_SINK,
    MidazolamMechanismParameters,
    MidazolamMechanisticPBPKModel,
    build_midazolam_reaction_network,
)
from our_star.models.segmented import FormulationParameters
from our_star.pbpk import DoseEvent, DrugParameters
from our_star.virtual_population import reference_patient
from our_star.virtual_trial import SolverSettings


ROOT = Path(__file__).resolve().parents[1]


def _components(*, factors=None, ugt_clearance=100.0):
    raw = yaml.safe_load(
        (
            ROOT
            / "configs/v04/replacement_candidates/midazolam_tanaka2014_zero_tuning.yaml"
        ).read_text(encoding="utf-8")
    )
    mechanism = yaml.safe_load(
        (
            ROOT / "configs/v04/mechanisms/midazolam_cyp3a_ugt.yaml"
        ).read_text(encoding="utf-8")
    )
    drug = DrugParameters.from_mapping(raw["drug"])
    formulation = FormulationParameters.from_mapping(raw["formulation"])
    fixed = mechanism["fixed_parameters"]
    parameters = MidazolamMechanismParameters(
        hydroxy_molecular_weight_g_mol=mechanism["species"]["primary_product"][
            "molecular_weight_g_mol"
        ],
        glucuronide_molecular_weight_g_mol=mechanism["species"][
            "conjugate_sink"
        ]["molecular_weight_g_mol"],
        hydroxy_fraction_unbound=fixed["hydroxy_fraction_unbound"],
        hydroxy_kp_gut=fixed["hydroxy_kp_gut"],
        hydroxy_kp_liver=fixed["hydroxy_kp_liver"],
        hydroxy_kp_kidney=fixed["hydroxy_kp_kidney"],
        hydroxy_kp_rest=fixed["hydroxy_kp_rest"],
        intestinal_cyp3a_unbound_clearance_l_h=fixed[
            "intestinal_cyp3a_unbound_clearance_l_h"
        ],
        hepatic_cyp3a_unbound_clearance_l_h=fixed[
            "hepatic_cyp3a_unbound_clearance_l_h"
        ],
        major_formation_fraction=fixed["major_formation_fraction"],
        ugt_unbound_clearance_l_h=ugt_clearance,
        effective_ugt_zone_activity=tuple(fixed["effective_ugt_zone_activity"]),
        provenance="test load of frozen v0.4 mechanism candidate",
    )
    return (
        drug,
        formulation,
        parameters,
        MidazolamMechanisticPBPKModel(
            drug,
            formulation,
            parameters,
            mechanism_factors=factors,
        ),
    )


def _simulate(*, factors=None, ugt_clearance=100.0, route="oral"):
    drug, _formulation, _parameters, model = _components(
        factors=factors, ugt_clearance=ugt_clearance
    )
    return simulate_amount_patient(
        reference_patient(),
        drug,
        (DoseEvent(0.0, 1.0, route),),
        model=model,
        duration_h=12.0,
        output_interval_h=0.25,
        solver=SolverSettings(
            max_step_h=0.02,
            relative_tolerance=1.0e-8,
            absolute_tolerance=1.0e-10,
        ),
    )


def test_midazolam_reaction_network_closes_both_mass_ledgers() -> None:
    drug, _formulation, parameters, _model = _components()
    network = build_midazolam_reaction_network(
        drug.molecular_weight_g_mol, parameters
    )
    assert {item.reaction_id for item in network.reactions} == {
        "midazolam_to_one_hydroxymidazolam",
        "one_hydroxymidazolam_to_glucuronide",
        "midazolam_to_minor_parent_moiety_sink",
    }
    assert set(network.external_participants) == {
        "cyp3a_net_oxygen_transfer",
        "ugt_net_glucuronyl_transfer",
    }
    assert max(
        abs(item.closure_error_mg_per_umol)
        for item in network.balance_diagnostics
    ) < 1.0e-12


def test_explicit_midazolam_model_forms_hydroxy_and_glucuronide() -> None:
    result = _simulate()
    trajectory = result.trajectory
    assert trajectory[f"plasma_{HYDROXY}_mg_l"].max() > 0.0
    assert trajectory[f"total_{GLUCURONIDE}_umol"].max() > 0.0
    assert trajectory[f"total_{MINOR_SINK}_umol"].max() > 0.0
    assert result.summary["max_abs_mass_balance_error_mg"] < 1.0e-5
    assert trajectory["augmented_minus_moiety_mass_mg"].abs().max() < 1.0e-5
    assert trajectory["research_only_not_clinically_validated"].eq(1.0).all()
    assert result.states_umol.min() >= -1.0e-12


def test_cyp3a_and_ugt_ablation_signatures_are_specific() -> None:
    baseline = _simulate()
    gut_off = _simulate(factors={"intestinal_cyp3a": 0.0})
    liver_off = _simulate(factors={"hepatic_cyp3a": 0.0})
    ugt_off = _simulate(factors={"effective_ugt": 0.0})

    parent_auc = np.trapezoid(
        baseline.trajectory["plasma_parent_mg_l"], baseline.times_h
    )
    gut_off_auc = np.trapezoid(
        gut_off.trajectory["plasma_parent_mg_l"], gut_off.times_h
    )
    liver_off_auc = np.trapezoid(
        liver_off.trajectory["plasma_parent_mg_l"], liver_off.times_h
    )
    assert gut_off_auc > parent_auc
    assert liver_off_auc > parent_auc
    assert ugt_off.trajectory[f"total_{GLUCURONIDE}_umol"].max() < 1.0e-10
    assert (
        ugt_off.trajectory[f"plasma_{HYDROXY}_mg_l"].max()
        > baseline.trajectory[f"plasma_{HYDROXY}_mg_l"].max()
    )
    for result in (baseline, gut_off, liver_off, ugt_off):
        assert result.summary["max_abs_mass_balance_error_mg"] < 1.0e-5


def test_intestinal_cyp3a_iv_effect_is_only_systemic_recircular_passage() -> None:
    baseline = _simulate(route="iv_bolus")
    gut_off = _simulate(route="iv_bolus", factors={"intestinal_cyp3a": 0.0})
    oral = _simulate(route="oral")
    oral_gut_off = _simulate(
        route="oral", factors={"intestinal_cyp3a": 0.0}
    )
    iv_parent_auc = np.trapezoid(
        baseline.trajectory["plasma_parent_mg_l"], baseline.times_h
    )
    iv_off_auc = np.trapezoid(
        gut_off.trajectory["plasma_parent_mg_l"], gut_off.times_h
    )
    oral_parent_auc = np.trapezoid(
        oral.trajectory["plasma_parent_mg_l"], oral.times_h
    )
    oral_off_auc = np.trapezoid(
        oral_gut_off.trajectory["plasma_parent_mg_l"], oral_gut_off.times_h
    )
    # Systemic drug perfuses the gut wall, so intestinal enzyme may still act
    # after IV administration.  Only the oral route has the additional direct
    # luminal-to-enterocyte first-pass component.
    assert iv_off_auc > iv_parent_auc
    assert oral_off_auc > oral_parent_auc
    assert (oral_off_auc / oral_parent_auc) > (iv_off_auc / iv_parent_auc)


def test_mechanism_parameter_contract_rejects_unphysical_values() -> None:
    _drug, _formulation, parameters, _model = _components()
    with pytest.raises(ValueError, match="major_formation_fraction"):
        MidazolamMechanismParameters(
            **{
                **parameters.__dict__,
                "major_formation_fraction": 1.0,
            }
        )
    with pytest.raises(ValueError, match="effective_ugt_zone_activity"):
        MidazolamMechanismParameters(
            **{
                **parameters.__dict__,
                "effective_ugt_zone_activity": (0.0, 0.0, 0.0),
            }
        )
