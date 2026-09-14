from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from our_star.models.amount_simulation import simulate_amount_patient
from our_star.models.hepatic_pathway import (
    HepaticPathwayParameters,
    HepaticPathwayPBPKModel,
    HepaticProductPathwaySpec,
    build_hepatic_pathway_network,
)
from our_star.models.segmented import FormulationParameters
from our_star.pbpk import DoseEvent, DrugParameters
from our_star.virtual_population import reference_patient
from our_star.virtual_trial import SolverSettings


ROOT = Path(__file__).resolve().parents[1]


def _components(*, factors=None):
    drug = DrugParameters(
        name="synthetic replacement candidate",
        smiles="CCO",
        molecular_weight_g_mol=100.0,
        fraction_unbound_plasma=0.7,
        absorption_rate_h=1.0,
        fraction_absorbed=0.9,
        kp_gut=1.0,
        kp_liver=1.2,
        kp_kidney=1.0,
        kp_rest=1.0,
        liver_vmax_mg_h=0.0,
        liver_km_mg_l=1.0,
        liver_zone_activity=(0.2, 0.3, 0.5),
        gut_vmax_mg_h=0.0,
        gut_km_mg_l=1.0,
        renal_clearance_l_h=0.6,
        metabolite_renal_clearance_l_h=0.0,
        metabolite_mass_yield=1.0,
        parameter_provenance="synthetic unit-test fixture",
    )
    formulation = FormulationParameters(
        formulation_id="synthetic_ir",
        dosage_form="immediate_release_tablet",
        initial_dissolved_fraction=0.2,
        dissolution_rate_h=3.0,
        gastric_emptying_rate_h=2.0,
        segment_transit_rate_h=(2.0, 1.5, 1.0),
        segment_absorption_rate_h=(3.0, 2.0, 1.0),
        wall_volume_fraction=(0.2, 0.4, 0.4),
        portal_flow_fraction=(0.2, 0.4, 0.4),
        parameter_provenance="synthetic unit-test fixture",
    )
    parameters = HepaticPathwayParameters(
        parent_species_id="candidate_parent",
        products=(
            HepaticProductPathwaySpec(
                species_id="measured_product",
                name="measured product",
                molecular_weight_g_mol=116.0,
                reaction_id="candidate_oxidation",
                mechanism_id="candidate_cyp",
                formation_clearance_l_h=3.0,
                zone_activity=(0.2, 0.3, 0.5),
                systemic=True,
                liver_partition_coefficient=1.0,
                renal_clearance_l_h=2.0,
                parameter_provenance="synthetic unit-test fixture",
            ),
            HepaticProductPathwaySpec(
                species_id="unmeasured_product",
                name="unmeasured product",
                molecular_weight_g_mol=86.0,
                reaction_id="candidate_dealkylation",
                mechanism_id="candidate_other_clearance",
                formation_clearance_l_h=1.0,
                zone_activity=(0.2, 0.3, 0.5),
                systemic=False,
                liver_partition_coefficient=1.0,
                renal_clearance_l_h=0.0,
                parameter_provenance="synthetic unit-test fixture",
            ),
        ),
        evidence_role="mixed_independent_and_bounded_engineering_priors",
        structural_limitations=("synthetic fixture has no clinical interpretation",),
        parameter_provenance="synthetic unit-test fixture",
    )
    return drug, formulation, parameters, HepaticPathwayPBPKModel(
        drug, formulation, parameters, mechanism_factors=factors
    )


def _simulate(*, factors=None):
    drug, _formulation, _parameters, model = _components(factors=factors)
    return simulate_amount_patient(
        reference_patient(),
        drug,
        (DoseEvent(0.0, 100.0),),
        model=model,
        duration_h=12.0,
        output_interval_h=0.25,
        solver=SolverSettings(max_step_h=0.05),
    )


def test_generic_pathway_reactions_close_parent_moiety_and_molecular_mass() -> None:
    drug, _formulation, parameters, _model = _components()
    network = build_hepatic_pathway_network(drug, parameters)
    assert {item.reaction_id for item in network.balance_diagnostics} == {
        "candidate_oxidation",
        "candidate_dealkylation",
    }
    assert max(
        abs(item.closure_error_mg_per_umol)
        for item in network.balance_diagnostics
    ) < 1.0e-12
    assert set(network.external_participants) == {
        "net_mass_equivalent_candidate_oxidation",
        "net_mass_equivalent_candidate_dealkylation",
    }


def test_generic_pathway_simulation_forms_product_and_closes_mass() -> None:
    result = _simulate()
    trajectory = result.trajectory
    assert trajectory["plasma_measured_product_mg_l"].max() > 0.0
    assert trajectory["total_unmeasured_product_umol"].max() > 0.0
    assert result.summary["max_abs_mass_balance_error_mg"] < 1.0e-5
    assert trajectory["augmented_minus_moiety_mass_mg"].abs().max() < 1.0e-5
    assert trajectory["candidate_screening_model_not_clinically_validated"].eq(1.0).all()
    assert result.states_umol.min() >= 0.0


@pytest.mark.parametrize(
    ("factor", "column"),
    (
        ({"candidate_cyp": 0.0}, "total_measured_product_umol"),
        ({"candidate_other_clearance": 0.0}, "total_unmeasured_product_umol"),
    ),
)
def test_mechanism_ablation_eliminates_only_named_product(factor, column) -> None:
    result = _simulate(factors=factor)
    assert result.trajectory[column].max() < 1.0e-10
    other = (
        "total_unmeasured_product_umol"
        if column == "total_measured_product_umol"
        else "total_measured_product_umol"
    )
    assert result.trajectory[other].max() > 0.0
    assert result.summary["max_abs_mass_balance_error_mg"] < 1.0e-5


def test_strict_config_parser_rejects_unknown_keys_and_missing_limitations() -> None:
    _drug, _formulation, parameters, _model = _components()
    raw = {
        "parent_species_id": parameters.parent_species_id,
        "products": [
            {
                "species_id": item.species_id,
                "name": item.name,
                "molecular_weight_g_mol": item.molecular_weight_g_mol,
                "reaction_id": item.reaction_id,
                "mechanism_id": item.mechanism_id,
                "formation_clearance_l_h": item.formation_clearance_l_h,
                "zone_activity": list(item.zone_activity),
                "systemic": item.systemic,
                "liver_partition_coefficient": item.liver_partition_coefficient,
                "renal_clearance_l_h": item.renal_clearance_l_h,
                "parameter_provenance": item.parameter_provenance,
            }
            for item in parameters.products
        ],
        "evidence_role": parameters.evidence_role,
        "structural_limitations": list(parameters.structural_limitations),
        "parameter_provenance": parameters.parameter_provenance,
    }
    parsed = HepaticPathwayParameters.from_mapping(raw)
    assert parsed == parameters
    with pytest.raises(ValueError, match="keys differ"):
        HepaticPathwayParameters.from_mapping({**raw, "outcome_fit": 1.0})
    with pytest.raises(ValueError, match="structural_limitations"):
        HepaticPathwayParameters.from_mapping({**raw, "structural_limitations": []})


def test_parent_only_lane_may_use_only_a_non_systemic_mass_balanced_sink() -> None:
    drug, formulation, parameters, _model = _components()
    parent_only = HepaticPathwayParameters(
        parent_species_id=parameters.parent_species_id,
        products=(parameters.products[1],),
        evidence_role=parameters.evidence_role,
        structural_limitations=parameters.structural_limitations,
        parameter_provenance=parameters.parameter_provenance,
    )
    model = HepaticPathwayPBPKModel(drug, formulation, parent_only)
    result = simulate_amount_patient(
        reference_patient(),
        drug,
        (DoseEvent(0.0, 100.0),),
        model=model,
        duration_h=6.0,
        output_interval_h=0.25,
        solver=SolverSettings(max_step_h=0.05),
    )
    assert set(model.central_states) == {parameters.parent_species_id}
    assert "plasma_unmeasured_product_mg_l" not in result.trajectory
    assert result.trajectory["total_unmeasured_product_umol"].max() > 0.0
    assert result.summary["max_abs_mass_balance_error_mg"] < 1.0e-5
