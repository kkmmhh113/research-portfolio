from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.integrate import solve_ivp

from our_star.core.amount_state import SpeciesDoseEvent
from our_star.core.species import AmountUnit
from our_star.models.amount_simulation import simulate_amount_patient
from our_star.models.captopril import (
    OBSERVED_PARENT_ANALYTE_ID,
    PARENT,
    UNRESOLVED_NONRENAL,
    CaptoprilCandidateError,
    CaptoprilMechanismParameters,
    CaptoprilRenalPBPKModel,
    audit_captopril_raw_positivity,
    build_captopril_reaction_network,
    build_captopril_clinical_scoring_input,
    captopril_observation_spec_sha256,
    captopril_state,
    load_captopril_candidate,
)
from our_star.observation import ObservationSpec
from our_star.pbpk import DoseEvent
from our_star.virtual_population import reference_patient
from our_star.virtual_trial import SolverSettings


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/v05/mechanisms/captopril.yaml"


def _components(*, factors=None):
    drug, mechanism, payload = load_captopril_candidate(CONFIG)
    model = CaptoprilRenalPBPKModel(
        drug,
        mechanism,
        mechanism_factors=factors,
    )
    return drug, mechanism, payload, model


def _simulate(*, factors=None, route="iv_bolus"):
    drug, _mechanism, _payload, model = _components(factors=factors)
    result = simulate_amount_patient(
        reference_patient(),
        drug,
        (DoseEvent(0.0, 10.0, route),),
        model=model,
        duration_h=12.0,
        output_interval_h=0.1,
        solver=SolverSettings(
            max_step_h=0.01,
            relative_tolerance=1.0e-8,
            absolute_tolerance=1.0e-10,
        ),
    )
    return model, result


def _valid_observation_spec() -> ObservationSpec:
    return ObservationSpec(
        observation_id="synthetic_whole_blood_unchanged_parent",
        analyte_id=OBSERVED_PARENT_ANALYTE_ID,
        source_matrix="whole_blood",
        modeled_matrix="whole_blood",
        quantity="total",
        matrix_bridge="identity",
        fixed_matrix_ratio=None,
        bound_recovery_fraction=0.0,
        residual_model_id="synthetic_only",
        provenance="synthetic observation-layer test",
    )


def _write_config(tmp_path: Path, payload: dict[str, object]) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "captopril.yaml"
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path


def test_fixed_clearance_split_is_deterministic_and_not_a_fit() -> None:
    _drug, mechanism, payload, _model = _components()
    assert mechanism.filtration_clearance_on_total_l_h == pytest.approx(5.4375)
    assert mechanism.secretion_clearance_on_total_l_h == pytest.approx(21.7225)
    assert mechanism.secretion_clearance_on_unbound_l_h == pytest.approx(
        29.96206896551724
    )
    assert (
        mechanism.secretion_clearance_on_total_l_h
        / mechanism.renal_clearance_l_h
        >= 0.78
    )
    assert mechanism.unresolved_nonrenal_clearance_l_h == pytest.approx(28.84)
    assert payload["mechanism"]["status"] == "active_fixed_candidate_not_admitted"
    assert payload["mechanism"]["reversible_disulfide_candidate"]["enabled"] is False


def test_parent_model_closes_mass_and_preserves_nonnegative_umol_states() -> None:
    model, result = _simulate()
    assert model.model_id == "our_star.captopril_parent_renal.v0.5.dev"
    assert result.summary["max_abs_mass_balance_error_mg"] < 1.0e-5
    assert result.states_umol.min() >= -1.0e-8
    assert result.trajectory["plasma_parent_mg_l"].max() > 0.0
    assert result.trajectory["urine_captopril_total_umol"].iloc[-1] > 0.0
    assert (
        result.trajectory["unresolved_nonrenal_captopril_equivalent_umol"].iloc[-1]
        > 0.0
    )
    final = result.trajectory.iloc[-1]
    assert (
        final["urine_captopril_total_umol"]
        / final["unresolved_nonrenal_captopril_equivalent_umol"]
    ) == pytest.approx(
        model.mechanisms.renal_clearance_l_h
        / model.mechanisms.unresolved_nonrenal_clearance_l_h,
        rel=1.0e-8,
    )
    assert (
        final["urine_captopril_filtration_umol"]
        / final["urine_captopril_secretion_umol"]
    ) == pytest.approx(
        model.mechanisms.filtration_clearance_on_total_l_h
        / model.mechanisms.secretion_clearance_on_total_l_h,
        rel=1.0e-8,
    )
    assert np.allclose(
        result.trajectory["tracked_molecular_mass_mg"],
        result.trajectory["drug_equivalent_mass_mg"],
        rtol=0.0,
        atol=1.0e-9,
    )
    assert result.trajectory["research_only_not_clinically_validated"].eq(1.0).all()
    assert result.trajectory["observation_captopril_authorized"].eq(0.0).all()
    assert "observation_captopril_mg_l" not in result.trajectory
    assert np.allclose(
        result.trajectory["central_blood_equivalent_captopril_mg_l"],
        result.trajectory["plasma_parent_mg_l"],
        rtol=0.0,
        atol=1.0e-14,
    )
    assert np.allclose(
        result.trajectory["urine_captopril_umol"],
        result.trajectory["urine_captopril_total_umol"],
        rtol=0.0,
        atol=1.0e-12,
    )
    assert np.allclose(
        result.trajectory["urine_captopril_mg"],
        result.trajectory["urine_parent_mg"],
        rtol=0.0,
        atol=1.0e-12,
    )


def test_oral_parent_model_closes_mass_and_keeps_absorption_assumptions_visible() -> None:
    _model, result = _simulate(route="oral")
    assert result.summary["max_abs_mass_balance_error_mg"] < 1.0e-5
    assert result.states_umol.min() >= -1.0e-8
    assert result.trajectory["plasma_parent_mg_l"].max() > 0.0
    assert result.trajectory["feces_parent_mg"].iloc[-1] == pytest.approx(
        10.0 * 0.29,
        abs=1.0e-8,
    )


def test_filtration_secretion_and_unresolved_loss_are_separately_ablatable() -> None:
    _baseline_model, baseline = _simulate()
    _gfr_model, gfr_off = _simulate(factors={"glomerular_filtration": 0.0})
    _sec_model, secretion_off = _simulate(
        factors={"active_tubular_secretion": 0.0}
    )
    _nonrenal_model, nonrenal_off = _simulate(
        factors={"unresolved_nonrenal_loss": 0.0}
    )

    assert gfr_off.trajectory["urine_captopril_filtration_umol"].max() < 1.0e-12
    assert (
        secretion_off.trajectory["urine_captopril_secretion_umol"].max()
        < 1.0e-12
    )
    assert (
        nonrenal_off.trajectory[
            "unresolved_nonrenal_captopril_equivalent_umol"
        ].max()
        < 1.0e-12
    )
    baseline_auc = np.trapezoid(
        baseline.trajectory["plasma_parent_mg_l"], baseline.times_h
    )
    for ablated in (gfr_off, secretion_off, nonrenal_off):
        auc = np.trapezoid(
            ablated.trajectory["plasma_parent_mg_l"], ablated.times_h
        )
        assert auc > baseline_auc
        assert ablated.summary["max_abs_mass_balance_error_mg"] < 1.0e-5


def test_parent_dose_basis_and_mg_umol_conversion_are_explicit() -> None:
    drug, _mechanism, _payload, model = _components()
    amount_mg = 25.0
    amount_umol = amount_mg * 1000.0 / drug.molecular_weight_g_mol
    by_mass = model.apply_species_doses(
        model.initial_state(),
        (
            SpeciesDoseEvent(
                time_h=0.0,
                species_id=PARENT,
                amount=amount_mg,
                unit=AmountUnit.MG,
                route="oral",
                dose_basis_id="captopril_parent",
            ),
        ),
    )
    by_moles = model.apply_species_doses(
        model.initial_state(),
        (
            SpeciesDoseEvent(
                time_h=0.0,
                species_id=PARENT,
                amount=amount_umol,
                unit=AmountUnit.UMOL,
                route="oral",
                dose_basis_id="captopril_parent",
            ),
        ),
    )
    assert np.allclose(by_mass, by_moles, rtol=0.0, atol=1.0e-12)
    assert by_mass.sum() == pytest.approx(amount_umol)
    depot = by_mass[model.state_registry.index(captopril_state("oral_absorption_depot"))]
    feces = by_mass[model.state_registry.index(captopril_state("feces"))]
    assert depot == pytest.approx(amount_umol * 0.71)
    assert feces == pytest.approx(amount_umol * 0.29)
    with pytest.raises(ValueError, match="dose basis"):
        model.apply_species_doses(
            model.initial_state(),
            (
                SpeciesDoseEvent(
                    time_h=0.0,
                    species_id=PARENT,
                    amount=25.0,
                    unit=AmountUnit.MG,
                    route="oral",
                    dose_basis_id="unspecified_salt_equivalent",
                ),
            ),
        )
    invalid_route = SpeciesDoseEvent(
        time_h=0.0,
        species_id=PARENT,
        amount=25.0,
        unit=AmountUnit.MG,
        route="oral",
        dose_basis_id="captopril_parent",
    )
    object.__setattr__(invalid_route, "route", "intramuscular")
    with pytest.raises(ValueError, match="route must be oral or iv_bolus"):
        model.apply_species_doses(model.initial_state(), (invalid_route,))


def test_reaction_is_parent_moiety_only_not_a_named_disulfide_claim() -> None:
    drug, mechanism, _payload, _model = _components()
    network = build_captopril_reaction_network(
        drug.molecular_weight_g_mol,
        provenance=mechanism.provenance,
    )
    assert set(network.species.ids) == {PARENT, UNRESOLVED_NONRENAL}
    assert network.reactions[0].reaction_id == (
        "captopril_to_unresolved_nonrenal_equivalent"
    )
    assert max(
        abs(item.closure_error_mg_per_umol)
        for item in network.balance_diagnostics
    ) < 1.0e-12


def test_reversible_disulfide_exchange_and_invalid_parameters_are_rejected() -> None:
    drug, mechanism, _payload, _model = _components()
    with pytest.raises(ValueError, match="not admitted"):
        replace(mechanism, disulfide_exchange_enabled=True)
    with pytest.raises(ValueError, match="unknown or non-admitted"):
        CaptoprilRenalPBPKModel(
            drug,
            mechanism,
            mechanism_factors={"reversible_disulfide_exchange": 1.0},
        )
    with pytest.raises(ValueError, match=r"fu \* GFR"):
        replace(mechanism, renal_clearance_l_h=1.0)
    with pytest.raises(ValueError, match=">= renal_clearance"):
        replace(mechanism, systemic_total_clearance_l_h=20.0)
    with pytest.raises(ValueError, match="fraction_unbound_plasma differs from frozen"):
        CaptoprilRenalPBPKModel(
            replace(drug, fraction_unbound_plasma=0.70),
            mechanism,
        )
    wrong_analyte = ObservationSpec(
        observation_id="synthetic_wrong_analyte",
        analyte_id="total_captopril_equivalents",
        source_matrix="whole_blood",
        modeled_matrix="whole_blood",
        quantity="total",
        matrix_bridge="identity",
        fixed_matrix_ratio=None,
        bound_recovery_fraction=1.0,
        residual_model_id=None,
        provenance="synthetic fail-closed test fixture",
    )
    with pytest.raises(ValueError, match="unchanged parent"):
        CaptoprilRenalPBPKModel(
            drug,
            mechanism,
            observation_spec=wrong_analyte,
        )


@pytest.mark.parametrize(
    "factors",
    [
        {"unknown_factor": 0.0},
        {"glomerular_filtration": 0.5},
        {"glomerular_filtration": 1.0},
        {"glomerular_filtration": 0.0, "active_tubular_secretion": 0.0},
    ],
)
def test_only_one_binary_test_ablation_is_allowed(factors) -> None:
    drug, mechanism, _payload = load_captopril_candidate(CONFIG)
    with pytest.raises(ValueError, match="unknown|test-only"):
        CaptoprilRenalPBPKModel(drug, mechanism, mechanism_factors=factors)


@pytest.mark.parametrize(
    "factors",
    [
        {"glomerular_filtration": False},
        {"glomerular_filtration": "0"},
        {1: 0.0},
    ],
)
def test_ablation_contract_rejects_type_coercion(factors) -> None:
    drug, mechanism, _payload = load_captopril_candidate(CONFIG)
    with pytest.raises(TypeError, match="booleans, strings, and coerced"):
        CaptoprilRenalPBPKModel(drug, mechanism, mechanism_factors=factors)


def test_run_contract_hash_separates_baseline_from_structural_ablation() -> None:
    drug, mechanism, _payload = load_captopril_candidate(CONFIG)
    baseline = CaptoprilRenalPBPKModel(drug, mechanism)
    ablation = CaptoprilRenalPBPKModel(
        drug,
        mechanism,
        mechanism_factors={"active_tubular_secretion": 0.0},
    )
    assert baseline.mechanism_run_mode == "canonical_baseline"
    assert ablation.mechanism_run_mode == "structural_ablation"
    assert baseline.mechanism_run_contract_sha256 != (
        ablation.mechanism_run_contract_sha256
    )
    assert len(baseline.mechanism_run_contract_sha256) == 64


def test_explicit_unchanged_parent_whole_blood_observation_is_required() -> None:
    drug, mechanism, _payload = load_captopril_candidate(CONFIG)
    valid = ObservationSpec(
        observation_id="synthetic_whole_blood_unchanged_parent",
        analyte_id=OBSERVED_PARENT_ANALYTE_ID,
        source_matrix="whole_blood",
        modeled_matrix="whole_blood",
        quantity="total",
        matrix_bridge="identity",
        fixed_matrix_ratio=None,
        bound_recovery_fraction=0.0,
        residual_model_id="synthetic_only",
        provenance="synthetic observation-layer test",
    )
    model = CaptoprilRenalPBPKModel(drug, mechanism, observation_spec=valid)
    state = model.apply_doses(
        model.initial_state(),
        (DoseEvent(0.0, 10.0, "iv_bolus"),),
    )
    record = model.concentration_record(state, reference_patient(), drug)
    assert record["observation_identity_resolved"] == 1.0
    assert record["observation_captopril_mg_l"] == pytest.approx(
        record["central_blood_equivalent_captopril_mg_l"]
    )

    proxy = ObservationSpec(
        observation_id="synthetic_plasma_proxy",
        analyte_id=OBSERVED_PARENT_ANALYTE_ID,
        source_matrix="plasma",
        modeled_matrix="whole_blood",
        quantity="total",
        matrix_bridge="declared_proxy_no_conversion",
        fixed_matrix_ratio=None,
        bound_recovery_fraction=0.0,
        residual_model_id=None,
        provenance="synthetic negative control",
    )
    with pytest.raises(ValueError, match="proxy-without-conversion"):
        CaptoprilRenalPBPKModel(drug, mechanism, observation_spec=proxy)

    mechanistic_unbound = ObservationSpec(
        observation_id="synthetic_unbound",
        analyte_id=OBSERVED_PARENT_ANALYTE_ID,
        source_matrix="whole_blood",
        modeled_matrix="whole_blood",
        quantity="unbound",
        matrix_bridge="identity",
        fixed_matrix_ratio=None,
        bound_recovery_fraction=0.0,
        residual_model_id=None,
        provenance="synthetic negative control",
    )
    with pytest.raises(ValueError, match="free-versus-bound"):
        CaptoprilRenalPBPKModel(
            drug,
            mechanism,
            observation_spec=mechanistic_unbound,
        )


@pytest.mark.parametrize("route", ["iv_bolus", "oral"])
def test_raw_unclipped_solver_states_remain_nonnegative(route: str) -> None:
    drug, _mechanism, _payload, model = _components()
    initial = model.apply_species_doses(
        model.initial_state(),
        (
            SpeciesDoseEvent(
                time_h=0.0,
                species_id=PARENT,
                amount=10.0,
                unit=AmountUnit.MG,
                route=route,
                dose_basis_id="captopril_parent",
            ),
        ),
    )
    patient = reference_patient()
    solution = solve_ivp(
        lambda time_h, state: model.rhs(time_h, state, patient, drug),
        (0.0, 12.0),
        initial,
        method="LSODA",
        rtol=1.0e-9,
        atol=1.0e-11,
        max_step=0.01,
    )
    assert solution.success
    # This inspects scipy's raw state matrix before virtual_trial clips tiny
    # negative amounts at its historical mg-labelled compatibility boundary.
    assert float(solution.y.min()) >= -1.0e-10


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("absorption_rate_h", 9.9),
        ("fraction_absorbed", 0.99),
        ("kp_gut", 2.0),
        ("kp_liver", 7.0),
        ("kp_kidney", 9.0),
        ("kp_rest", 0.2),
        ("liver_vmax_mg_h", 1.0),
        ("gut_vmax_mg_h", 1.0),
        ("metabolite_renal_clearance_l_h", 1.0),
        ("metabolite_mass_yield", 1.0),
    ],
)
def test_constructor_rejects_every_active_or_unused_drug_mutation(
    field: str, value: float
) -> None:
    drug, mechanism, _payload = load_captopril_candidate(CONFIG)
    with pytest.raises(CaptoprilCandidateError, match="frozen|unused"):
        CaptoprilRenalPBPKModel(replace(drug, **{field: value}), mechanism)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("fraction_unbound", 0.70),
        ("reference_gfr_l_h", 6.0),
        ("renal_clearance_l_h", 26.0),
        ("systemic_total_clearance_l_h", 55.0),
    ],
)
def test_constructor_rejects_mechanism_parameter_mutation(
    field: str, value: float
) -> None:
    drug, mechanism, _payload = load_captopril_candidate(CONFIG)
    changed = replace(mechanism, **{field: value})
    with pytest.raises(CaptoprilCandidateError, match="frozen"):
        CaptoprilRenalPBPKModel(drug, changed)


def test_strict_config_identity_schema_derivations_and_hashes(tmp_path: Path) -> None:
    drug, mechanism, payload = load_captopril_candidate(CONFIG)
    assert len(payload["canonical_config_sha256"]) == 64
    assert len(payload["canonical_active_parameter_sha256"]) == 64
    model = CaptoprilRenalPBPKModel(
        drug,
        mechanism,
        candidate_config_sha256=payload["canonical_config_sha256"],
    )
    assert model.candidate_config_sha256 == payload["canonical_config_sha256"]
    assert model.active_parameter_sha256 == payload[
        "canonical_active_parameter_sha256"
    ]
    assert model.mechanism_run_contract["active_parameter_sha256"] == (
        model.active_parameter_sha256
    )

    source = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    mutations = []
    unknown_top = copy.deepcopy(source)
    unknown_top["unknown"] = True
    mutations.append(unknown_top)
    unknown_drug = copy.deepcopy(source)
    unknown_drug["drug"]["hidden_parameter"] = 1.0
    mutations.append(unknown_drug)
    wrong_identity = copy.deepcopy(source)
    wrong_identity["candidate_id"] = "another_candidate"
    mutations.append(wrong_identity)
    wrong_mechanism = copy.deepcopy(source)
    wrong_mechanism["mechanism"]["mechanism_id"] = "another_mechanism"
    mutations.append(wrong_mechanism)
    wrong_derivation = copy.deepcopy(source)
    wrong_derivation["mechanism"]["deterministic_derivations"][
        "secretion_clearance_on_total_l_h"
    ]["value"] = 21.0
    mutations.append(wrong_derivation)
    wrong_dose_contract = copy.deepcopy(source)
    wrong_dose_contract["dose_contract"]["oral_fraction_absorbed"] = 0.70
    mutations.append(wrong_dose_contract)
    for index, changed in enumerate(mutations):
        path = _write_config(tmp_path / str(index), changed)
        with pytest.raises(CaptoprilCandidateError):
            load_captopril_candidate(path)


def test_duplicate_yaml_key_is_rejected_before_candidate_construction(
    tmp_path: Path,
) -> None:
    original = CONFIG.read_text(encoding="utf-8")
    path = tmp_path / "duplicate.yaml"
    path.write_text(
        original.replace(
            "candidate_id: captopril_parent_renal_independent_evidence_v05",
            "candidate_id: first\n"
            "candidate_id: captopril_parent_renal_independent_evidence_v05",
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(CaptoprilCandidateError, match="duplicate YAML key"):
        load_captopril_candidate(path)


def test_generic_summary_is_explicitly_internal_and_cannot_be_scored() -> None:
    drug, mechanism, payload = load_captopril_candidate(CONFIG)
    unauthorized = CaptoprilRenalPBPKModel(
        drug,
        mechanism,
        candidate_config_sha256=payload["canonical_config_sha256"],
    )
    result = simulate_amount_patient(
        reference_patient(),
        drug,
        (DoseEvent(0.0, 10.0, "iv_bolus"),),
        model=unauthorized,
        duration_h=1.0,
        output_interval_h=0.1,
    )
    assert result.trajectory[
        "generic_pk_summary_authorized_for_clinical_scoring"
    ].eq(0.0).all()
    assert result.trajectory["generic_pk_summary_is_internal_state_only"].eq(1.0).all()
    with pytest.raises(PermissionError, match="ObservationSpec hash"):
        build_captopril_clinical_scoring_input(
            unauthorized,
            result.trajectory,
            expected_observation_spec_sha256="0" * 64,
        )
    renamed = result.trajectory.rename(
        columns={"plasma_parent_mg_l": "observation_captopril_mg_l"}
    )
    with pytest.raises(PermissionError, match="ObservationSpec hash"):
        build_captopril_clinical_scoring_input(
            unauthorized,
            renamed,
            expected_observation_spec_sha256="0" * 64,
        )


def test_only_exact_authorized_observation_column_builds_scoring_input() -> None:
    drug, mechanism, payload = load_captopril_candidate(CONFIG)
    spec = _valid_observation_spec()
    model = CaptoprilRenalPBPKModel(
        drug,
        mechanism,
        observation_spec=spec,
        candidate_config_sha256=payload["canonical_config_sha256"],
    )
    result = simulate_amount_patient(
        reference_patient(),
        drug,
        (DoseEvent(0.0, 10.0, "iv_bolus"),),
        model=model,
        duration_h=1.0,
        output_interval_h=0.1,
    )
    spec_hash = captopril_observation_spec_sha256(spec)
    scoring = build_captopril_clinical_scoring_input(
        model,
        result.trajectory,
        expected_observation_spec_sha256=spec_hash,
    )
    assert scoring.observation_spec_sha256 == spec_hash
    assert scoring.source_column == "observation_captopril_mg_l"
    assert scoring.generic_summary_used is False
    assert scoring.times_h.flags.writeable is False
    assert np.allclose(
        scoring.concentrations_mg_l,
        result.trajectory["observation_captopril_mg_l"],
    )
    with pytest.raises(PermissionError, match="absent or changed"):
        build_captopril_clinical_scoring_input(
            model,
            result.trajectory,
            expected_observation_spec_sha256="0" * 64,
        )
    missing = result.trajectory.drop(columns=["observation_captopril_mg_l"])
    with pytest.raises(PermissionError, match="contract is incomplete"):
        build_captopril_clinical_scoring_input(
            model,
            missing,
            expected_observation_spec_sha256=spec_hash,
        )
    unauthorized_flag = result.trajectory.copy()
    unauthorized_flag["observation_captopril_authorized"] = 0.0
    with pytest.raises(PermissionError, match="authorization"):
        build_captopril_clinical_scoring_input(
            model,
            unauthorized_flag,
            expected_observation_spec_sha256=spec_hash,
        )


@pytest.mark.parametrize("route", ["iv_bolus", "oral"])
@pytest.mark.parametrize("dose_mg", [0.01, 10.0, 1000.0])
@pytest.mark.parametrize("renal_function", [0.0, 1.0])
def test_captopril_specific_raw_positivity_stress_report(
    route: str, dose_mg: float, renal_function: float
) -> None:
    drug, mechanism, payload = load_captopril_candidate(CONFIG)
    model = CaptoprilRenalPBPKModel(
        drug,
        mechanism,
        candidate_config_sha256=payload["canonical_config_sha256"],
    )
    report = audit_captopril_raw_positivity(
        model,
        reference_patient(renal_function_fraction=renal_function),
        route=route,
        dose_mg=dose_mg,
        duration_h=12.0,
        max_step_h=0.1,
    )
    assert report.passed is True
    assert report.solver_success is True
    assert report.raw_min_state_umol >= -report.tolerance_umol
    assert report.boundary_min_derivative_umol_h >= -report.tolerance_umol
    assert report.clip_parent_equivalent_mass_mg <= (
        report.clip_mass_tolerance_mg
    )
    assert report.model_run_contract_sha256 == model.mechanism_run_contract_sha256
