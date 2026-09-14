"""Independent-evidence captopril renal candidate for v0.5 development.

This model is a deliberately small, micromole-native baseline.  It represents
oral input, perfusion-limited systemic distribution, and unchanged-parent
renal elimination split into glomerular filtration and net tubular secretion.
The remainder of independently reported total clearance is retained as an
unresolved parent-moiety sink.  It is *not* labelled as a measured captopril
disulfide concentration.

Captopril forms mixed and dimeric disulfides in humans, but the evidence set
used for this candidate does not provide transferable whole-human forward and
reverse kinetic constants.  Reversible disulfide exchange is therefore
explicitly disabled rather than fitted or guessed.

This is research software.  It is not validated for diagnosis, dosing,
efficacy, toxicity, or replacement of a human study.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml
from scipy.integrate import solve_ivp

from ..chemistry.accounting import (
    DualAmountLedger,
    OpenReaction,
    OpenReactionNetwork,
    SpeciesMoiety,
)
from ..chemistry.reactions import StoichiometricTerm
from ..core.amount_state import (
    AmountStateRegistry,
    AmountStateSpec,
    FluxKey,
    SpeciesDoseEvent,
)
from ..core.amount_system import (
    AmountModelContext,
    AmountModuleResult,
    AmountSystemAssembler,
)
from ..core.species import AmountUnit, ChemicalSpecies, SpeciesRegistry
from ..observation import ObservationSpec, predict_observation
from ..pbpk import DoseEvent, DrugParameters
from .multispecies import MultiSpeciesPBPKModel


PARENT = "captopril"
UNRESOLVED_NONRENAL = "unresolved_nonrenal_captopril_equivalent"
ORAL_INPUT = FluxKey("oral_input_to_liver", PARENT)
REFERENCE_BODY_WEIGHT_KG = 70.0
REFERENCE_CAPTOPRIL_MW_G_MOL = 217.29
OBSERVED_PARENT_ANALYTE_ID = "captopril_unchanged_parent_operational_total"
CANDIDATE_SCHEMA_VERSION = "our_star.captopril_mechanism.v0.5"
CANDIDATE_ID = "captopril_parent_renal_independent_evidence_v05"
MECHANISM_ID = "captopril_filtration_secretion_parent_baseline_v05"
CANDIDATE_STATUS = "development_candidate_not_external_admitted"
MECHANISM_STATUS = "active_fixed_candidate_not_admitted"
CLINICAL_STATUS = "research_only_not_for_dosing_or_trial_replacement"
CANONICAL_ACTIVE_DRUG_VALUES = {
    "name": "captopril_parent_independent_evidence_v05",
    "smiles": "C[C@H](CS)C(=O)N1CCC[C@H]1C(=O)O",
    "molecular_weight_g_mol": REFERENCE_CAPTOPRIL_MW_G_MOL,
    "fraction_unbound_plasma": 0.725,
    "absorption_rate_h": 2.18,
    "fraction_absorbed": 0.71,
    "kp_gut": 1.0,
    "kp_liver": 1.0,
    "kp_kidney": 1.0,
    "kp_rest": 1.0,
}
CANONICAL_UNUSED_DRUG_VALUES = {
    "liver_vmax_mg_h": 0.0,
    "liver_km_mg_l": 1.0,
    "liver_zone_activity": (1.0, 1.0, 1.0),
    "gut_vmax_mg_h": 0.0,
    "gut_km_mg_l": 1.0,
    "renal_clearance_l_h": 0.0,
    "metabolite_renal_clearance_l_h": 0.0,
    "metabolite_mass_yield": 0.0,
}
CANONICAL_MECHANISM_VALUES = {
    "fraction_unbound": 0.725,
    "reference_gfr_l_h": 7.5,
    "renal_clearance_l_h": 27.16,
    "systemic_total_clearance_l_h": 56.0,
    "disulfide_exchange_enabled": False,
}
CANONICAL_DETERMINISTIC_DERIVATIONS = {
    "filtration_clearance_on_total_l_h": 5.4375,
    "secretion_clearance_on_total_l_h": 21.7225,
    "secretion_clearance_on_unbound_l_h": 29.96206896551724,
    "unresolved_nonrenal_clearance_l_h": 28.84,
}
ABLATION_FACTORS = frozenset(
    {
        "glomerular_filtration",
        "active_tubular_secretion",
        "unresolved_nonrenal_loss",
    }
)

_SHA_RE = re.compile(r"[0-9a-f]{64}")


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


class CaptoprilCandidateError(ValueError):
    """Raised when a candidate differs from the frozen zero-tuning contract."""


class _UniqueSafeLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueSafeLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[object, object]:
    result: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise CaptoprilCandidateError(f"duplicate YAML key: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise CaptoprilCandidateError(f"{label} must be a string-keyed mapping")
    return value  # type: ignore[return-value]


def _exact_keys(
    value: Mapping[str, Any], expected: frozenset[str], label: str
) -> None:
    actual = set(value)
    if actual != expected:
        raise CaptoprilCandidateError(
            f"{label} keys differ; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _exact_float(value: object, expected: float, label: str) -> float:
    if type(value) not in {int, float}:
        raise CaptoprilCandidateError(f"{label} must be an exact numeric value")
    numeric = float(value)
    if not np.isfinite(numeric) or numeric != expected:
        raise CaptoprilCandidateError(
            f"{label} differs from frozen value {expected!r}: {value!r}"
        )
    return numeric


def _derived_float(value: object, expected: float, label: str) -> float:
    """Compare deterministic arithmetic while allowing one-ULP roundoff."""

    if type(value) not in {int, float}:
        raise CaptoprilCandidateError(f"{label} must be an exact numeric value")
    numeric = float(value)
    if not np.isfinite(numeric) or not np.isclose(
        numeric, expected, rtol=0.0, atol=1.0e-14
    ):
        raise CaptoprilCandidateError(
            f"{label} differs from deterministic value {expected!r}: {value!r}"
        )
    return numeric


def captopril_state(compartment_id: str, species_id: str = PARENT) -> str:
    """Return the unambiguous amount-state name used by this model."""

    return f"{compartment_id}::{species_id}"


def captopril_observation_spec_record(spec: ObservationSpec) -> dict[str, object]:
    """Return the exact immutable scoring fields of an observation bridge."""

    if not isinstance(spec, ObservationSpec):
        raise TypeError("spec must be an ObservationSpec")
    return {
        "schema_version": "our_star.ObservationSpec.v1",
        **asdict(spec),
    }


def captopril_observation_spec_sha256(spec: ObservationSpec) -> str:
    """Hash all observation identity, matrix, quantity, and residual fields."""

    return _canonical_sha256(captopril_observation_spec_record(spec))


def _canonical_active_parameter_record(
    drug: DrugParameters,
    mechanisms: "CaptoprilMechanismParameters",
) -> dict[str, object]:
    return {
        "schema_version": "our_star.captopril_active_parameters.v1",
        "candidate_id": CANDIDATE_ID,
        "mechanism_id": MECHANISM_ID,
        "drug": {
            **{
                key: getattr(drug, key)
                for key in CANONICAL_ACTIVE_DRUG_VALUES
            },
            **{
                key: list(getattr(drug, key))
                if key == "liver_zone_activity"
                else getattr(drug, key)
                for key in CANONICAL_UNUSED_DRUG_VALUES
            },
        },
        "mechanism": {
            **{
                key: getattr(mechanisms, key)
                for key in CANONICAL_MECHANISM_VALUES
            },
            "deterministic_derivations": {
                "filtration_clearance_on_total_l_h": (
                    mechanisms.filtration_clearance_on_total_l_h
                ),
                "secretion_clearance_on_total_l_h": (
                    mechanisms.secretion_clearance_on_total_l_h
                ),
                "secretion_clearance_on_unbound_l_h": (
                    mechanisms.secretion_clearance_on_unbound_l_h
                ),
                "unresolved_nonrenal_clearance_l_h": (
                    mechanisms.unresolved_nonrenal_clearance_l_h
                ),
            },
        },
    }


def _validate_canonical_model_parameters(
    drug: DrugParameters,
    mechanisms: "CaptoprilMechanismParameters",
) -> dict[str, object]:
    for name, expected in CANONICAL_ACTIVE_DRUG_VALUES.items():
        observed = getattr(drug, name)
        if isinstance(expected, str):
            if observed != expected:
                raise CaptoprilCandidateError(
                    f"drug.{name} differs from the frozen captopril identity"
                )
        else:
            _exact_float(observed, expected, f"drug.{name}")
    for name, expected in CANONICAL_UNUSED_DRUG_VALUES.items():
        observed = getattr(drug, name)
        if isinstance(expected, tuple):
            if tuple(observed) != expected:
                raise CaptoprilCandidateError(
                    f"unused drug.{name} must equal its frozen neutral default"
                )
        else:
            _exact_float(observed, expected, f"unused drug.{name}")
    for name, expected in CANONICAL_MECHANISM_VALUES.items():
        observed = getattr(mechanisms, name)
        if isinstance(expected, bool):
            if type(observed) is not bool or observed is not expected:
                raise CaptoprilCandidateError(
                    f"mechanism.{name} differs from its frozen value"
                )
        else:
            _exact_float(observed, expected, f"mechanism.{name}")
    derivations = {
        "filtration_clearance_on_total_l_h": (
            mechanisms.filtration_clearance_on_total_l_h
        ),
        "secretion_clearance_on_total_l_h": (
            mechanisms.secretion_clearance_on_total_l_h
        ),
        "secretion_clearance_on_unbound_l_h": (
            mechanisms.secretion_clearance_on_unbound_l_h
        ),
        "unresolved_nonrenal_clearance_l_h": (
            mechanisms.unresolved_nonrenal_clearance_l_h
        ),
    }
    for name, expected in CANONICAL_DETERMINISTIC_DERIVATIONS.items():
        _derived_float(derivations[name], expected, f"derived mechanism.{name}")
    return _canonical_active_parameter_record(drug, mechanisms)


@dataclass(frozen=True)
class CaptoprilMechanismParameters:
    """Fixed independent-evidence renal-clearance contract.

    ``renal_clearance_l_h`` and ``systemic_total_clearance_l_h`` act on total
    systemic blood-equivalent parent concentration for the 70 kg reference
    subject.  The directly observed renal clearance is not treated as an
    intrinsic kidney clearance and then subjected to a second flow limitation.
    Filtration and secretion are not separate fitted parameters: filtration is
    ``fu * GFR`` and secretion is the nonnegative residual required to recover
    the independently measured renal clearance.
    """

    fraction_unbound: float
    reference_gfr_l_h: float
    renal_clearance_l_h: float
    systemic_total_clearance_l_h: float
    disulfide_exchange_enabled: bool
    provenance: str

    def __post_init__(self) -> None:
        fraction = float(self.fraction_unbound)
        if not np.isfinite(fraction) or not 0.0 < fraction <= 1.0:
            raise ValueError("fraction_unbound must be finite and in (0, 1]")
        object.__setattr__(self, "fraction_unbound", fraction)
        for name in (
            "reference_gfr_l_h",
            "renal_clearance_l_h",
            "systemic_total_clearance_l_h",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0")
            object.__setattr__(self, name, value)
        if self.renal_clearance_l_h <= self.filtration_clearance_on_total_l_h:
            raise ValueError(
                "renal_clearance_l_h must exceed fu * GFR to support the "
                "independently observed net-secretion branch"
            )
        if self.systemic_total_clearance_l_h < self.renal_clearance_l_h:
            raise ValueError(
                "systemic_total_clearance_l_h must be >= renal_clearance_l_h"
            )
        if bool(self.disulfide_exchange_enabled):
            raise ValueError(
                "reversible captopril-disulfide exchange is not admitted: "
                "independent whole-human forward/reverse priors are unavailable"
            )
        object.__setattr__(self, "disulfide_exchange_enabled", False)
        if not isinstance(self.provenance, str) or not self.provenance.strip():
            raise ValueError("captopril mechanism provenance is required")

    @property
    def filtration_clearance_on_total_l_h(self) -> float:
        return self.fraction_unbound * self.reference_gfr_l_h

    @property
    def secretion_clearance_on_total_l_h(self) -> float:
        return self.renal_clearance_l_h - self.filtration_clearance_on_total_l_h

    @property
    def secretion_clearance_on_unbound_l_h(self) -> float:
        return self.secretion_clearance_on_total_l_h / self.fraction_unbound

    @property
    def unresolved_nonrenal_clearance_l_h(self) -> float:
        return self.systemic_total_clearance_l_h - self.renal_clearance_l_h


def build_captopril_reaction_network(
    molecular_weight_g_mol: float,
    *,
    provenance: str,
) -> OpenReactionNetwork:
    """Build a 1:1 parent-moiety bookkeeping reaction for unresolved loss."""

    molecular_weight = float(molecular_weight_g_mol)
    if not np.isfinite(molecular_weight) or molecular_weight <= 0.0:
        raise ValueError("captopril molecular weight must be finite and > 0")
    if not provenance.strip():
        raise ValueError("reaction provenance is required")
    species = (
        ChemicalSpecies(PARENT, "captopril", molecular_weight),
        ChemicalSpecies(
            UNRESOLVED_NONRENAL,
            "unresolved nonrenal captopril parent-moiety equivalent",
            molecular_weight,
        ),
    )
    return OpenReactionNetwork(
        species,
        (
            OpenReaction(
                reaction_id="captopril_to_unresolved_nonrenal_equivalent",
                terms=(
                    StoichiometricTerm(PARENT, -1.0),
                    StoichiometricTerm(UNRESOLVED_NONRENAL, 1.0),
                ),
                description=(
                    "One-to-one parent-moiety accounting for the residual of "
                    "whole-body minus renal clearance. The product is not a "
                    "named metabolite or an observation. "
                    + provenance
                ),
            ),
        ),
    )


def _amount(
    states: AmountStateRegistry,
    vector: np.ndarray,
    compartment: str,
    species_id: str = PARENT,
) -> float:
    return max(
        float(vector[states.index(captopril_state(compartment, species_id))]),
        0.0,
    )


def _build_states(
    network: OpenReactionNetwork,
) -> tuple[AmountStateRegistry, DualAmountLedger]:
    species = SpeciesRegistry(tuple(network.species))
    physical_compartments = (
        "oral_absorption_depot",
        "central",
        "liver_zone1",
        "liver_zone2",
        "liver_zone3",
        "kidney",
        "rest",
        "urine_filtration",
        "urine_secretion",
        "feces",
    )
    sink_compartments = {"urine_filtration", "urine_secretion", "feces"}
    records = [
        AmountStateSpec(
            captopril_state(compartment),
            PARENT,
            compartment,
            sink=compartment in sink_compartments,
        )
        for compartment in physical_compartments
    ]
    records.append(
        AmountStateSpec(
            captopril_state("unresolved_nonrenal_sink", UNRESOLVED_NONRENAL),
            UNRESOLVED_NONRENAL,
            "unresolved_nonrenal_sink",
            sink=True,
        )
    )
    states = AmountStateRegistry(records, species)
    ledger = DualAmountLedger(
        states,
        (
            SpeciesMoiety(PARENT, {PARENT: 1.0}),
            SpeciesMoiety(UNRESOLVED_NONRENAL, {PARENT: 1.0}),
        ),
        conserved_moiety_id=PARENT,
        reference_species_id=PARENT,
        external_participants=network.external_participants,
        external_import_states={},
        external_export_states={},
    )
    return states, ledger


class _CaptoprilOralInputModule:
    name = "captopril_oral_input"
    required_states = (captopril_state("oral_absorption_depot"),)
    required_inputs: tuple[FluxKey, ...] = ()

    def evaluate(
        self,
        _time_h: float,
        amounts_umol: np.ndarray,
        context: AmountModelContext,
        states: AmountStateRegistry,
        _inputs_umol_h: Mapping[FluxKey, float],
    ) -> AmountModuleResult:
        depot = _amount(states, amounts_umol, "oral_absorption_depot")
        absorption = context.drug.absorption_rate_h * depot
        return AmountModuleResult(
            derivatives_umol_h={
                captopril_state("oral_absorption_depot"): -absorption
            },
            outputs_umol_h={ORAL_INPUT: float(absorption)},
        )


class _CaptoprilLiverDistributionModule:
    name = "captopril_three_zone_liver_distribution"
    required_states = tuple(
        captopril_state(compartment)
        for compartment in (
            "central",
            "liver_zone1",
            "liver_zone2",
            "liver_zone3",
        )
    )
    required_inputs = (ORAL_INPUT,)

    def evaluate(
        self,
        _time_h: float,
        amounts_umol: np.ndarray,
        context: AmountModelContext,
        states: AmountStateRegistry,
        inputs_umol_h: Mapping[FluxKey, float],
    ) -> AmountModuleResult:
        patient = context.patient
        drug = context.drug
        central = _amount(states, amounts_umol, "central") / patient.central_volume_l
        zone_volume_l = patient.liver_volume_l / 3.0
        zones = tuple(
            _amount(states, amounts_umol, f"liver_zone{index}")
            / zone_volume_l
            / drug.kp_liver
            for index in (1, 2, 3)
        )
        flow = patient.liver_flow_l_h
        return AmountModuleResult(
            derivatives_umol_h={
                captopril_state("central"): flow * (zones[2] - central),
                captopril_state("liver_zone1"): (
                    inputs_umol_h[ORAL_INPUT]
                    + flow * central
                    - flow * zones[0]
                ),
                captopril_state("liver_zone2"): flow * (zones[0] - zones[1]),
                captopril_state("liver_zone3"): flow * (zones[1] - zones[2]),
            }
        )


class _CaptoprilKidneyModule:
    name = "captopril_renal_filtration_and_secretion"
    required_states = tuple(
        captopril_state(compartment)
        for compartment in (
            "central",
            "kidney",
            "urine_filtration",
            "urine_secretion",
        )
    )
    required_inputs: tuple[FluxKey, ...] = ()

    def __init__(self, parameters: CaptoprilMechanismParameters) -> None:
        self.parameters = parameters

    def evaluate(
        self,
        _time_h: float,
        amounts_umol: np.ndarray,
        context: AmountModelContext,
        states: AmountStateRegistry,
        _inputs_umol_h: Mapping[FluxKey, float],
    ) -> AmountModuleResult:
        patient = context.patient
        drug = context.drug
        central = _amount(states, amounts_umol, "central") / patient.central_volume_l
        kidney = (
            _amount(states, amounts_umol, "kidney")
            / patient.kidney_volume_l
            / drug.kp_kidney
        )
        exchange = patient.renal_flow_l_h * (central - kidney)
        renal_scale = (
            patient.renal_function_fraction
            * (patient.body_weight_kg / REFERENCE_BODY_WEIGHT_KG) ** 0.75
        )
        # These are observed systemic renal clearances.  Applying them to a
        # kidney-tissue concentration would reinterpret them as intrinsic
        # clearances and impose a second perfusion limitation.
        filtration = (
            self.parameters.filtration_clearance_on_total_l_h
            * renal_scale
            * context.mechanism_factor("glomerular_filtration")
            * central
        )
        secretion = (
            self.parameters.secretion_clearance_on_unbound_l_h
            * self.parameters.fraction_unbound
            * renal_scale
            * context.mechanism_factor("active_tubular_secretion")
            * central
        )
        return AmountModuleResult(
            derivatives_umol_h={
                captopril_state("central"): -exchange - filtration - secretion,
                captopril_state("kidney"): exchange,
                captopril_state("urine_filtration"): filtration,
                captopril_state("urine_secretion"): secretion,
            }
        )


class _CaptoprilRestModule:
    name = "captopril_rest_distribution"
    required_states = tuple(
        captopril_state(compartment) for compartment in ("central", "rest")
    )
    required_inputs: tuple[FluxKey, ...] = ()

    def evaluate(
        self,
        _time_h: float,
        amounts_umol: np.ndarray,
        context: AmountModelContext,
        states: AmountStateRegistry,
        _inputs_umol_h: Mapping[FluxKey, float],
    ) -> AmountModuleResult:
        patient = context.patient
        central = _amount(states, amounts_umol, "central") / patient.central_volume_l
        rest = (
            _amount(states, amounts_umol, "rest")
            / patient.rest_volume_l
            / context.drug.kp_rest
        )
        exchange = patient.rest_flow_l_h * (central - rest)
        return AmountModuleResult(
            derivatives_umol_h={
                captopril_state("central"): -exchange,
                captopril_state("rest"): exchange,
            }
        )


class _CaptoprilUnresolvedNonrenalModule:
    name = "captopril_unresolved_nonrenal_parent_moiety_loss"
    required_states = (
        captopril_state("central"),
        captopril_state("unresolved_nonrenal_sink", UNRESOLVED_NONRENAL),
    )
    required_inputs: tuple[FluxKey, ...] = ()

    def __init__(
        self,
        parameters: CaptoprilMechanismParameters,
        network: OpenReactionNetwork,
    ) -> None:
        self.parameters = parameters
        self.network = network

    def evaluate(
        self,
        _time_h: float,
        amounts_umol: np.ndarray,
        context: AmountModelContext,
        states: AmountStateRegistry,
        _inputs_umol_h: Mapping[FluxKey, float],
    ) -> AmountModuleResult:
        patient = context.patient
        central = _amount(states, amounts_umol, "central") / patient.central_volume_l
        scale = (patient.body_weight_kg / REFERENCE_BODY_WEIGHT_KG) ** 0.75
        rate = (
            self.parameters.unresolved_nonrenal_clearance_l_h
            * scale
            * context.mechanism_factor("unresolved_nonrenal_loss")
            * central
        )
        diagnostic = self.network.mass_rate_diagnostic(
            {"captopril_to_unresolved_nonrenal_equivalent": rate}
        )
        if abs(diagnostic.closure_error_mg_h) > 1.0e-9:
            raise FloatingPointError("captopril unresolved-loss reaction did not close")
        return AmountModuleResult(
            derivatives_umol_h={
                captopril_state("central"): -rate,
                captopril_state(
                    "unresolved_nonrenal_sink", UNRESOLVED_NONRENAL
                ): rate,
            }
        )


class CaptoprilRenalPBPKModel(MultiSpeciesPBPKModel):
    """Zero-tuning parent renal baseline with explicit elimination sinks."""

    def __init__(
        self,
        drug: DrugParameters,
        mechanisms: CaptoprilMechanismParameters,
        *,
        observation_spec: ObservationSpec | None = None,
        mechanism_factors: Mapping[str, float] | None = None,
        candidate_config_sha256: str | None = None,
    ) -> None:
        active_parameters = _validate_canonical_model_parameters(drug, mechanisms)
        if candidate_config_sha256 is not None and (
            not isinstance(candidate_config_sha256, str)
            or _SHA_RE.fullmatch(candidate_config_sha256) is None
        ):
            raise ValueError("candidate_config_sha256 must be a lowercase SHA-256")
        if not np.isclose(
            drug.fraction_unbound_plasma,
            mechanisms.fraction_unbound,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError(
                "drug and captopril mechanism fraction_unbound values must match"
            )
        if drug.renal_clearance_l_h != 0.0:
            raise ValueError(
                "drug.renal_clearance_l_h must be zero; the explicit filtration/"
                "secretion mechanism owns renal elimination"
            )
        raw_factors = dict(mechanism_factors or {})
        if any(
            not isinstance(key, str)
            or type(value) not in {int, float}
            or isinstance(value, bool)
            for key, value in raw_factors.items()
        ):
            raise TypeError(
                "captopril mechanism factors require string names and exact numeric "
                "values; booleans, strings, and coerced values are forbidden"
            )
        factors = {key: float(value) for key, value in raw_factors.items()}
        unknown_factors = sorted(set(factors) - ABLATION_FACTORS)
        if unknown_factors:
            raise ValueError(
                f"unknown or non-admitted captopril mechanism factors: {unknown_factors}"
            )
        if factors and (
            len(factors) != 1 or next(iter(factors.values())) != 0.0
        ):
            raise ValueError(
                "captopril mechanism factors are test-only: provide no factors "
                "for the canonical baseline or exactly one named 0.0 ablation"
            )
        network = build_captopril_reaction_network(
            drug.molecular_weight_g_mol,
            provenance=mechanisms.provenance,
        )
        states, ledger = _build_states(network)
        assembler = AmountSystemAssembler(
            states,
            (
                _CaptoprilOralInputModule(),
                _CaptoprilLiverDistributionModule(),
                _CaptoprilKidneyModule(mechanisms),
                _CaptoprilRestModule(),
                _CaptoprilUnresolvedNonrenalModule(mechanisms, network),
            ),
        )
        self.mechanisms = mechanisms
        if observation_spec is not None:
            if not isinstance(observation_spec, ObservationSpec):
                raise TypeError("observation_spec must be an ObservationSpec or None")
            if observation_spec.analyte_id != OBSERVED_PARENT_ANALYTE_ID:
                raise ValueError(
                    "captopril observation must identify unchanged parent; generic "
                    "captopril or total/disulfide-equivalent assays are unsupported"
                )
            if observation_spec.modeled_matrix != "whole_blood":
                raise ValueError(
                    "the model state is whole-blood-equivalent; plasma or serum "
                    "requires an explicit fixed-ratio bridge"
                )
            if observation_spec.matrix_bridge == "declared_proxy_no_conversion":
                raise ValueError(
                    "proxy-without-conversion cannot authorize captopril scoring"
                )
            if (
                observation_spec.quantity != "total"
                or observation_spec.bound_recovery_fraction != 0.0
            ):
                raise ValueError(
                    "the v0.5 captopril model supports only an operational total "
                    "unchanged-parent bridge with no explicit bound-state recovery; "
                    "it cannot make a mechanistic free-versus-bound assay claim"
                )
        self.observation_spec = observation_spec
        self.observation_spec_sha256 = (
            None
            if observation_spec is None
            else captopril_observation_spec_sha256(observation_spec)
        )
        self.active_parameter_record = active_parameters
        self.active_parameter_sha256 = _canonical_sha256(active_parameters)
        self.candidate_config_sha256 = candidate_config_sha256
        self.mechanism_run_mode = (
            "canonical_baseline" if not factors else "structural_ablation"
        )
        self.mechanism_run_contract = {
            "schema_version": "our_star.captopril_mechanism_run_contract.v1",
            "mode": self.mechanism_run_mode,
            "mechanism_factors": dict(sorted(factors.items())),
            "active_parameter_sha256": self.active_parameter_sha256,
            "candidate_config_sha256": self.candidate_config_sha256,
            "observation_spec_sha256": self.observation_spec_sha256,
        }
        self.mechanism_run_contract_sha256 = hashlib.sha256(
            json.dumps(
                self.mechanism_run_contract,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        super().__init__(
            model_id="our_star.captopril_parent_renal.v0.5.dev",
            drug=drug,
            formulation=None,
            states=states,
            assembler=assembler,
            reaction_network=network,
            ledger=ledger,
            parent_species_id=PARENT,
            dose_targets={
                ("oral", PARENT): captopril_state("oral_absorption_depot"),
                ("iv_bolus", PARENT): captopril_state("central"),
            },
            central_states={PARENT: captopril_state("central")},
            # The superclass accepts one urine state per species.  The
            # concentration record below replaces this placeholder with the
            # explicit filtration + secretion total.
            urine_states={PARENT: captopril_state("urine_filtration")},
            feces_parent_state=captopril_state("feces"),
            parent_liver_states=tuple(
                captopril_state(f"liver_zone{index}") for index in (1, 2, 3)
            ),
            mechanism_factors=factors,
        )

    def apply_species_doses(
        self,
        state: np.ndarray,
        events: Sequence[SpeciesDoseEvent],
    ) -> np.ndarray:
        values = self.state_registry.validate_vector(state).copy()
        allowed_bases = {"captopril_parent", "legacy_active_parent_mg"}
        for event in events:
            if event.species_id != PARENT:
                raise ValueError("captopril model accepts parent dose events only")
            if event.route not in {"oral", "iv_bolus"}:
                raise ValueError("captopril dose route must be oral or iv_bolus")
            if event.dose_basis_id not in allowed_bases:
                raise ValueError(
                    "dose basis must explicitly be captopril_parent; salts or "
                    "equivalent doses require a separate conversion contract"
                )
            amount_umol = float(
                self.state_registry.species[PARENT].convert_amount(
                    event.amount,
                    event.unit,
                    AmountUnit.UMOL,
                )
            )
            if event.route == "iv_bolus":
                values[self.state_registry.index(captopril_state("central"))] += (
                    amount_umol
                )
                continue
            # Oral only beyond this point; unsupported routes failed closed.
            absorption_eligible = amount_umol * self.drug.fraction_absorbed
            values[
                self.state_registry.index(captopril_state("oral_absorption_depot"))
            ] += absorption_eligible
            values[self.state_registry.index(captopril_state("feces"))] += (
                amount_umol - absorption_eligible
            )
        return values

    def apply_doses(
        self,
        state: np.ndarray,
        events: Sequence[DoseEvent],
    ) -> np.ndarray:
        return self.apply_species_doses(
            state,
            tuple(
                SpeciesDoseEvent(
                    time_h=event.time_h,
                    species_id=PARENT,
                    amount=event.amount_mg,
                    unit=AmountUnit.MG,
                    route=event.route,
                    dose_basis_id="legacy_active_parent_mg",
                )
                for event in events
            ),
        )

    def concentration_record(self, state, patient, drug) -> dict[str, Any]:
        record = super().concentration_record(state, patient, drug)
        values = self.state_registry.validate_vector(state)
        filtration = _amount(self.state_registry, values, "urine_filtration")
        secretion = _amount(self.state_registry, values, "urine_secretion")
        urine_total = filtration + secretion
        unresolved = _amount(
            self.state_registry,
            values,
            "unresolved_nonrenal_sink",
            UNRESOLVED_NONRENAL,
        )
        species = self.state_registry.species[PARENT]
        central_umol = _amount(self.state_registry, values, "central")
        record.update(
            {
                # The shared simulator requires plasma_parent_mg_l for generic
                # summaries.  For this candidate it is only a central
                # blood-equivalent state readout, not an assay prediction.
                "central_blood_equivalent_captopril_umol_l": (
                    central_umol / patient.central_volume_l
                ),
                "central_blood_equivalent_captopril_mg_l": record[
                    "plasma_parent_mg_l"
                ],
                "legacy_plasma_parent_alias_is_clinical_observation": 0.0,
                "generic_pk_summary_authorized_for_clinical_scoring": 0.0,
                "generic_pk_summary_is_internal_state_only": 1.0,
                "observation_identity_resolved": float(
                    self.observation_spec is not None
                ),
                "urine_captopril_filtration_umol": filtration,
                "urine_captopril_secretion_umol": secretion,
                "urine_captopril_total_umol": urine_total,
                "urine_captopril_umol": urine_total,
                "urine_captopril_mg": float(
                    species.convert_amount(
                        urine_total, AmountUnit.UMOL, AmountUnit.MG
                    )
                ),
                "urine_parent_mg": float(
                    species.convert_amount(urine_total, AmountUnit.UMOL, AmountUnit.MG)
                ),
                "unresolved_nonrenal_captopril_equivalent_umol": unresolved,
                "unresolved_nonrenal_captopril_equivalent_mg": float(
                    species.convert_amount(unresolved, AmountUnit.UMOL, AmountUnit.MG)
                ),
                "other_products_mg": float(
                    species.convert_amount(unresolved, AmountUnit.UMOL, AmountUnit.MG)
                ),
                "reversible_disulfide_exchange_enabled": 0.0,
                "unresolved_sink_is_named_disulfide_observation": 0.0,
                "research_only_not_clinically_validated": 1.0,
            }
        )
        if self.observation_spec is None:
            record["observation_captopril_authorized"] = 0.0
        else:
            observation = predict_observation(
                self.observation_spec,
                mobile_total_amount_umol=central_umol,
                bound_amount_umol=0.0,
                modeled_volume_l=patient.central_volume_l,
                molecular_weight_g_mol=species.molecular_weight_g_mol,
                fraction_unbound=self.mechanisms.fraction_unbound,
            )
            record.update(
                {
                    "observation_captopril_authorized": 1.0,
                    "observation_spec_sha256": self.observation_spec_sha256,
                    "model_run_contract_sha256": (
                        self.mechanism_run_contract_sha256
                    ),
                    "observation_captopril_umol_l": observation.concentration_umol_l,
                    "observation_captopril_mg_l": observation.concentration_mg_l,
                    "observation_uses_proxy_matrix": float(
                        observation.proxy_without_conversion
                    ),
                }
            )
        return record


@dataclass(frozen=True)
class CaptoprilClinicalScoringInput:
    """One authorized observation vector; generic simulator summaries excluded."""

    times_h: np.ndarray
    concentrations_mg_l: np.ndarray
    observation_spec_sha256: str
    model_run_contract_sha256: str
    source_column: str = "observation_captopril_mg_l"
    generic_summary_used: bool = False

    def __post_init__(self) -> None:
        times = np.asarray(self.times_h, dtype=np.float64).copy()
        concentrations = np.asarray(
            self.concentrations_mg_l, dtype=np.float64
        ).copy()
        if (
            times.ndim != 1
            or times.size == 0
            or concentrations.shape != times.shape
            or np.any(~np.isfinite(times))
            or np.any(np.diff(times) <= 0.0)
            or np.any(~np.isfinite(concentrations))
            or np.any(concentrations < 0.0)
        ):
            raise ValueError("clinical scoring vectors must be aligned, finite, and nonnegative")
        for label, value in (
            ("observation_spec_sha256", self.observation_spec_sha256),
            ("model_run_contract_sha256", self.model_run_contract_sha256),
        ):
            if _SHA_RE.fullmatch(value) is None:
                raise ValueError(f"{label} must be a lowercase SHA-256")
        if self.source_column != "observation_captopril_mg_l":
            raise PermissionError("generic central-state columns cannot be scored")
        if self.generic_summary_used is not False:
            raise PermissionError("generic PK summaries are internal-state diagnostics only")
        times.setflags(write=False)
        concentrations.setflags(write=False)
        object.__setattr__(self, "times_h", times)
        object.__setattr__(self, "concentrations_mg_l", concentrations)


def build_captopril_clinical_scoring_input(
    model: CaptoprilRenalPBPKModel,
    trajectory: pd.DataFrame,
    *,
    expected_observation_spec_sha256: str,
) -> CaptoprilClinicalScoringInput:
    """Build scoring input only from the exact authorized observation column."""

    if not isinstance(model, CaptoprilRenalPBPKModel):
        raise TypeError("model must be a CaptoprilRenalPBPKModel")
    if not isinstance(trajectory, pd.DataFrame) or trajectory.empty:
        raise TypeError("trajectory must be a non-empty DataFrame")
    if _SHA_RE.fullmatch(str(expected_observation_spec_sha256)) is None:
        raise ValueError("expected_observation_spec_sha256 must be a lowercase SHA-256")
    if (
        model.observation_spec is None
        or model.observation_spec_sha256 is None
        or model.observation_spec_sha256 != expected_observation_spec_sha256
    ):
        raise PermissionError("frozen captopril ObservationSpec hash is absent or changed")
    required = {
        "time_h",
        "observation_captopril_authorized",
        "observation_identity_resolved",
        "observation_captopril_mg_l",
        "observation_spec_sha256",
        "model_run_contract_sha256",
        "legacy_plasma_parent_alias_is_clinical_observation",
        "generic_pk_summary_authorized_for_clinical_scoring",
    }
    missing = sorted(required - set(trajectory.columns))
    if missing:
        raise PermissionError(
            "authorized captopril observation column contract is incomplete: "
            + ", ".join(missing)
        )
    if (
        not trajectory["observation_captopril_authorized"].eq(1.0).all()
        or not trajectory["observation_identity_resolved"].eq(1.0).all()
        or not trajectory[
            "legacy_plasma_parent_alias_is_clinical_observation"
        ].eq(0.0).all()
        or not trajectory[
            "generic_pk_summary_authorized_for_clinical_scoring"
        ].eq(0.0).all()
    ):
        raise PermissionError("captopril observation authorization is false or ambiguous")
    if (
        not trajectory["observation_spec_sha256"].eq(
            expected_observation_spec_sha256
        ).all()
        or not trajectory["model_run_contract_sha256"].eq(
            model.mechanism_run_contract_sha256
        ).all()
    ):
        raise PermissionError("trajectory observation/run contract hash is absent or changed")
    return CaptoprilClinicalScoringInput(
        times_h=trajectory["time_h"].to_numpy(dtype=np.float64),
        concentrations_mg_l=trajectory[
            "observation_captopril_mg_l"
        ].to_numpy(dtype=np.float64),
        observation_spec_sha256=expected_observation_spec_sha256,
        model_run_contract_sha256=model.mechanism_run_contract_sha256,
    )


@dataclass(frozen=True)
class CaptoprilPositivityReport:
    """Raw, pre-clipping positivity and parent-moiety mass audit."""

    schema_version: str
    model_run_contract_sha256: str
    solver_method: str
    route: str
    dose_mg: float
    duration_h: float
    solver_success: bool
    raw_min_state_umol: float
    boundary_min_derivative_umol_h: float
    clip_parent_equivalent_mass_mg: float
    tolerance_umol: float
    clip_mass_tolerance_mg: float
    passed: bool

    def __post_init__(self) -> None:
        if self.schema_version != "our_star.captopril_positivity_report.v1":
            raise ValueError("unsupported captopril positivity report schema")
        if _SHA_RE.fullmatch(self.model_run_contract_sha256) is None:
            raise ValueError("positivity report requires a run-contract SHA-256")
        if self.route not in {"oral", "iv_bolus"}:
            raise ValueError("positivity report route is invalid")
        numeric = (
            self.dose_mg,
            self.duration_h,
            self.raw_min_state_umol,
            self.boundary_min_derivative_umol_h,
            self.clip_parent_equivalent_mass_mg,
            self.tolerance_umol,
            self.clip_mass_tolerance_mg,
        )
        if any(not np.isfinite(value) for value in numeric):
            raise ValueError("positivity report values must be finite")
        if self.dose_mg <= 0.0 or self.duration_h <= 0.0:
            raise ValueError("positivity report dose and duration must be positive")
        if self.tolerance_umol < 0.0 or self.clip_mass_tolerance_mg < 0.0:
            raise ValueError("positivity tolerances must be nonnegative")


def audit_captopril_raw_positivity(
    model: CaptoprilRenalPBPKModel,
    patient: Any,
    *,
    route: str,
    dose_mg: float,
    duration_h: float,
    solver_method: str = "BDF",
    rtol: float = 1.0e-8,
    atol: float = 1.0e-10,
    max_step_h: float = 0.1,
    tolerance_umol: float = 1.0e-8,
    clip_mass_tolerance_mg: float = 1.0e-9,
) -> CaptoprilPositivityReport:
    """Solve raw states and fail closed on clipping mass or boundary outflow."""

    if not isinstance(model, CaptoprilRenalPBPKModel):
        raise TypeError("model must be a CaptoprilRenalPBPKModel")
    dose = SpeciesDoseEvent(
        time_h=0.0,
        species_id=PARENT,
        amount=dose_mg,
        unit=AmountUnit.MG,
        route=route,
        dose_basis_id="captopril_parent",
    )
    initial = model.apply_species_doses(model.initial_state(), (dose,))
    solution = solve_ivp(
        lambda time_h, state: model.rhs(time_h, state, patient, model.drug),
        (0.0, float(duration_h)),
        initial,
        method=solver_method,
        rtol=float(rtol),
        atol=float(atol),
        max_step=float(max_step_h),
    )
    raw_min = float(solution.y.min()) if solution.y.size else float("-inf")
    negative_umol = np.maximum(-solution.y, 0.0)
    clip_mass = float(
        negative_umol.sum(axis=0).max(initial=0.0)
        * model.drug.molecular_weight_g_mol
        / 1000.0
    )
    boundary_min = float("inf")
    zero = model.initial_state()
    probes = (zero, initial, *tuple(solution.y.T))
    for probe in probes:
        for index in range(len(model.state_registry)):
            boundary = np.maximum(np.asarray(probe, dtype=np.float64), 0.0)
            boundary[index] = 0.0
            derivative = model.rhs(0.0, boundary, patient, model.drug)
            boundary_min = min(boundary_min, float(derivative[index]))
    passed = bool(
        solution.success
        and raw_min >= -float(tolerance_umol)
        and boundary_min >= -float(tolerance_umol)
        and clip_mass <= float(clip_mass_tolerance_mg)
    )
    report = CaptoprilPositivityReport(
        schema_version="our_star.captopril_positivity_report.v1",
        model_run_contract_sha256=model.mechanism_run_contract_sha256,
        solver_method=str(solver_method),
        route=route,
        dose_mg=float(dose_mg),
        duration_h=float(duration_h),
        solver_success=bool(solution.success),
        raw_min_state_umol=raw_min,
        boundary_min_derivative_umol_h=boundary_min,
        clip_parent_equivalent_mass_mg=clip_mass,
        tolerance_umol=float(tolerance_umol),
        clip_mass_tolerance_mg=float(clip_mass_tolerance_mg),
        passed=passed,
    )
    if not report.passed:
        raise FloatingPointError(
            "captopril raw positivity/clip-mass gate failed: "
            f"solver={report.solver_success}, min={report.raw_min_state_umol:.3e} umol, "
            f"boundary={report.boundary_min_derivative_umol_h:.3e} umol/h, "
            f"clip={report.clip_parent_equivalent_mass_mg:.3e} mg"
        )
    return report


def load_captopril_candidate(
    path: str | Path,
) -> tuple[DrugParameters, CaptoprilMechanismParameters, Mapping[str, Any]]:
    """Load the frozen v0.5 candidate without any clinical outcome access."""

    config_path = Path(path)
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            payload = yaml.load(handle, Loader=_UniqueSafeLoader)
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise CaptoprilCandidateError("cannot read captopril candidate config") from error
    payload = _mapping(payload, "captopril candidate config")
    _exact_keys(
        payload,
        frozenset(
            {
                "schema_version",
                "candidate_id",
                "status",
                "clinical_status",
                "context_of_use",
                "drug",
                "mechanism",
                "dose_contract",
                "evidence",
                "engineering_assumptions",
                "forbidden_outcome_access_during_this_build",
            }
        ),
        "captopril candidate config",
    )
    if (
        payload["schema_version"] != CANDIDATE_SCHEMA_VERSION
        or payload["candidate_id"] != CANDIDATE_ID
        or payload["status"] != CANDIDATE_STATUS
        or payload["clinical_status"] != CLINICAL_STATUS
        or not isinstance(payload["context_of_use"], str)
        or not payload["context_of_use"].strip()
    ):
        raise CaptoprilCandidateError("captopril candidate identity/status changed")

    drug_raw = _mapping(payload["drug"], "captopril drug")
    _exact_keys(
        drug_raw,
        frozenset(
            {
                *CANONICAL_ACTIVE_DRUG_VALUES,
                *CANONICAL_UNUSED_DRUG_VALUES,
                "parameter_provenance",
            }
        ),
        "captopril drug",
    )
    mechanism = _mapping(payload["mechanism"], "captopril mechanism")
    _exact_keys(
        mechanism,
        frozenset(
            {
                "mechanism_id",
                "status",
                "provenance",
                "fixed_parameters",
                "deterministic_derivations",
                "mechanism_factors",
                "reversible_disulfide_candidate",
            }
        ),
        "captopril mechanism",
    )
    if (
        mechanism["mechanism_id"] != MECHANISM_ID
        or mechanism["status"] != MECHANISM_STATUS
        or not isinstance(mechanism["provenance"], str)
        or not mechanism["provenance"].strip()
    ):
        raise CaptoprilCandidateError("captopril mechanism identity/status changed")
    fixed = _mapping(mechanism["fixed_parameters"], "fixed mechanism parameters")
    _exact_keys(
        fixed,
        frozenset(
            {
                "fraction_unbound",
                "reference_gfr_l_h",
                "renal_clearance_l_h",
                "systemic_total_clearance_l_h",
            }
        ),
        "fixed mechanism parameters",
    )
    disulfide = _mapping(
        mechanism["reversible_disulfide_candidate"],
        "reversible disulfide candidate",
    )
    _exact_keys(
        disulfide,
        frozenset({"enabled", "status", "forward_rate_h", "reverse_rate_h", "reason"}),
        "reversible disulfide candidate",
    )
    if (
        disulfide["enabled"] is not False
        or disulfide["status"] != "disabled_not_admitted"
        or disulfide["forward_rate_h"] is not None
        or disulfide["reverse_rate_h"] is not None
        or not isinstance(disulfide["reason"], str)
        or not disulfide["reason"].strip()
    ):
        raise CaptoprilCandidateError("reversible disulfide candidate is fail-open")
    parameters = CaptoprilMechanismParameters(
        fraction_unbound=float(fixed["fraction_unbound"]),
        reference_gfr_l_h=float(fixed["reference_gfr_l_h"]),
        renal_clearance_l_h=float(fixed["renal_clearance_l_h"]),
        systemic_total_clearance_l_h=float(
            fixed["systemic_total_clearance_l_h"]
        ),
        disulfide_exchange_enabled=bool(disulfide["enabled"]),
        provenance=str(mechanism["provenance"]),
    )
    drug = DrugParameters.from_mapping(drug_raw)
    active_record = _validate_canonical_model_parameters(drug, parameters)

    derivations = _mapping(
        mechanism["deterministic_derivations"], "deterministic derivations"
    )
    _exact_keys(
        derivations,
        frozenset(CANONICAL_DETERMINISTIC_DERIVATIONS),
        "deterministic derivations",
    )
    expected_formulas = {
        "filtration_clearance_on_total_l_h": "fraction_unbound * reference_gfr_l_h",
        "secretion_clearance_on_total_l_h": (
            "renal_clearance_l_h - filtration_clearance_on_total_l_h"
        ),
        "secretion_clearance_on_unbound_l_h": (
            "secretion_clearance_on_total_l_h / fraction_unbound"
        ),
        "unresolved_nonrenal_clearance_l_h": (
            "systemic_total_clearance_l_h - renal_clearance_l_h"
        ),
    }
    for name, expected in CANONICAL_DETERMINISTIC_DERIVATIONS.items():
        item = _mapping(derivations[name], f"derivation {name}")
        _exact_keys(item, frozenset({"formula", "value"}), f"derivation {name}")
        if item["formula"] != expected_formulas[name]:
            raise CaptoprilCandidateError(f"derivation formula changed: {name}")
        _exact_float(item["value"], expected, f"derivation value {name}")

    factors = mechanism["mechanism_factors"]
    if not isinstance(factors, list) or factors != [
        "glomerular_filtration",
        "active_tubular_secretion",
        "unresolved_nonrenal_loss",
    ]:
        raise CaptoprilCandidateError("configured mechanism factor set/order changed")

    dose_contract = _mapping(payload["dose_contract"], "dose contract")
    _exact_keys(
        dose_contract,
        frozenset(
            {
                "accepted_basis_id",
                "legacy_basis_id",
                "oral_fraction_absorbed",
                "oral_accounting",
            }
        ),
        "dose contract",
    )
    if (
        dose_contract["accepted_basis_id"] != "captopril_parent"
        or dose_contract["legacy_basis_id"] != "legacy_active_parent_mg"
        or not isinstance(dose_contract["oral_accounting"], str)
        or not dose_contract["oral_accounting"].strip()
    ):
        raise CaptoprilCandidateError("captopril dose contract identity changed")
    _exact_float(
        dose_contract["oral_fraction_absorbed"],
        CANONICAL_ACTIVE_DRUG_VALUES["fraction_absorbed"],
        "dose_contract.oral_fraction_absorbed",
    )

    evidence = _mapping(payload["evidence"], "evidence")
    _exact_keys(
        evidence,
        frozenset(
            {
                "official_label",
                "primary_healthy_iv_oral_pk",
                "primary_renal_handling",
                "primary_disulfide_clinical_evidence",
            }
        ),
        "evidence",
    )
    if not isinstance(payload["engineering_assumptions"], list) or not payload[
        "engineering_assumptions"
    ]:
        raise CaptoprilCandidateError("engineering assumptions must be nonempty")
    attestation = _mapping(
        payload["forbidden_outcome_access_during_this_build"],
        "outcome-access attestation",
    )
    _exact_keys(
        attestation,
        frozenset({"study_ids", "attestation"}),
        "outcome-access attestation",
    )
    if (
        attestation["study_ids"] != ["PKDB00822", "PKDB01088", "PKDB01069"]
        or not isinstance(attestation["attestation"], str)
        or not attestation["attestation"].strip()
    ):
        raise CaptoprilCandidateError("outcome-access attestation changed")

    config_content_sha256 = _canonical_sha256(payload)
    expected_config_sha256 = payload.get("canonical_config_sha256")
    if expected_config_sha256 is not None:
        # Reserved for a future self-hashed schema; unknown top keys remain
        # forbidden in v1, so a mutable unverified digest cannot be smuggled in.
        raise CaptoprilCandidateError("self-hash is not part of the v1 config schema")
    result = dict(payload)
    result["canonical_config_sha256"] = config_content_sha256
    result["canonical_active_parameter_sha256"] = _canonical_sha256(active_record)
    return drug, parameters, result


__all__ = [
    "PARENT",
    "UNRESOLVED_NONRENAL",
    "CANDIDATE_ID",
    "CANDIDATE_SCHEMA_VERSION",
    "CANONICAL_ACTIVE_DRUG_VALUES",
    "CANONICAL_MECHANISM_VALUES",
    "CANONICAL_UNUSED_DRUG_VALUES",
    "CaptoprilCandidateError",
    "CaptoprilClinicalScoringInput",
    "CaptoprilMechanismParameters",
    "CaptoprilPositivityReport",
    "CaptoprilRenalPBPKModel",
    "audit_captopril_raw_positivity",
    "build_captopril_reaction_network",
    "build_captopril_clinical_scoring_input",
    "captopril_observation_spec_record",
    "captopril_observation_spec_sha256",
    "captopril_state",
    "load_captopril_candidate",
]
