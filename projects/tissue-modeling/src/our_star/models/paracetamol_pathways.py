"""Development-scaffold paracetamol UGT/SULT/oxidative pathway PBPK model.

This module is a software/mechanism scaffold, not a fitted or externally
validated paracetamol predictor.  It reuses only independently documented v0.3
parent/formulation inputs and explicitly labels synthetic placeholder pathway
values in configuration.  No clinical outcome is read by this code.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from ..chemistry.accounting import (
    DualAmountLedger,
    ExternalParticipant,
    ExternalParticipantRole,
    OpenReaction,
    OpenReactionNetwork,
    SpeciesMoiety,
    formula_molecular_weight_g_mol,
)
from ..chemistry.kinetics import (
    LinearClearanceKinetics,
    MichaelisMentenKinetics,
    ScaledKineticPathway,
    evaluate_competing_pathways,
)
from ..chemistry.reactions import MolecularFormula, SpeciesFormula, StoichiometricTerm
from ..core.amount_state import AmountStateRegistry, AmountStateSpec, FluxKey, SpeciesDoseEvent
from ..core.amount_system import (
    AmountModelContext,
    AmountModuleResult,
    AmountSystemAssembler,
)
from ..core.species import AmountUnit, ChemicalSpecies, SpeciesRegistry
from ..models.segmented import FormulationParameters
from ..pbpk import DrugParameters, PatientPhysiology
from .multispecies import MultiSpeciesPBPKModel


PARACETAMOL = "paracetamol"
PARACETAMOL_GLUCURONIDE = "paracetamol_glucuronide"
PARACETAMOL_SULFATE = "paracetamol_sulfate"
PARACETAMOL_OXIDATIVE_CONJUGATE = "paracetamol_oxidative_conjugate"
PARACETAMOL_METABOLITES = (
    PARACETAMOL_GLUCURONIDE,
    PARACETAMOL_SULFATE,
    PARACETAMOL_OXIDATIVE_CONJUGATE,
)

GLUCURONYL_EQUIVALENT = "glucuronyl_equivalent"
SULFATE_EQUIVALENT = "sulfate_equivalent"
CYSTEINE_EQUIVALENT = "cysteine_equivalent"
HYDROGEN_EQUIVALENT = "hydrogen_equivalent"

UGT_REACTION = "paracetamol_ugt_conjugation"
SULT_REACTION = "paracetamol_sult_conjugation"
CYP_REACTION = "paracetamol_cyp_to_aggregate_thiol_conjugate"

SEGMENTS = ("duodenum", "jejunum", "ileum")
LIVER_ZONES = ("liver_zone1", "liver_zone2", "liver_zone3")
REFERENCE_LIVER_VOLUME_L = 1.8
PORTAL_PARENT = FluxKey("portal_to_liver", PARACETAMOL)


def amount_state_name(compartment_id: str, species_id: str) -> str:
    return f"{compartment_id}::{species_id}"


def external_import_state_name(participant_id: str) -> str:
    return f"external_import::{participant_id}"


def external_export_state_name(participant_id: str) -> str:
    return f"external_export::{participant_id}"


@dataclass(frozen=True)
class SaturablePathwayParameters:
    reaction_id: str
    mechanism_id: str
    vmax_umol_h: float
    km_umol_l: float
    zone_activity: tuple[float, float, float]
    parameter_provenance: str

    def __post_init__(self) -> None:
        if not self.reaction_id.strip() or not self.mechanism_id.strip():
            raise ValueError("pathway reaction_id and mechanism_id are required")
        vmax = float(self.vmax_umol_h)
        km = float(self.km_umol_l)
        zones = tuple(float(value) for value in self.zone_activity)
        if not np.isfinite(vmax) or vmax < 0.0:
            raise ValueError("pathway vmax_umol_h must be finite and >= 0")
        if not np.isfinite(km) or km <= 0.0:
            raise ValueError("pathway km_umol_l must be finite and > 0")
        if len(zones) != 3 or any(
            not np.isfinite(value) or value < 0.0 for value in zones
        ):
            raise ValueError("pathway zone_activity requires three nonnegative values")
        if sum(zones) <= 0.0:
            raise ValueError("at least one pathway zone must have positive activity")
        if not self.parameter_provenance.strip():
            raise ValueError("pathway parameter_provenance is required")
        object.__setattr__(self, "vmax_umol_h", vmax)
        object.__setattr__(self, "km_umol_l", km)
        object.__setattr__(self, "zone_activity", zones)

    @property
    def normalized_zone_activity(self) -> tuple[float, float, float]:
        total = float(sum(self.zone_activity))
        return tuple(value / total for value in self.zone_activity)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "SaturablePathwayParameters":
        values = dict(raw)
        values["zone_activity"] = tuple(values["zone_activity"])
        return cls(**values)


@dataclass(frozen=True)
class LinearPathwayParameters:
    reaction_id: str
    mechanism_id: str
    intrinsic_clearance_l_h: float
    zone_activity: tuple[float, float, float]
    parameter_provenance: str

    def __post_init__(self) -> None:
        if not self.reaction_id.strip() or not self.mechanism_id.strip():
            raise ValueError("pathway reaction_id and mechanism_id are required")
        clearance = float(self.intrinsic_clearance_l_h)
        zones = tuple(float(value) for value in self.zone_activity)
        if not np.isfinite(clearance) or clearance < 0.0:
            raise ValueError("intrinsic_clearance_l_h must be finite and >= 0")
        if len(zones) != 3 or any(
            not np.isfinite(value) or value < 0.0 for value in zones
        ):
            raise ValueError("pathway zone_activity requires three nonnegative values")
        if sum(zones) <= 0.0:
            raise ValueError("at least one pathway zone must have positive activity")
        if not self.parameter_provenance.strip():
            raise ValueError("pathway parameter_provenance is required")
        object.__setattr__(self, "intrinsic_clearance_l_h", clearance)
        object.__setattr__(self, "zone_activity", zones)

    @property
    def normalized_zone_activity(self) -> tuple[float, float, float]:
        total = float(sum(self.zone_activity))
        return tuple(value / total for value in self.zone_activity)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "LinearPathwayParameters":
        values = dict(raw)
        values["zone_activity"] = tuple(values["zone_activity"])
        return cls(**values)


@dataclass(frozen=True)
class ParacetamolPathwayParameters:
    """Pre-outcome Tier-A mechanism parameters and evidence status."""

    ugt: SaturablePathwayParameters
    sult: SaturablePathwayParameters
    cyp: LinearPathwayParameters
    metabolite_renal_clearance_l_h: Mapping[str, float]
    metabolite_liver_partition_coefficient: Mapping[str, float]
    evidence_role: str
    parameter_provenance: str

    def __post_init__(self) -> None:
        if self.ugt.reaction_id != UGT_REACTION:
            raise ValueError(f"UGT reaction_id must be {UGT_REACTION}")
        if self.sult.reaction_id != SULT_REACTION:
            raise ValueError(f"SULT reaction_id must be {SULT_REACTION}")
        if self.cyp.reaction_id != CYP_REACTION:
            raise ValueError(f"CYP reaction_id must be {CYP_REACTION}")
        renal = {
            str(key): float(value)
            for key, value in self.metabolite_renal_clearance_l_h.items()
        }
        partition = {
            str(key): float(value)
            for key, value in self.metabolite_liver_partition_coefficient.items()
        }
        expected = set(PARACETAMOL_METABOLITES)
        if set(renal) != expected or set(partition) != expected:
            raise ValueError(
                "metabolite renal-clearance and liver-partition mappings must "
                f"contain exactly {sorted(expected)}"
            )
        if any(not np.isfinite(value) or value < 0.0 for value in renal.values()):
            raise ValueError("metabolite renal clearances must be finite and >= 0")
        if any(not np.isfinite(value) or value <= 0.0 for value in partition.values()):
            raise ValueError("metabolite liver partition values must be finite and > 0")
        allowed_roles = {
            "independent_translation",
            "synthetic_engine_scaffold",
            "mixed_independent_and_synthetic_scaffold",
        }
        if self.evidence_role not in allowed_roles:
            raise ValueError(f"unsupported evidence_role: {self.evidence_role}")
        if not self.parameter_provenance.strip():
            raise ValueError("paracetamol pathway parameter_provenance is required")
        object.__setattr__(
            self, "metabolite_renal_clearance_l_h", MappingProxyType(renal)
        )
        object.__setattr__(
            self,
            "metabolite_liver_partition_coefficient",
            MappingProxyType(partition),
        )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ParacetamolPathwayParameters":
        values = dict(raw)
        pathways = dict(values.pop("pathways"))
        return cls(
            ugt=SaturablePathwayParameters.from_mapping(pathways["ugt"]),
            sult=SaturablePathwayParameters.from_mapping(pathways["sult"]),
            cyp=LinearPathwayParameters.from_mapping(pathways["cyp"]),
            **values,
        )


def _paracetamol_formulas() -> Mapping[str, MolecularFormula]:
    return MappingProxyType(
        {
            PARACETAMOL: MolecularFormula.from_mapping(
                {"C": 8, "H": 9, "N": 1, "O": 2}
            ),
            PARACETAMOL_GLUCURONIDE: MolecularFormula.from_mapping(
                {"C": 14, "H": 17, "N": 1, "O": 8}
            ),
            PARACETAMOL_SULFATE: MolecularFormula.from_mapping(
                {"C": 8, "H": 9, "N": 1, "O": 5, "S": 1}
            ),
            PARACETAMOL_OXIDATIVE_CONJUGATE: MolecularFormula.from_mapping(
                {"C": 11, "H": 14, "N": 2, "O": 4, "S": 1}
            ),
            GLUCURONYL_EQUIVALENT: MolecularFormula.from_mapping(
                {"C": 6, "H": 8, "O": 6}
            ),
            SULFATE_EQUIVALENT: MolecularFormula.from_mapping({"O": 3, "S": 1}),
            CYSTEINE_EQUIVALENT: MolecularFormula.from_mapping(
                {"C": 3, "H": 7, "N": 1, "O": 2, "S": 1}
            ),
            HYDROGEN_EQUIVALENT: MolecularFormula.from_mapping({"H": 2}),
        }
    )


def build_paracetamol_tier_a_network() -> OpenReactionNetwork:
    """Build formula-balanced UGT, SULT, and aggregate oxidative chemistry."""

    formulas = _paracetamol_formulas()
    physical_species = (
        ChemicalSpecies(
            PARACETAMOL,
            "paracetamol",
            formula_molecular_weight_g_mol(formulas[PARACETAMOL]),
            structure="CC(=O)NC1=CC=C(C=C1)O",
        ),
        ChemicalSpecies(
            PARACETAMOL_GLUCURONIDE,
            "paracetamol glucuronide",
            formula_molecular_weight_g_mol(formulas[PARACETAMOL_GLUCURONIDE]),
        ),
        ChemicalSpecies(
            PARACETAMOL_SULFATE,
            "paracetamol sulfate",
            formula_molecular_weight_g_mol(formulas[PARACETAMOL_SULFATE]),
        ),
        ChemicalSpecies(
            PARACETAMOL_OXIDATIVE_CONJUGATE,
            "aggregate paracetamol cysteine/thiol conjugate",
            formula_molecular_weight_g_mol(
                formulas[PARACETAMOL_OXIDATIVE_CONJUGATE]
            ),
        ),
    )
    formula_records = tuple(
        SpeciesFormula(species.species_id, formulas[species.species_id])
        for species in physical_species
    )
    group_provenance = (
        "Formula-derived net transfer-group bookkeeping; no external clinical "
        "outcome or fitted amount is used."
    )
    return OpenReactionNetwork(
        physical_species,
        (
            OpenReaction(
                UGT_REACTION,
                (
                    StoichiometricTerm(PARACETAMOL, -1.0),
                    StoichiometricTerm(PARACETAMOL_GLUCURONIDE, 1.0),
                ),
                (
                    ExternalParticipant(
                        GLUCURONYL_EQUIVALENT,
                        "net glucuronyl transfer group",
                        -1.0,
                        formula_molecular_weight_g_mol(
                            formulas[GLUCURONYL_EQUIVALENT]
                        ),
                        ExternalParticipantRole.NET_TRANSFER_GROUP,
                        formulas[GLUCURONYL_EQUIVALENT],
                        group_provenance,
                    ),
                ),
                "One parent drug moiety is retained in the glucuronide.",
            ),
            OpenReaction(
                SULT_REACTION,
                (
                    StoichiometricTerm(PARACETAMOL, -1.0),
                    StoichiometricTerm(PARACETAMOL_SULFATE, 1.0),
                ),
                (
                    ExternalParticipant(
                        SULFATE_EQUIVALENT,
                        "net sulfate transfer group",
                        -1.0,
                        formula_molecular_weight_g_mol(
                            formulas[SULFATE_EQUIVALENT]
                        ),
                        ExternalParticipantRole.NET_TRANSFER_GROUP,
                        formulas[SULFATE_EQUIVALENT],
                        group_provenance,
                    ),
                ),
                "One parent drug moiety is retained in the sulfate conjugate.",
            ),
            OpenReaction(
                CYP_REACTION,
                (
                    StoichiometricTerm(PARACETAMOL, -1.0),
                    StoichiometricTerm(
                        PARACETAMOL_OXIDATIVE_CONJUGATE, 1.0
                    ),
                ),
                (
                    ExternalParticipant(
                        CYSTEINE_EQUIVALENT,
                        "aggregate thiol trapping equivalent",
                        -1.0,
                        formula_molecular_weight_g_mol(
                            formulas[CYSTEINE_EQUIVALENT]
                        ),
                        ExternalParticipantRole.ENDOGENOUS_CARRIER,
                        formulas[CYSTEINE_EQUIVALENT],
                        group_provenance,
                    ),
                    ExternalParticipant(
                        HYDROGEN_EQUIVALENT,
                        "net hydrogen coproduct equivalent",
                        1.0,
                        formula_molecular_weight_g_mol(
                            formulas[HYDROGEN_EQUIVALENT]
                        ),
                        ExternalParticipantRole.COPRODUCT,
                        formulas[HYDROGEN_EQUIVALENT],
                        group_provenance,
                    ),
                ),
                "Tier-A aggregate CYP oxidation plus thiol trapping; NAPQI is not a measured state.",
            ),
        ),
        species_formulas=formula_records,
    )


def build_paracetamol_amount_states(
    network: OpenReactionNetwork,
) -> tuple[
    AmountStateRegistry,
    Mapping[str, str],
    Mapping[str, str],
]:
    """Create physical and cumulative external bookkeeping µmol states."""

    accounting_species = tuple(
        ChemicalSpecies(
            participant.participant_id,
            participant.name,
            participant.molecular_weight_g_mol,
        )
        for participant in network.external_participants.values()
    )
    species = SpeciesRegistry((*tuple(network.species), *accounting_species))
    states: list[AmountStateSpec] = []

    parent_compartments = (
        "stomach_solid",
        "stomach_dissolved",
        *(f"{segment}_lumen" for segment in SEGMENTS),
        *(f"{segment}_wall" for segment in SEGMENTS),
        "central",
        *LIVER_ZONES,
        "kidney",
        "rest",
        "urine",
        "feces",
    )
    for compartment in parent_compartments:
        states.append(
            AmountStateSpec(
                amount_state_name(compartment, PARACETAMOL),
                PARACETAMOL,
                compartment,
                sink=compartment in {"urine", "feces"},
            )
        )
    for species_id in PARACETAMOL_METABOLITES:
        for compartment in ("central", *LIVER_ZONES, "urine"):
            states.append(
                AmountStateSpec(
                    amount_state_name(compartment, species_id),
                    species_id,
                    compartment,
                    sink=compartment == "urine",
                )
            )

    import_states: dict[str, str] = {}
    export_states: dict[str, str] = {}
    for participant_id, participant in network.external_participants.items():
        if any(
            item.participant_id == participant_id and item.coefficient < 0.0
            for reaction in network.reactions
            for item in reaction.external_participants
        ):
            name = external_import_state_name(participant_id)
            import_states[participant_id] = name
            states.append(
                AmountStateSpec(
                    name,
                    participant_id,
                    "external_import",
                    sink=True,
                    accounting_only=True,
                )
            )
        if any(
            item.participant_id == participant_id and item.coefficient > 0.0
            for reaction in network.reactions
            for item in reaction.external_participants
        ):
            name = external_export_state_name(participant_id)
            export_states[participant_id] = name
            states.append(
                AmountStateSpec(
                    name,
                    participant_id,
                    "external_export",
                    sink=True,
                    accounting_only=True,
                )
            )
    return (
        AmountStateRegistry(states, species),
        MappingProxyType(import_states),
        MappingProxyType(export_states),
    )


def _amount(
    states: AmountStateRegistry,
    amounts_umol: np.ndarray,
    compartment_id: str,
    species_id: str,
) -> float:
    return float(amounts_umol[states.index(amount_state_name(compartment_id, species_id))])


class _ParacetamolGutModule:
    name = "paracetamol_segmented_gut_umol"
    required_states = tuple(
        amount_state_name(compartment, PARACETAMOL)
        for compartment in (
            "stomach_solid",
            "stomach_dissolved",
            *(f"{segment}_lumen" for segment in SEGMENTS),
            *(f"{segment}_wall" for segment in SEGMENTS),
            "central",
            "feces",
        )
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
        drug = context.drug
        formulation = context.formulation
        if not isinstance(formulation, FormulationParameters):
            raise TypeError("paracetamol gut requires FormulationParameters")
        central = _amount(states, amounts_umol, "central", PARACETAMOL)
        central_concentration = central / patient.central_volume_l
        solid = _amount(states, amounts_umol, "stomach_solid", PARACETAMOL)
        dissolved = _amount(states, amounts_umol, "stomach_dissolved", PARACETAMOL)
        dissolution = formulation.dissolution_rate_h * solid
        gastric_emptying = formulation.gastric_emptying_rate_h * dissolved
        derivatives: dict[str, float] = {
            amount_state_name("stomach_solid", PARACETAMOL): -dissolution,
            amount_state_name("stomach_dissolved", PARACETAMOL): (
                dissolution - gastric_emptying
            ),
            amount_state_name("central", PARACETAMOL): (
                -patient.portal_flow_l_h * central_concentration
            ),
        }
        incoming_lumen = gastric_emptying
        returns: list[float] = []
        for index, segment in enumerate(SEGMENTS):
            lumen = _amount(states, amounts_umol, f"{segment}_lumen", PARACETAMOL)
            wall = _amount(states, amounts_umol, f"{segment}_wall", PARACETAMOL)
            absorption = (
                drug.fraction_absorbed
                * formulation.segment_absorption_rate_h[index]
                * lumen
            )
            transit = formulation.segment_transit_rate_h[index] * lumen
            wall_volume = (
                patient.gut_volume_l * formulation.wall_volume_fraction[index]
            )
            wall_equivalent = wall / wall_volume / drug.kp_gut
            segment_flow = (
                patient.portal_flow_l_h * formulation.portal_flow_fraction[index]
            )
            venous_return = segment_flow * wall_equivalent
            derivatives[amount_state_name(f"{segment}_lumen", PARACETAMOL)] = (
                incoming_lumen - absorption - transit
            )
            derivatives[amount_state_name(f"{segment}_wall", PARACETAMOL)] = (
                absorption
                + segment_flow * central_concentration
                - venous_return
            )
            incoming_lumen = transit
            returns.append(venous_return)
        derivatives[amount_state_name("feces", PARACETAMOL)] = incoming_lumen
        return AmountModuleResult(
            derivatives,
            {PORTAL_PARENT: float(sum(returns))},
        )


class _ParacetamolLiverModule:
    name = "paracetamol_multispecies_liver_umol"
    required_inputs = (PORTAL_PARENT,)

    def __init__(
        self,
        pathway_parameters: ParacetamolPathwayParameters,
        network: OpenReactionNetwork,
        import_states: Mapping[str, str],
        export_states: Mapping[str, str],
    ) -> None:
        self.parameters = pathway_parameters
        self.network = network
        self.import_states = dict(import_states)
        self.export_states = dict(export_states)
        physical = tuple(
            amount_state_name(compartment, species_id)
            for species_id in (PARACETAMOL, *PARACETAMOL_METABOLITES)
            for compartment in ("central", *LIVER_ZONES)
        )
        self.required_states = (
            *physical,
            *tuple(import_states.values()),
            *tuple(export_states.values()),
        )
        self._pathways_by_zone = tuple(
            (
                ScaledKineticPathway(
                    UGT_REACTION,
                    pathway_parameters.ugt.mechanism_id,
                    MichaelisMentenKinetics(
                        pathway_parameters.ugt.vmax_umol_h,
                        pathway_parameters.ugt.km_umol_l,
                    ),
                    pathway_parameters.ugt.normalized_zone_activity[index],
                ),
                ScaledKineticPathway(
                    SULT_REACTION,
                    pathway_parameters.sult.mechanism_id,
                    MichaelisMentenKinetics(
                        pathway_parameters.sult.vmax_umol_h,
                        pathway_parameters.sult.km_umol_l,
                    ),
                    pathway_parameters.sult.normalized_zone_activity[index],
                ),
                ScaledKineticPathway(
                    CYP_REACTION,
                    pathway_parameters.cyp.mechanism_id,
                    LinearClearanceKinetics(
                        pathway_parameters.cyp.intrinsic_clearance_l_h
                    ),
                    pathway_parameters.cyp.normalized_zone_activity[index],
                ),
            )
            for index in range(3)
        )

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
        zone_volume = patient.liver_volume_l / 3.0
        q_liver = patient.liver_flow_l_h
        parent_central = (
            _amount(states, amounts_umol, "central", PARACETAMOL)
            / patient.central_volume_l
        )
        parent_zone_equivalent = tuple(
            _amount(states, amounts_umol, zone, PARACETAMOL)
            / zone_volume
            / drug.kp_liver
            for zone in LIVER_ZONES
        )
        shared_scale = (
            patient.enzyme_activity_factor
            * patient.liver_function_fraction
            * patient.liver_volume_l
            / REFERENCE_LIVER_VOLUME_L
        )
        rates_by_zone = tuple(
            evaluate_competing_pathways(
                drug.fraction_unbound_plasma * parent_zone_equivalent[index],
                self._pathways_by_zone[index],
                mechanism_factors=context.mechanism_factors,
                shared_scale=shared_scale,
            )
            for index in range(3)
        )
        reaction_by_zone = tuple(
            self.network.derivatives(rates) for rates in rates_by_zone
        )

        derivatives: dict[str, float] = {
            amount_state_name("liver_zone1", PARACETAMOL): (
                inputs_umol_h[PORTAL_PARENT]
                + patient.hepatic_artery_flow_l_h * parent_central
                - q_liver * parent_zone_equivalent[0]
            ),
            amount_state_name("liver_zone2", PARACETAMOL): (
                q_liver * parent_zone_equivalent[0]
                - q_liver * parent_zone_equivalent[1]
            ),
            amount_state_name("liver_zone3", PARACETAMOL): (
                q_liver * parent_zone_equivalent[1]
                - q_liver * parent_zone_equivalent[2]
            ),
            amount_state_name("central", PARACETAMOL): (
                q_liver * parent_zone_equivalent[2]
                - patient.hepatic_artery_flow_l_h * parent_central
            ),
        }
        for index, reaction_derivatives in enumerate(reaction_by_zone):
            zone = LIVER_ZONES[index]
            for species_id, rate in reaction_derivatives.species_umol_h.items():
                state_name = amount_state_name(zone, species_id)
                derivatives[state_name] = derivatives.get(state_name, 0.0) + rate
            for participant_id, signed_rate in (
                reaction_derivatives.external_participant_umol_h.items()
            ):
                if signed_rate < 0.0:
                    state_name = self.import_states[participant_id]
                    derivatives[state_name] = (
                        derivatives.get(state_name, 0.0) - signed_rate
                    )
                elif signed_rate > 0.0:
                    state_name = self.export_states[participant_id]
                    derivatives[state_name] = (
                        derivatives.get(state_name, 0.0) + signed_rate
                    )

        for species_id in PARACETAMOL_METABOLITES:
            central_concentration = (
                _amount(states, amounts_umol, "central", species_id)
                / patient.central_volume_l
            )
            kp = self.parameters.metabolite_liver_partition_coefficient[species_id]
            zone_equivalent = tuple(
                _amount(states, amounts_umol, zone, species_id)
                / zone_volume
                / kp
                for zone in LIVER_ZONES
            )
            derivatives[amount_state_name("central", species_id)] = (
                q_liver * zone_equivalent[2]
                - q_liver * central_concentration
            )
            for index, zone in enumerate(LIVER_ZONES):
                incoming = (
                    q_liver * central_concentration
                    if index == 0
                    else q_liver * zone_equivalent[index - 1]
                )
                state_name = amount_state_name(zone, species_id)
                derivatives[state_name] = (
                    derivatives.get(state_name, 0.0)
                    + incoming
                    - q_liver * zone_equivalent[index]
                )
        return AmountModuleResult(derivatives)


class _ParacetamolKidneyModule:
    name = "paracetamol_multispecies_kidney_umol"
    required_inputs: tuple[FluxKey, ...] = ()

    def __init__(self, parameters: ParacetamolPathwayParameters) -> None:
        self.parameters = parameters
        self.required_states = (
            amount_state_name("central", PARACETAMOL),
            amount_state_name("kidney", PARACETAMOL),
            amount_state_name("urine", PARACETAMOL),
            *tuple(
                amount_state_name(compartment, species_id)
                for species_id in PARACETAMOL_METABOLITES
                for compartment in ("central", "urine")
            ),
        )

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
        central = (
            _amount(states, amounts_umol, "central", PARACETAMOL)
            / patient.central_volume_l
        )
        kidney_equivalent = (
            _amount(states, amounts_umol, "kidney", PARACETAMOL)
            / patient.kidney_volume_l
            / drug.kp_kidney
        )
        renal_scale = (
            patient.renal_function_fraction
            * (patient.body_weight_kg / 70.0) ** 0.75
        )
        parent_elimination = (
            drug.renal_clearance_l_h * renal_scale * kidney_equivalent
        )
        exchange = patient.renal_flow_l_h * (central - kidney_equivalent)
        derivatives: dict[str, float] = {
            amount_state_name("kidney", PARACETAMOL): (
                exchange - parent_elimination
            ),
            amount_state_name("central", PARACETAMOL): -exchange,
            amount_state_name("urine", PARACETAMOL): parent_elimination,
        }
        for species_id in PARACETAMOL_METABOLITES:
            central_concentration = (
                _amount(states, amounts_umol, "central", species_id)
                / patient.central_volume_l
            )
            elimination = (
                self.parameters.metabolite_renal_clearance_l_h[species_id]
                * renal_scale
                * central_concentration
            )
            derivatives[amount_state_name("central", species_id)] = -elimination
            derivatives[amount_state_name("urine", species_id)] = elimination
        return AmountModuleResult(derivatives)


class _ParacetamolRestModule:
    name = "paracetamol_rest_of_body_umol"
    required_states = (
        amount_state_name("central", PARACETAMOL),
        amount_state_name("rest", PARACETAMOL),
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
        drug = context.drug
        central = (
            _amount(states, amounts_umol, "central", PARACETAMOL)
            / patient.central_volume_l
        )
        rest_equivalent = (
            _amount(states, amounts_umol, "rest", PARACETAMOL)
            / patient.rest_volume_l
            / drug.kp_rest
        )
        exchange = patient.rest_flow_l_h * (central - rest_equivalent)
        return AmountModuleResult(
            {
                amount_state_name("rest", PARACETAMOL): exchange,
                amount_state_name("central", PARACETAMOL): -exchange,
            }
        )


class ParacetamolPathwayPBPKModel(MultiSpeciesPBPKModel):
    """Three-pathway, stable-metabolite paracetamol development scaffold."""

    def __init__(
        self,
        drug: DrugParameters,
        formulation: FormulationParameters,
        pathway_parameters: ParacetamolPathwayParameters,
        *,
        mechanism_factors: Mapping[str, float] | None = None,
    ) -> None:
        network = build_paracetamol_tier_a_network()
        states, import_states, export_states = build_paracetamol_amount_states(
            network
        )
        parent_mw = network.species[PARACETAMOL].molecular_weight_g_mol
        if abs(drug.molecular_weight_g_mol - parent_mw) > 0.5:
            raise ValueError(
                "drug molecular weight is inconsistent with formula-derived paracetamol"
            )
        ledger = DualAmountLedger(
            states,
            (
                SpeciesMoiety(species_id, {PARACETAMOL: 1.0})
                for species_id in (PARACETAMOL, *PARACETAMOL_METABOLITES)
            ),
            conserved_moiety_id=PARACETAMOL,
            reference_species_id=PARACETAMOL,
            external_participants=network.external_participants,
            external_import_states=import_states,
            external_export_states=export_states,
        )
        assembler = AmountSystemAssembler(
            states,
            (
                _ParacetamolGutModule(),
                _ParacetamolLiverModule(
                    pathway_parameters,
                    network,
                    import_states,
                    export_states,
                ),
                _ParacetamolKidneyModule(pathway_parameters),
                _ParacetamolRestModule(),
            ),
        )
        self.pathway_parameters = pathway_parameters
        self.external_import_states = import_states
        self.external_export_states = export_states
        super().__init__(
            model_id="our_star.paracetamol_multispecies_pbpk.v0.4.dev",
            drug=drug,
            formulation=formulation,
            states=states,
            assembler=assembler,
            reaction_network=network,
            ledger=ledger,
            parent_species_id=PARACETAMOL,
            dose_targets={
                ("oral", PARACETAMOL): amount_state_name(
                    "stomach_solid", PARACETAMOL
                ),
                ("iv_bolus", PARACETAMOL): amount_state_name(
                    "central", PARACETAMOL
                ),
            },
            central_states={
                species_id: amount_state_name("central", species_id)
                for species_id in (PARACETAMOL, *PARACETAMOL_METABOLITES)
            },
            urine_states={
                species_id: amount_state_name("urine", species_id)
                for species_id in (PARACETAMOL, *PARACETAMOL_METABOLITES)
            },
            feces_parent_state=amount_state_name("feces", PARACETAMOL),
            parent_liver_states=tuple(
                amount_state_name(zone, PARACETAMOL) for zone in LIVER_ZONES
            ),
            mechanism_factors=mechanism_factors,
        )

    def apply_species_doses(
        self,
        state: np.ndarray,
        events: Sequence[SpeciesDoseEvent],
    ) -> np.ndarray:
        values = self.state_registry.validate_vector(state).copy()
        for event in events:
            if event.species_id != PARACETAMOL:
                raise ValueError("Tier-A model accepts paracetamol dose events only")
            amount_umol = float(
                self.state_registry.species[PARACETAMOL].convert_amount(
                    event.amount, event.unit, AmountUnit.UMOL
                )
            )
            if event.route == "iv_bolus":
                values[
                    self.state_registry.index(
                        amount_state_name("central", PARACETAMOL)
                    )
                ] += amount_umol
                continue
            dissolved = amount_umol * self.formulation.initial_dissolved_fraction
            values[
                self.state_registry.index(
                    amount_state_name("stomach_dissolved", PARACETAMOL)
                )
            ] += dissolved
            values[
                self.state_registry.index(
                    amount_state_name("stomach_solid", PARACETAMOL)
                )
            ] += amount_umol - dissolved
        return values

    def concentration_record(
        self,
        state: np.ndarray,
        patient: PatientPhysiology,
        drug: DrugParameters,
    ) -> dict[str, float]:
        record = super().concentration_record(state, patient, drug)
        values = self.state_registry.validate_vector(state)
        wall_umol = sum(
            _amount(
                self.state_registry,
                values,
                f"{segment}_wall",
                PARACETAMOL,
            )
            for segment in SEGMENTS
        )
        wall_mg = self._species_mass_mg(PARACETAMOL, wall_umol)
        record["gut_parent_mg_l"] = wall_mg / patient.gut_volume_l
        record["pathway_scaffold_not_clinically_validated"] = 1.0
        return record
