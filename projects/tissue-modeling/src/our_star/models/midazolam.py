"""Explicit gut--liver midazolam reaction model for v0.4 development.

The first replacement screen intentionally predicted only parent midazolam.
This module is the next, nested mechanism candidate.  It adds three chemical
events without changing the frozen parent priors:

* intestinal and hepatic CYP3A formation of 1-hydroxymidazolam;
* an explicitly tracked remainder pathway for minor parent metabolites; and
* UGT-mediated conversion of 1-hydroxymidazolam to a glucuronide sink.

All physical and accounting states use micromoles.  The glucuronide state is a
cumulative formation sink, not a claimed plasma or urinary concentration.  A
future renal transport module may distribute that species only after an
independent disposition prior or an identifiable development dataset exists.

This is research software.  It is not a dosing, efficacy, toxicity, sedation,
or clinical-trial replacement model.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from ..chemistry.accounting import (
    DualAmountLedger,
    ExternalParticipant,
    ExternalParticipantRole,
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
from ..pbpk import DrugParameters
from .multispecies import MultiSpeciesPBPKModel
from .segmented import FormulationParameters


PARENT = "midazolam"
HYDROXY = "one_hydroxymidazolam"
GLUCURONIDE = "one_hydroxymidazolam_glucuronide"
MINOR_SINK = "midazolam_minor_metabolite_moiety"

SEGMENTS = ("duodenum", "jejunum", "ileum")
LIVER_ZONES = ("liver_zone1", "liver_zone2", "liver_zone3")
REFERENCE_LIVER_VOLUME_L = 1.8
REFERENCE_GUT_VOLUME_L = 1.2


def midazolam_state(compartment_id: str, species_id: str) -> str:
    return f"{compartment_id}::{species_id}"


@dataclass(frozen=True)
class MidazolamMechanismParameters:
    """Evidence-gated parameters for the nested midazolam mechanism.

    Clearances ending in ``unbound_clearance_l_h`` multiply an unbound
    blood-equivalent concentration.  ``ugt_unbound_clearance_l_h`` is the one
    development-estimable parameter in the v0.4 candidate; formation fraction,
    parent clearances, binding, and distribution defaults are fixed before the
    development metabolite curve is scored.
    """

    hydroxy_molecular_weight_g_mol: float
    glucuronide_molecular_weight_g_mol: float
    hydroxy_fraction_unbound: float
    hydroxy_kp_gut: float
    hydroxy_kp_liver: float
    hydroxy_kp_kidney: float
    hydroxy_kp_rest: float
    intestinal_cyp3a_unbound_clearance_l_h: float
    hepatic_cyp3a_unbound_clearance_l_h: float
    major_formation_fraction: float
    ugt_unbound_clearance_l_h: float
    effective_ugt_zone_activity: tuple[float, float, float]
    provenance: str

    def __post_init__(self) -> None:
        positive = (
            "hydroxy_molecular_weight_g_mol",
            "glucuronide_molecular_weight_g_mol",
            "hydroxy_kp_gut",
            "hydroxy_kp_liver",
            "hydroxy_kp_kidney",
            "hydroxy_kp_rest",
            "hepatic_cyp3a_unbound_clearance_l_h",
            "ugt_unbound_clearance_l_h",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0")
            object.__setattr__(self, name, value)
        gut_clearance = float(self.intestinal_cyp3a_unbound_clearance_l_h)
        if not np.isfinite(gut_clearance) or gut_clearance < 0.0:
            raise ValueError(
                "intestinal_cyp3a_unbound_clearance_l_h must be finite and >= 0"
            )
        object.__setattr__(
            self, "intestinal_cyp3a_unbound_clearance_l_h", gut_clearance
        )
        for name in ("hydroxy_fraction_unbound", "major_formation_fraction"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be finite and in (0, 1)")
            object.__setattr__(self, name, value)
        activity = tuple(float(value) for value in self.effective_ugt_zone_activity)
        if (
            len(activity) != 3
            or any(not np.isfinite(value) or value < 0.0 for value in activity)
            or sum(activity) <= 0.0
        ):
            raise ValueError(
                "effective_ugt_zone_activity requires three nonnegative values "
                "with positive sum"
            )
        if not isinstance(self.provenance, str) or not self.provenance.strip():
            raise ValueError("midazolam mechanism provenance is required")
        object.__setattr__(self, "effective_ugt_zone_activity", activity)

    @property
    def normalized_effective_ugt_zone_activity(self) -> tuple[float, float, float]:
        total = float(sum(self.effective_ugt_zone_activity))
        return tuple(value / total for value in self.effective_ugt_zone_activity)


def build_midazolam_reaction_network(
    parent_molecular_weight_g_mol: float,
    parameters: MidazolamMechanismParameters,
) -> OpenReactionNetwork:
    parent_mw = float(parent_molecular_weight_g_mol)
    if not np.isfinite(parent_mw) or parent_mw <= 0.0:
        raise ValueError("parent molecular weight must be finite and > 0")
    hydroxy_delta = parameters.hydroxy_molecular_weight_g_mol - parent_mw
    glucuronyl_delta = (
        parameters.glucuronide_molecular_weight_g_mol
        - parameters.hydroxy_molecular_weight_g_mol
    )
    if hydroxy_delta <= 0.0 or glucuronyl_delta <= 0.0:
        raise ValueError("hydroxylation and glucuronidation must import positive mass")
    species = (
        ChemicalSpecies(PARENT, "midazolam", parent_mw),
        ChemicalSpecies(
            HYDROXY,
            "1-hydroxymidazolam",
            parameters.hydroxy_molecular_weight_g_mol,
        ),
        ChemicalSpecies(
            GLUCURONIDE,
            "1-hydroxymidazolam glucuronide",
            parameters.glucuronide_molecular_weight_g_mol,
        ),
        ChemicalSpecies(
            MINOR_SINK,
            "unresolved minor midazolam metabolite parent moiety",
            parent_mw,
        ),
    )
    oxygen = ExternalParticipant(
        participant_id="cyp3a_net_oxygen_transfer",
        name="net oxygen mass imported during hydroxylation",
        coefficient=-1.0,
        molecular_weight_g_mol=hydroxy_delta,
        role=ExternalParticipantRole.NET_TRANSFER_GROUP,
        provenance=(
            "Molecular-mass transfer bookkeeping for CYP3A hydroxylation; "
            "not a claim that the full CYP cofactor system is explicitly modeled."
        ),
    )
    glucuronyl = ExternalParticipant(
        participant_id="ugt_net_glucuronyl_transfer",
        name="net glucuronyl mass imported during conjugation",
        coefficient=-1.0,
        molecular_weight_g_mol=glucuronyl_delta,
        role=ExternalParticipantRole.NET_TRANSFER_GROUP,
        provenance=(
            "Net conjugating mass bookkeeping for UGT glucuronidation; UDPGA "
            "and coproduct identities remain outside the reduced PBPK state."
        ),
    )
    reactions = (
        OpenReaction(
            reaction_id="midazolam_to_one_hydroxymidazolam",
            terms=(
                StoichiometricTerm(PARENT, -1.0),
                StoichiometricTerm(HYDROXY, 1.0),
            ),
            external_participants=(oxygen,),
            description="CYP3A-mediated 1:1 molar hydroxylation of midazolam.",
        ),
        OpenReaction(
            reaction_id="one_hydroxymidazolam_to_glucuronide",
            terms=(
                StoichiometricTerm(HYDROXY, -1.0),
                StoichiometricTerm(GLUCURONIDE, 1.0),
            ),
            external_participants=(glucuronyl,),
            description="UGT-mediated 1:1 molar conjugation of 1-OH-midazolam.",
        ),
        OpenReaction(
            reaction_id="midazolam_to_minor_parent_moiety_sink",
            terms=(
                StoichiometricTerm(PARENT, -1.0),
                StoichiometricTerm(MINOR_SINK, 1.0),
            ),
            description=(
                "Parent-moiety accounting branch for independently known minor "
                "products that are not assigned a plasma concentration."
            ),
        ),
    )
    return OpenReactionNetwork(species, reactions)


def _build_states(
    network: OpenReactionNetwork,
) -> tuple[AmountStateRegistry, Mapping[str, str]]:
    accounting_species = tuple(
        ChemicalSpecies(item.participant_id, item.name, item.molecular_weight_g_mol)
        for item in network.external_participants.values()
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
    hydroxy_compartments = (
        *(f"{segment}_wall" for segment in SEGMENTS),
        "central",
        *LIVER_ZONES,
        "kidney",
        "rest",
    )
    for compartment in parent_compartments:
        states.append(
            AmountStateSpec(
                midazolam_state(compartment, PARENT),
                PARENT,
                compartment,
                sink=compartment in {"urine", "feces"},
            )
        )
    for compartment in hydroxy_compartments:
        states.append(
            AmountStateSpec(
                midazolam_state(compartment, HYDROXY),
                HYDROXY,
                compartment,
            )
        )
    states.extend(
        (
            AmountStateSpec(
                midazolam_state("conjugation_sink", GLUCURONIDE),
                GLUCURONIDE,
                "conjugation_sink",
                sink=True,
            ),
            AmountStateSpec(
                midazolam_state("minor_metabolic_sink", MINOR_SINK),
                MINOR_SINK,
                "minor_metabolic_sink",
                sink=True,
            ),
        )
    )
    import_states: dict[str, str] = {}
    for participant_id in network.external_participants:
        state_name = f"external_import::{participant_id}"
        import_states[participant_id] = state_name
        states.append(
            AmountStateSpec(
                state_name,
                participant_id,
                "external_import",
                sink=True,
                accounting_only=True,
            )
        )
    return AmountStateRegistry(states, species), MappingProxyType(import_states)


def _amount(
    states: AmountStateRegistry,
    vector: np.ndarray,
    compartment: str,
    species_id: str,
) -> float:
    return max(
        float(vector[states.index(midazolam_state(compartment, species_id))]),
        0.0,
    )


def _reaction_contribution(
    network: OpenReactionNetwork,
    import_states: Mapping[str, str],
    reaction_id: str,
    rate_umol_h: float,
    tracked_targets: Mapping[str, str],
) -> dict[str, float]:
    derivatives = network.derivatives({reaction_id: max(float(rate_umol_h), 0.0)})
    result: dict[str, float] = {}
    for species_id, rate in derivatives.species_umol_h.items():
        state_name = tracked_targets[species_id]
        result[state_name] = result.get(state_name, 0.0) + rate
    for participant_id, rate in derivatives.external_participant_umol_h.items():
        if rate >= 0.0:
            raise RuntimeError("midazolam open reactions unexpectedly export mass")
        state_name = import_states[participant_id]
        result[state_name] = result.get(state_name, 0.0) - rate
    return result


def _merge(target: dict[str, float], source: Mapping[str, float]) -> None:
    for key, value in source.items():
        target[key] = target.get(key, 0.0) + float(value)


class _MidazolamGutModule:
    required_inputs: tuple[FluxKey, ...] = ()

    def __init__(
        self,
        parameters: MidazolamMechanismParameters,
        network: OpenReactionNetwork,
        import_states: Mapping[str, str],
    ) -> None:
        self.parameters = parameters
        self.network = network
        self.import_states = import_states
        self.name = "midazolam_gut_wall_cyp3a"
        self.parent_portal = FluxKey("portal_to_liver", PARENT)
        self.hydroxy_portal = FluxKey("portal_to_liver", HYDROXY)
        self.required_states = tuple(
            (
                midazolam_state("stomach_solid", PARENT),
                midazolam_state("stomach_dissolved", PARENT),
                midazolam_state("central", PARENT),
                midazolam_state("central", HYDROXY),
                midazolam_state("feces", PARENT),
                midazolam_state("minor_metabolic_sink", MINOR_SINK),
            )
            + tuple(
                midazolam_state(f"{segment}_lumen", PARENT)
                for segment in SEGMENTS
            )
            + tuple(
                midazolam_state(f"{segment}_wall", species_id)
                for segment in SEGMENTS
                for species_id in (PARENT, HYDROXY)
            )
            + tuple(import_states.values())
        )

    def evaluate(self, _time_h, vector, context, states, _inputs):
        patient = context.patient
        drug = context.drug
        formulation = context.formulation
        if not isinstance(formulation, FormulationParameters):
            raise TypeError("midazolam gut requires FormulationParameters")
        parent_central = _amount(states, vector, "central", PARENT) / patient.central_volume_l
        hydroxy_central = _amount(states, vector, "central", HYDROXY) / patient.central_volume_l
        solid = _amount(states, vector, "stomach_solid", PARENT)
        dissolved = _amount(states, vector, "stomach_dissolved", PARENT)
        dissolution = formulation.dissolution_rate_h * solid
        emptying = formulation.gastric_emptying_rate_h * dissolved
        derivatives: dict[str, float] = {
            midazolam_state("stomach_solid", PARENT): -dissolution,
            midazolam_state("stomach_dissolved", PARENT): dissolution - emptying,
            midazolam_state("central", PARENT): -patient.portal_flow_l_h * parent_central,
            midazolam_state("central", HYDROXY): -patient.portal_flow_l_h * hydroxy_central,
        }
        incoming_lumen = emptying
        parent_returns = 0.0
        hydroxy_returns = 0.0
        for index, segment in enumerate(SEGMENTS):
            lumen = _amount(states, vector, f"{segment}_lumen", PARENT)
            parent_wall = _amount(states, vector, f"{segment}_wall", PARENT)
            hydroxy_wall = _amount(states, vector, f"{segment}_wall", HYDROXY)
            absorption = (
                drug.fraction_absorbed
                * formulation.segment_absorption_rate_h[index]
                * lumen
            )
            transit = formulation.segment_transit_rate_h[index] * lumen
            wall_volume = patient.gut_volume_l * formulation.wall_volume_fraction[index]
            flow = patient.portal_flow_l_h * formulation.portal_flow_fraction[index]
            parent_equivalent = parent_wall / wall_volume / drug.kp_gut
            hydroxy_equivalent = (
                hydroxy_wall / wall_volume / self.parameters.hydroxy_kp_gut
            )
            parent_return = flow * parent_equivalent
            hydroxy_return = flow * hydroxy_equivalent
            total_cyp3a = (
                self.parameters.intestinal_cyp3a_unbound_clearance_l_h
                * formulation.portal_flow_fraction[index]
                * drug.fraction_unbound_plasma
                * parent_equivalent
                * context.mechanism_factor("intestinal_cyp3a")
                * patient.enzyme_activity_factor
                * patient.gut_volume_l
                / REFERENCE_GUT_VOLUME_L
            )
            major_rate = total_cyp3a * self.parameters.major_formation_fraction
            minor_rate = total_cyp3a - major_rate
            parent_wall_state = midazolam_state(f"{segment}_wall", PARENT)
            hydroxy_wall_state = midazolam_state(f"{segment}_wall", HYDROXY)
            derivatives[midazolam_state(f"{segment}_lumen", PARENT)] = (
                incoming_lumen - absorption - transit
            )
            derivatives[parent_wall_state] = (
                absorption + flow * parent_central - parent_return
            )
            derivatives[hydroxy_wall_state] = (
                flow * hydroxy_central - hydroxy_return
            )
            _merge(
                derivatives,
                _reaction_contribution(
                    self.network,
                    self.import_states,
                    "midazolam_to_one_hydroxymidazolam",
                    major_rate,
                    {PARENT: parent_wall_state, HYDROXY: hydroxy_wall_state},
                ),
            )
            _merge(
                derivatives,
                _reaction_contribution(
                    self.network,
                    self.import_states,
                    "midazolam_to_minor_parent_moiety_sink",
                    minor_rate,
                    {
                        PARENT: parent_wall_state,
                        MINOR_SINK: midazolam_state(
                            "minor_metabolic_sink", MINOR_SINK
                        ),
                    },
                ),
            )
            incoming_lumen = transit
            parent_returns += parent_return
            hydroxy_returns += hydroxy_return
        derivatives[midazolam_state("feces", PARENT)] = incoming_lumen
        return AmountModuleResult(
            derivatives,
            {
                self.parent_portal: parent_returns,
                self.hydroxy_portal: hydroxy_returns,
            },
        )


class _MidazolamLiverModule:
    def __init__(
        self,
        parameters: MidazolamMechanismParameters,
        network: OpenReactionNetwork,
        import_states: Mapping[str, str],
    ) -> None:
        self.parameters = parameters
        self.network = network
        self.import_states = import_states
        self.name = "midazolam_three_zone_cyp3a_ugt"
        self.parent_portal = FluxKey("portal_to_liver", PARENT)
        self.hydroxy_portal = FluxKey("portal_to_liver", HYDROXY)
        self.required_inputs = (self.parent_portal, self.hydroxy_portal)
        self.required_states = tuple(
            (
                midazolam_state("central", PARENT),
                midazolam_state("central", HYDROXY),
                midazolam_state("conjugation_sink", GLUCURONIDE),
                midazolam_state("minor_metabolic_sink", MINOR_SINK),
            )
            + tuple(
                midazolam_state(zone, species_id)
                for zone in LIVER_ZONES
                for species_id in (PARENT, HYDROXY)
            )
            + tuple(import_states.values())
        )

    def evaluate(self, _time_h, vector, context, states, inputs):
        patient = context.patient
        drug = context.drug
        zone_volume = patient.liver_volume_l / 3.0
        q_liver = patient.liver_flow_l_h
        parent_central = _amount(states, vector, "central", PARENT) / patient.central_volume_l
        hydroxy_central = _amount(states, vector, "central", HYDROXY) / patient.central_volume_l
        parent_zones = tuple(
            _amount(states, vector, zone, PARENT) / zone_volume / drug.kp_liver
            for zone in LIVER_ZONES
        )
        hydroxy_zones = tuple(
            _amount(states, vector, zone, HYDROXY)
            / zone_volume
            / self.parameters.hydroxy_kp_liver
            for zone in LIVER_ZONES
        )
        parent_activity = drug.normalized_zone_activity
        ugt_activity = self.parameters.normalized_effective_ugt_zone_activity
        liver_scale = (
            patient.enzyme_activity_factor
            * patient.liver_function_fraction
            * patient.liver_volume_l
            / REFERENCE_LIVER_VOLUME_L
        )
        cyp_rates = tuple(
            self.parameters.hepatic_cyp3a_unbound_clearance_l_h
            * parent_activity[index]
            * drug.fraction_unbound_plasma
            * parent_zones[index]
            * liver_scale
            * context.mechanism_factor("hepatic_cyp3a")
            for index in range(3)
        )
        ugt_rates = tuple(
            self.parameters.ugt_unbound_clearance_l_h
            * ugt_activity[index]
            * self.parameters.hydroxy_fraction_unbound
            * hydroxy_zones[index]
            * liver_scale
            * context.mechanism_factor("effective_ugt")
            for index in range(3)
        )
        derivatives: dict[str, float] = {
            midazolam_state("central", PARENT): (
                q_liver * parent_zones[2]
                - patient.hepatic_artery_flow_l_h * parent_central
            ),
            midazolam_state("central", HYDROXY): (
                q_liver * hydroxy_zones[2]
                - patient.hepatic_artery_flow_l_h * hydroxy_central
            ),
        }
        for index, zone in enumerate(LIVER_ZONES):
            parent_in = (
                inputs[self.parent_portal]
                + patient.hepatic_artery_flow_l_h * parent_central
                if index == 0
                else q_liver * parent_zones[index - 1]
            )
            hydroxy_in = (
                inputs[self.hydroxy_portal]
                + patient.hepatic_artery_flow_l_h * hydroxy_central
                if index == 0
                else q_liver * hydroxy_zones[index - 1]
            )
            parent_state = midazolam_state(zone, PARENT)
            hydroxy_state = midazolam_state(zone, HYDROXY)
            derivatives[parent_state] = parent_in - q_liver * parent_zones[index]
            derivatives[hydroxy_state] = (
                hydroxy_in - q_liver * hydroxy_zones[index]
            )
            major_rate = cyp_rates[index] * self.parameters.major_formation_fraction
            minor_rate = cyp_rates[index] - major_rate
            _merge(
                derivatives,
                _reaction_contribution(
                    self.network,
                    self.import_states,
                    "midazolam_to_one_hydroxymidazolam",
                    major_rate,
                    {PARENT: parent_state, HYDROXY: hydroxy_state},
                ),
            )
            _merge(
                derivatives,
                _reaction_contribution(
                    self.network,
                    self.import_states,
                    "midazolam_to_minor_parent_moiety_sink",
                    minor_rate,
                    {
                        PARENT: parent_state,
                        MINOR_SINK: midazolam_state(
                            "minor_metabolic_sink", MINOR_SINK
                        ),
                    },
                ),
            )
            _merge(
                derivatives,
                _reaction_contribution(
                    self.network,
                    self.import_states,
                    "one_hydroxymidazolam_to_glucuronide",
                    ugt_rates[index],
                    {
                        HYDROXY: hydroxy_state,
                        GLUCURONIDE: midazolam_state(
                            "conjugation_sink", GLUCURONIDE
                        ),
                    },
                ),
            )
        return AmountModuleResult(derivatives)


class _MidazolamKidneyModule:
    required_inputs: tuple[FluxKey, ...] = ()

    def __init__(self, parameters: MidazolamMechanismParameters) -> None:
        self.parameters = parameters
        self.name = "midazolam_kidney_distribution"
        self.required_states = tuple(
            midazolam_state(compartment, species_id)
            for species_id in (PARENT, HYDROXY)
            for compartment in ("central", "kidney")
        ) + (midazolam_state("urine", PARENT),)

    def evaluate(self, _time_h, vector, context, states, _inputs):
        patient = context.patient
        drug = context.drug
        parent_central = _amount(states, vector, "central", PARENT) / patient.central_volume_l
        parent_kidney = (
            _amount(states, vector, "kidney", PARENT)
            / patient.kidney_volume_l
            / drug.kp_kidney
        )
        hydroxy_central = _amount(states, vector, "central", HYDROXY) / patient.central_volume_l
        hydroxy_kidney = (
            _amount(states, vector, "kidney", HYDROXY)
            / patient.kidney_volume_l
            / self.parameters.hydroxy_kp_kidney
        )
        parent_exchange = patient.renal_flow_l_h * (parent_central - parent_kidney)
        hydroxy_exchange = patient.renal_flow_l_h * (
            hydroxy_central - hydroxy_kidney
        )
        renal_scale = (
            patient.renal_function_fraction
            * (patient.body_weight_kg / 70.0) ** 0.75
        )
        unchanged = drug.renal_clearance_l_h * renal_scale * parent_kidney
        return AmountModuleResult(
            {
                midazolam_state("central", PARENT): -parent_exchange,
                midazolam_state("kidney", PARENT): parent_exchange - unchanged,
                midazolam_state("urine", PARENT): unchanged,
                midazolam_state("central", HYDROXY): -hydroxy_exchange,
                midazolam_state("kidney", HYDROXY): hydroxy_exchange,
            }
        )


class _MidazolamRestModule:
    required_inputs: tuple[FluxKey, ...] = ()

    def __init__(self, parameters: MidazolamMechanismParameters) -> None:
        self.parameters = parameters
        self.name = "midazolam_rest_distribution"
        self.required_states = tuple(
            midazolam_state(compartment, species_id)
            for species_id in (PARENT, HYDROXY)
            for compartment in ("central", "rest")
        )

    def evaluate(self, _time_h, vector, context, states, _inputs):
        patient = context.patient
        drug = context.drug
        parent_central = _amount(states, vector, "central", PARENT) / patient.central_volume_l
        parent_rest = (
            _amount(states, vector, "rest", PARENT)
            / patient.rest_volume_l
            / drug.kp_rest
        )
        hydroxy_central = _amount(states, vector, "central", HYDROXY) / patient.central_volume_l
        hydroxy_rest = (
            _amount(states, vector, "rest", HYDROXY)
            / patient.rest_volume_l
            / self.parameters.hydroxy_kp_rest
        )
        parent_exchange = patient.rest_flow_l_h * (parent_central - parent_rest)
        hydroxy_exchange = patient.rest_flow_l_h * (
            hydroxy_central - hydroxy_rest
        )
        return AmountModuleResult(
            {
                midazolam_state("central", PARENT): -parent_exchange,
                midazolam_state("rest", PARENT): parent_exchange,
                midazolam_state("central", HYDROXY): -hydroxy_exchange,
                midazolam_state("rest", HYDROXY): hydroxy_exchange,
            }
        )


class MidazolamMechanisticPBPKModel(MultiSpeciesPBPKModel):
    """Nested explicit midazolam/1-OH mechanism candidate."""

    def __init__(
        self,
        drug: DrugParameters,
        formulation: FormulationParameters,
        mechanisms: MidazolamMechanismParameters,
        *,
        mechanism_factors: Mapping[str, float] | None = None,
    ) -> None:
        if abs(drug.molecular_weight_g_mol - 325.77) > 0.5:
            raise ValueError("midazolam model requires the midazolam parent species")
        network = build_midazolam_reaction_network(
            drug.molecular_weight_g_mol, mechanisms
        )
        states, import_states = _build_states(network)
        ledger = DualAmountLedger(
            states,
            tuple(
                SpeciesMoiety(species_id, {PARENT: 1.0})
                for species_id in network.species.ids
            ),
            conserved_moiety_id=PARENT,
            reference_species_id=PARENT,
            external_participants=network.external_participants,
            external_import_states=import_states,
            external_export_states={},
        )
        assembler = AmountSystemAssembler(
            states,
            (
                _MidazolamGutModule(mechanisms, network, import_states),
                _MidazolamLiverModule(mechanisms, network, import_states),
                _MidazolamKidneyModule(mechanisms),
                _MidazolamRestModule(mechanisms),
            ),
        )
        self.mechanisms = mechanisms
        super().__init__(
            model_id="our_star.midazolam_gut_liver_reaction_pbpk.v0.4.dev",
            drug=drug,
            formulation=formulation,
            states=states,
            assembler=assembler,
            reaction_network=network,
            ledger=ledger,
            parent_species_id=PARENT,
            dose_targets={
                ("oral", PARENT): midazolam_state("stomach_solid", PARENT),
                ("iv_bolus", PARENT): midazolam_state("central", PARENT),
            },
            central_states={
                PARENT: midazolam_state("central", PARENT),
                HYDROXY: midazolam_state("central", HYDROXY),
            },
            urine_states={PARENT: midazolam_state("urine", PARENT)},
            feces_parent_state=midazolam_state("feces", PARENT),
            parent_liver_states=tuple(
                midazolam_state(zone, PARENT) for zone in LIVER_ZONES
            ),
            mechanism_factors=mechanism_factors,
        )

    def apply_species_doses(
        self, state: np.ndarray, events: Sequence[SpeciesDoseEvent]
    ) -> np.ndarray:
        values = self.state_registry.validate_vector(state).copy()
        for event in events:
            if event.species_id != PARENT:
                raise ValueError("midazolam mechanism model accepts parent doses only")
            amount_umol = float(
                self.state_registry.species[PARENT].convert_amount(
                    event.amount, event.unit, AmountUnit.UMOL
                )
            )
            if event.route == "iv_bolus":
                values[
                    self.state_registry.index(midazolam_state("central", PARENT))
                ] += amount_umol
                continue
            dissolved = amount_umol * self.formulation.initial_dissolved_fraction
            values[
                self.state_registry.index(
                    midazolam_state("stomach_dissolved", PARENT)
                )
            ] += dissolved
            values[
                self.state_registry.index(midazolam_state("stomach_solid", PARENT))
            ] += amount_umol - dissolved
        return values

    def concentration_record(self, state, patient, drug):
        record = super().concentration_record(state, patient, drug)
        values = self.state_registry.validate_vector(state)
        for species_id, compartment in (
            (GLUCURONIDE, "conjugation_sink"),
            (MINOR_SINK, "minor_metabolic_sink"),
        ):
            amount_umol = _amount(states=self.state_registry, vector=values, compartment=compartment, species_id=species_id)
            record[f"total_{species_id}_umol"] = amount_umol
            record[f"total_{species_id}_mg"] = self._species_mass_mg(
                species_id, amount_umol
            )
        record["plasma_one_hydroxymidazolam_mg_l"] = record[
            f"plasma_{HYDROXY}_mg_l"
        ]
        record["glucuronide_is_formation_sink_not_observation"] = 1.0
        record["research_only_not_clinically_validated"] = 1.0
        return record


__all__ = [
    "GLUCURONIDE",
    "HYDROXY",
    "MINOR_SINK",
    "MidazolamMechanismParameters",
    "MidazolamMechanisticPBPKModel",
    "build_midazolam_reaction_network",
    "midazolam_state",
]
