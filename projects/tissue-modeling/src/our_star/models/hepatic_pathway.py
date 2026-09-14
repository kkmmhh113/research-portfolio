"""Generic molar parent-to-product hepatic pathway PBPK model.

This module is the common development-screening model used for replacement
drug candidates in v0.4 amendment 01.  It deliberately implements only the
mechanisms shared by the candidates: segmented oral input, perfusion through
three serial liver zones, one or more linear hepatic formation pathways,
parent renal loss, and systemic/renal disposition of explicitly measured
products.  Drug-specific mechanisms such as enteric release, gut-wall
metabolism, stereoselectivity, transporter saturation, or auto-inhibition are
not silently approximated here; candidate configuration must list them as
structural limitations.

All dynamic states are micromoles.  A parent-moiety ledger and an augmented
molecular-mass ledger are both closed, including the net untracked mass
imported or exported by oxidation, reduction, hydrolysis, or dealkylation.
This is research software and is not a clinical dosing model.
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
)
from ..chemistry.kinetics import (
    LinearClearanceKinetics,
    ScaledKineticPathway,
    evaluate_competing_pathways,
)
from ..chemistry.reactions import StoichiometricTerm
from ..core.amount_state import (
    AmountStateRegistry,
    AmountStateSpec,
    FluxKey,
    SpeciesDoseEvent,
)
from ..core.amount_system import AmountModelContext, AmountModuleResult, AmountSystemAssembler
from ..core.species import AmountUnit, ChemicalSpecies, SpeciesRegistry
from ..pbpk import DrugParameters
from .multispecies import MultiSpeciesPBPKModel
from .segmented import FormulationParameters


SEGMENTS = ("duodenum", "jejunum", "ileum")
LIVER_ZONES = ("liver_zone1", "liver_zone2", "liver_zone3")
REFERENCE_LIVER_VOLUME_L = 1.8


def amount_state_name(compartment_id: str, species_id: str) -> str:
    return f"{compartment_id}::{species_id}"


def _external_state_name(direction: str, participant_id: str) -> str:
    return f"external_{direction}::{participant_id}"


@dataclass(frozen=True)
class HepaticProductPathwaySpec:
    """One 1:1 molar parent-origin pathway.

    ``systemic=True`` creates liver-zone, central, and urine states.  A false
    value creates only a cumulative metabolic sink and is suitable for the
    explicitly declared remainder of total hepatic clearance.
    """

    species_id: str
    name: str
    molecular_weight_g_mol: float
    reaction_id: str
    mechanism_id: str
    formation_clearance_l_h: float
    zone_activity: tuple[float, float, float]
    systemic: bool
    liver_partition_coefficient: float
    renal_clearance_l_h: float
    parameter_provenance: str

    def __post_init__(self) -> None:
        for label in ("species_id", "name", "reaction_id", "mechanism_id"):
            value = getattr(self, label)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{label} is required")
        numeric_positive = {
            "molecular_weight_g_mol": self.molecular_weight_g_mol,
            "liver_partition_coefficient": self.liver_partition_coefficient,
        }
        for label, raw in numeric_positive.items():
            value = float(raw)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{label} must be finite and > 0")
            object.__setattr__(self, label, value)
        for label in ("formation_clearance_l_h", "renal_clearance_l_h"):
            value = float(getattr(self, label))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{label} must be finite and >= 0")
            object.__setattr__(self, label, value)
        zones = tuple(float(value) for value in self.zone_activity)
        if (
            len(zones) != 3
            or any(not np.isfinite(value) or value < 0.0 for value in zones)
            or sum(zones) <= 0.0
        ):
            raise ValueError("zone_activity requires three nonnegative values with positive sum")
        if type(self.systemic) is not bool:
            raise TypeError("systemic must be a boolean")
        if not self.parameter_provenance.strip():
            raise ValueError("parameter_provenance is required")
        object.__setattr__(self, "zone_activity", zones)

    @property
    def normalized_zone_activity(self) -> tuple[float, float, float]:
        total = float(sum(self.zone_activity))
        return tuple(value / total for value in self.zone_activity)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "HepaticProductPathwaySpec":
        values = dict(raw)
        expected = {
            "species_id",
            "name",
            "molecular_weight_g_mol",
            "reaction_id",
            "mechanism_id",
            "formation_clearance_l_h",
            "zone_activity",
            "systemic",
            "liver_partition_coefficient",
            "renal_clearance_l_h",
            "parameter_provenance",
        }
        if set(values) != expected:
            raise ValueError(
                "hepatic product pathway keys differ; "
                f"missing={sorted(expected - set(values))}, "
                f"extra={sorted(set(values) - expected)}"
            )
        if not isinstance(values["zone_activity"], list):
            raise TypeError("zone_activity must be a YAML/JSON list")
        values["zone_activity"] = tuple(values["zone_activity"])
        return cls(**values)


@dataclass(frozen=True)
class HepaticPathwayParameters:
    """Strict candidate-specific pathway configuration."""

    parent_species_id: str
    products: tuple[HepaticProductPathwaySpec, ...]
    evidence_role: str
    structural_limitations: tuple[str, ...]
    parameter_provenance: str

    def __post_init__(self) -> None:
        if not isinstance(self.parent_species_id, str) or not self.parent_species_id.strip():
            raise ValueError("parent_species_id is required")
        products = tuple(self.products)
        if not products or any(not isinstance(item, HepaticProductPathwaySpec) for item in products):
            raise ValueError("at least one typed product pathway is required")
        for attribute in ("species_id", "reaction_id", "mechanism_id"):
            values = tuple(getattr(item, attribute) for item in products)
            if len(values) != len(set(values)):
                raise ValueError(f"product {attribute} values must be unique")
        if self.parent_species_id in {item.species_id for item in products}:
            raise ValueError("product species cannot reuse parent_species_id")
        # A parent-only zero-tuning lane may intentionally represent total
        # metabolism as one non-systemic parent-moiety sink.  It must still be
        # a named, mass-balanced reaction, but it must not fabricate a plasma
        # concentration for an independently unparameterized metabolite.
        if self.evidence_role not in {
            "independent_zero_tuning_priors",
            "mixed_independent_and_bounded_engineering_priors",
        }:
            raise ValueError("unsupported evidence_role")
        limitations = tuple(self.structural_limitations)
        if not limitations or any(not isinstance(item, str) or not item.strip() for item in limitations):
            raise ValueError("structural_limitations must be a nonempty string list")
        if not self.parameter_provenance.strip():
            raise ValueError("parameter_provenance is required")
        object.__setattr__(self, "products", products)
        object.__setattr__(self, "structural_limitations", limitations)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "HepaticPathwayParameters":
        expected = {
            "parent_species_id",
            "products",
            "evidence_role",
            "structural_limitations",
            "parameter_provenance",
        }
        if set(raw) != expected:
            raise ValueError(
                "hepatic pathway parameter keys differ; "
                f"missing={sorted(expected - set(raw))}, extra={sorted(set(raw) - expected)}"
            )
        if not isinstance(raw["products"], list) or not isinstance(raw["structural_limitations"], list):
            raise TypeError("products and structural_limitations must be lists")
        return cls(
            parent_species_id=str(raw["parent_species_id"]),
            products=tuple(
                HepaticProductPathwaySpec.from_mapping(item)
                for item in raw["products"]
            ),
            evidence_role=str(raw["evidence_role"]),
            structural_limitations=tuple(str(item) for item in raw["structural_limitations"]),
            parameter_provenance=str(raw["parameter_provenance"]),
        )


def _reaction_external_participants(
    *, parent_mw: float, product: HepaticProductPathwaySpec
) -> tuple[ExternalParticipant, ...]:
    delta = product.molecular_weight_g_mol - parent_mw
    if abs(delta) <= 1.0e-12:
        return ()
    participant_id = f"net_mass_equivalent_{product.reaction_id}"
    coefficient = -1.0 if delta > 0.0 else 1.0
    role = (
        ExternalParticipantRole.NET_TRANSFER_GROUP
        if coefficient < 0.0
        else ExternalParticipantRole.COPRODUCT
    )
    return (
        ExternalParticipant(
            participant_id=participant_id,
            name=f"untracked net molecular mass equivalent for {product.reaction_id}",
            coefficient=coefficient,
            molecular_weight_g_mol=abs(delta),
            role=role,
            provenance=(
                "Molecular-weight difference bookkeeping only; the biochemical "
                "cofactor/coproduct identity must be supplied by a drug-specific "
                "reaction upgrade before mechanistic clinical interpretation."
            ),
        ),
    )


def build_hepatic_pathway_network(
    drug: DrugParameters,
    parameters: HepaticPathwayParameters,
) -> OpenReactionNetwork:
    parent = ChemicalSpecies(
        parameters.parent_species_id,
        drug.name,
        drug.molecular_weight_g_mol,
        structure=drug.smiles,
    )
    products = tuple(
        ChemicalSpecies(item.species_id, item.name, item.molecular_weight_g_mol)
        for item in parameters.products
    )
    reactions = tuple(
        OpenReaction(
            reaction_id=item.reaction_id,
            terms=(
                StoichiometricTerm(parameters.parent_species_id, -1.0),
                StoichiometricTerm(item.species_id, 1.0),
            ),
            external_participants=_reaction_external_participants(
                parent_mw=drug.molecular_weight_g_mol, product=item
            ),
            description=(
                "One parent drug moiety produces one configured product moiety; "
                "untracked molecular mass is represented explicitly."
            ),
        )
        for item in parameters.products
    )
    return OpenReactionNetwork((parent, *products), reactions)


def build_hepatic_pathway_states(
    network: OpenReactionNetwork,
    parameters: HepaticPathwayParameters,
) -> tuple[AmountStateRegistry, Mapping[str, str], Mapping[str, str]]:
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
    for compartment in parent_compartments:
        states.append(
            AmountStateSpec(
                amount_state_name(compartment, parameters.parent_species_id),
                parameters.parent_species_id,
                compartment,
                sink=compartment in {"urine", "feces"},
            )
        )
    for product in parameters.products:
        compartments = ("central", *LIVER_ZONES, "urine") if product.systemic else ("metabolic_sink",)
        for compartment in compartments:
            states.append(
                AmountStateSpec(
                    amount_state_name(compartment, product.species_id),
                    product.species_id,
                    compartment,
                    sink=compartment in {"urine", "metabolic_sink"},
                )
            )
    import_states: dict[str, str] = {}
    export_states: dict[str, str] = {}
    for participant_id in network.external_participants:
        coefficients = tuple(
            item.coefficient
            for reaction in network.reactions
            for item in reaction.external_participants
            if item.participant_id == participant_id
        )
        if any(value < 0.0 for value in coefficients):
            state_name = _external_state_name("import", participant_id)
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
        if any(value > 0.0 for value in coefficients):
            state_name = _external_state_name("export", participant_id)
            export_states[participant_id] = state_name
            states.append(
                AmountStateSpec(
                    state_name,
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


class _GenericGutModule:
    required_inputs: tuple[FluxKey, ...] = ()

    def __init__(self, parent_species_id: str) -> None:
        self.parent = parent_species_id
        self.name = f"{parent_species_id}_segmented_gut_umol"
        self.portal_flux = FluxKey("portal_to_liver", parent_species_id)
        self.required_states = tuple(
            amount_state_name(compartment, parent_species_id)
            for compartment in (
                "stomach_solid",
                "stomach_dissolved",
                *(f"{segment}_lumen" for segment in SEGMENTS),
                *(f"{segment}_wall" for segment in SEGMENTS),
                "central",
                "feces",
            )
        )

    def evaluate(self, _time_h, amounts_umol, context, states, _inputs_umol_h):
        patient = context.patient
        drug = context.drug
        formulation = context.formulation
        if not isinstance(formulation, FormulationParameters):
            raise TypeError("generic hepatic gut requires FormulationParameters")
        central = _amount(states, amounts_umol, "central", self.parent)
        central_concentration = central / patient.central_volume_l
        solid = _amount(states, amounts_umol, "stomach_solid", self.parent)
        dissolved = _amount(states, amounts_umol, "stomach_dissolved", self.parent)
        dissolution = formulation.dissolution_rate_h * solid
        gastric_emptying = formulation.gastric_emptying_rate_h * dissolved
        derivatives: dict[str, float] = {
            amount_state_name("stomach_solid", self.parent): -dissolution,
            amount_state_name("stomach_dissolved", self.parent): dissolution - gastric_emptying,
            amount_state_name("central", self.parent): -patient.portal_flow_l_h * central_concentration,
        }
        incoming_lumen = gastric_emptying
        venous_returns: list[float] = []
        for index, segment in enumerate(SEGMENTS):
            lumen = _amount(states, amounts_umol, f"{segment}_lumen", self.parent)
            wall = _amount(states, amounts_umol, f"{segment}_wall", self.parent)
            absorption = drug.fraction_absorbed * formulation.segment_absorption_rate_h[index] * lumen
            transit = formulation.segment_transit_rate_h[index] * lumen
            wall_volume = patient.gut_volume_l * formulation.wall_volume_fraction[index]
            wall_equivalent = wall / wall_volume / drug.kp_gut
            segment_flow = patient.portal_flow_l_h * formulation.portal_flow_fraction[index]
            venous_return = segment_flow * wall_equivalent
            derivatives[amount_state_name(f"{segment}_lumen", self.parent)] = incoming_lumen - absorption - transit
            derivatives[amount_state_name(f"{segment}_wall", self.parent)] = absorption + segment_flow * central_concentration - venous_return
            incoming_lumen = transit
            venous_returns.append(venous_return)
        derivatives[amount_state_name("feces", self.parent)] = incoming_lumen
        return AmountModuleResult(derivatives, {self.portal_flux: float(sum(venous_returns))})


class _GenericLiverModule:
    def __init__(
        self,
        parameters: HepaticPathwayParameters,
        network: OpenReactionNetwork,
        import_states: Mapping[str, str],
        export_states: Mapping[str, str],
    ) -> None:
        self.parameters = parameters
        self.parent = parameters.parent_species_id
        self.network = network
        self.import_states = dict(import_states)
        self.export_states = dict(export_states)
        self.name = f"{self.parent}_hepatic_pathways_umol"
        self.portal_flux = FluxKey("portal_to_liver", self.parent)
        self.required_inputs = (self.portal_flux,)
        physical = [
            amount_state_name(compartment, self.parent)
            for compartment in ("central", *LIVER_ZONES)
        ]
        for product in parameters.products:
            if product.systemic:
                physical.extend(
                    amount_state_name(compartment, product.species_id)
                    for compartment in ("central", *LIVER_ZONES)
                )
            else:
                physical.append(amount_state_name("metabolic_sink", product.species_id))
        self.required_states = (*physical, *import_states.values(), *export_states.values())
        self._product_by_species = {item.species_id: item for item in parameters.products}
        self._pathways_by_zone = tuple(
            tuple(
                ScaledKineticPathway(
                    product.reaction_id,
                    product.mechanism_id,
                    LinearClearanceKinetics(product.formation_clearance_l_h),
                    product.normalized_zone_activity[index],
                )
                for product in parameters.products
            )
            for index in range(3)
        )

    def evaluate(self, _time_h, amounts_umol, context, states, inputs_umol_h):
        patient = context.patient
        drug = context.drug
        zone_volume = patient.liver_volume_l / 3.0
        q_liver = patient.liver_flow_l_h
        central_concentration = _amount(states, amounts_umol, "central", self.parent) / patient.central_volume_l
        zone_equivalent = tuple(
            _amount(states, amounts_umol, zone, self.parent) / zone_volume / drug.kp_liver
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
                drug.fraction_unbound_plasma * zone_equivalent[index],
                self._pathways_by_zone[index],
                mechanism_factors=context.mechanism_factors,
                shared_scale=shared_scale,
            )
            for index in range(3)
        )
        derivatives: dict[str, float] = {
            amount_state_name("liver_zone1", self.parent): inputs_umol_h[self.portal_flux] + patient.hepatic_artery_flow_l_h * central_concentration - q_liver * zone_equivalent[0],
            amount_state_name("liver_zone2", self.parent): q_liver * zone_equivalent[0] - q_liver * zone_equivalent[1],
            amount_state_name("liver_zone3", self.parent): q_liver * zone_equivalent[1] - q_liver * zone_equivalent[2],
            amount_state_name("central", self.parent): q_liver * zone_equivalent[2] - patient.hepatic_artery_flow_l_h * central_concentration,
        }
        for rates, zone in zip(rates_by_zone, LIVER_ZONES, strict=True):
            reaction_derivatives = self.network.derivatives(rates)
            for species_id, rate in reaction_derivatives.species_umol_h.items():
                product = self._product_by_species.get(species_id)
                if product is not None and not product.systemic:
                    state_name = amount_state_name("metabolic_sink", species_id)
                else:
                    state_name = amount_state_name(zone, species_id)
                derivatives[state_name] = derivatives.get(state_name, 0.0) + rate
            for participant_id, signed_rate in reaction_derivatives.external_participant_umol_h.items():
                if signed_rate < 0.0:
                    state_name = self.import_states[participant_id]
                    derivatives[state_name] = derivatives.get(state_name, 0.0) - signed_rate
                elif signed_rate > 0.0:
                    state_name = self.export_states[participant_id]
                    derivatives[state_name] = derivatives.get(state_name, 0.0) + signed_rate
        for product in self.parameters.products:
            if not product.systemic:
                continue
            product_central = _amount(states, amounts_umol, "central", product.species_id) / patient.central_volume_l
            product_zones = tuple(
                _amount(states, amounts_umol, zone, product.species_id)
                / zone_volume
                / product.liver_partition_coefficient
                for zone in LIVER_ZONES
            )
            derivatives[amount_state_name("central", product.species_id)] = q_liver * product_zones[2] - q_liver * product_central
            for index, zone in enumerate(LIVER_ZONES):
                incoming = q_liver * product_central if index == 0 else q_liver * product_zones[index - 1]
                state_name = amount_state_name(zone, product.species_id)
                derivatives[state_name] = derivatives.get(state_name, 0.0) + incoming - q_liver * product_zones[index]
        return AmountModuleResult(derivatives)


class _GenericKidneyModule:
    required_inputs: tuple[FluxKey, ...] = ()

    def __init__(self, parameters: HepaticPathwayParameters) -> None:
        self.parameters = parameters
        self.parent = parameters.parent_species_id
        self.name = f"{self.parent}_kidney_umol"
        self.required_states = (
            amount_state_name("central", self.parent),
            amount_state_name("kidney", self.parent),
            amount_state_name("urine", self.parent),
            *tuple(
                amount_state_name(compartment, product.species_id)
                for product in parameters.products
                if product.systemic
                for compartment in ("central", "urine")
            ),
        )

    def evaluate(self, _time_h, amounts_umol, context, states, _inputs_umol_h):
        patient = context.patient
        drug = context.drug
        central = _amount(states, amounts_umol, "central", self.parent) / patient.central_volume_l
        kidney_equivalent = _amount(states, amounts_umol, "kidney", self.parent) / patient.kidney_volume_l / drug.kp_kidney
        renal_scale = patient.renal_function_fraction * (patient.body_weight_kg / 70.0) ** 0.75
        parent_elimination = drug.renal_clearance_l_h * renal_scale * kidney_equivalent
        exchange = patient.renal_flow_l_h * (central - kidney_equivalent)
        derivatives: dict[str, float] = {
            amount_state_name("kidney", self.parent): exchange - parent_elimination,
            amount_state_name("central", self.parent): -exchange,
            amount_state_name("urine", self.parent): parent_elimination,
        }
        for product in self.parameters.products:
            if not product.systemic:
                continue
            product_central = _amount(states, amounts_umol, "central", product.species_id) / patient.central_volume_l
            elimination = product.renal_clearance_l_h * renal_scale * product_central
            derivatives[amount_state_name("central", product.species_id)] = -elimination
            derivatives[amount_state_name("urine", product.species_id)] = elimination
        return AmountModuleResult(derivatives)


class _GenericRestModule:
    required_inputs: tuple[FluxKey, ...] = ()

    def __init__(self, parent_species_id: str) -> None:
        self.parent = parent_species_id
        self.name = f"{parent_species_id}_rest_umol"
        self.required_states = (
            amount_state_name("central", self.parent),
            amount_state_name("rest", self.parent),
        )

    def evaluate(self, _time_h, amounts_umol, context, states, _inputs_umol_h):
        patient = context.patient
        drug = context.drug
        central = _amount(states, amounts_umol, "central", self.parent) / patient.central_volume_l
        rest_equivalent = _amount(states, amounts_umol, "rest", self.parent) / patient.rest_volume_l / drug.kp_rest
        exchange = patient.rest_flow_l_h * (central - rest_equivalent)
        return AmountModuleResult(
            {
                amount_state_name("rest", self.parent): exchange,
                amount_state_name("central", self.parent): -exchange,
            }
        )


class HepaticPathwayPBPKModel(MultiSpeciesPBPKModel):
    """Common zero-tuning development comparator for replacement candidates."""

    def __init__(
        self,
        drug: DrugParameters,
        formulation: FormulationParameters,
        pathway_parameters: HepaticPathwayParameters,
        *,
        mechanism_factors: Mapping[str, float] | None = None,
    ) -> None:
        network = build_hepatic_pathway_network(drug, pathway_parameters)
        states, import_states, export_states = build_hepatic_pathway_states(
            network, pathway_parameters
        )
        moieties = tuple(
            SpeciesMoiety(species_id, {pathway_parameters.parent_species_id: 1.0})
            for species_id in network.species.ids
        )
        ledger = DualAmountLedger(
            states,
            moieties,
            conserved_moiety_id=pathway_parameters.parent_species_id,
            reference_species_id=pathway_parameters.parent_species_id,
            external_participants=network.external_participants,
            external_import_states=import_states,
            external_export_states=export_states,
        )
        gut = _GenericGutModule(pathway_parameters.parent_species_id)
        assembler = AmountSystemAssembler(
            states,
            (
                gut,
                _GenericLiverModule(pathway_parameters, network, import_states, export_states),
                _GenericKidneyModule(pathway_parameters),
                _GenericRestModule(pathway_parameters.parent_species_id),
            ),
        )
        systemic_products = tuple(item for item in pathway_parameters.products if item.systemic)
        self.pathway_parameters = pathway_parameters
        self.external_import_states = import_states
        self.external_export_states = export_states
        super().__init__(
            model_id=f"our_star.hepatic_pathway_pbpk.v0.4.dev.{pathway_parameters.parent_species_id}",
            drug=drug,
            formulation=formulation,
            states=states,
            assembler=assembler,
            reaction_network=network,
            ledger=ledger,
            parent_species_id=pathway_parameters.parent_species_id,
            dose_targets={
                ("oral", pathway_parameters.parent_species_id): amount_state_name("stomach_solid", pathway_parameters.parent_species_id),
                ("iv_bolus", pathway_parameters.parent_species_id): amount_state_name("central", pathway_parameters.parent_species_id),
            },
            central_states={
                pathway_parameters.parent_species_id: amount_state_name("central", pathway_parameters.parent_species_id),
                **{
                    item.species_id: amount_state_name("central", item.species_id)
                    for item in systemic_products
                },
            },
            urine_states={
                pathway_parameters.parent_species_id: amount_state_name("urine", pathway_parameters.parent_species_id),
                **{
                    item.species_id: amount_state_name("urine", item.species_id)
                    for item in systemic_products
                },
            },
            feces_parent_state=amount_state_name("feces", pathway_parameters.parent_species_id),
            parent_liver_states=tuple(
                amount_state_name(zone, pathway_parameters.parent_species_id)
                for zone in LIVER_ZONES
            ),
            mechanism_factors=mechanism_factors,
        )

    def apply_species_doses(self, state: np.ndarray, events: Sequence[SpeciesDoseEvent]) -> np.ndarray:
        values = self.state_registry.validate_vector(state).copy()
        for event in events:
            if event.species_id != self.parent_species_id:
                raise ValueError("generic hepatic-pathway model accepts parent dose events only")
            amount_umol = float(
                self.state_registry.species[self.parent_species_id].convert_amount(
                    event.amount, event.unit, AmountUnit.UMOL
                )
            )
            if event.route == "iv_bolus":
                values[self.state_registry.index(amount_state_name("central", self.parent_species_id))] += amount_umol
                continue
            dissolved = amount_umol * self.formulation.initial_dissolved_fraction
            values[self.state_registry.index(amount_state_name("stomach_dissolved", self.parent_species_id))] += dissolved
            values[self.state_registry.index(amount_state_name("stomach_solid", self.parent_species_id))] += amount_umol - dissolved
        return values

    def concentration_record(self, state, patient, drug):
        record = super().concentration_record(state, patient, drug)
        values = self.state_registry.validate_vector(state)
        wall_umol = sum(
            _amount(self.state_registry, values, f"{segment}_wall", self.parent_species_id)
            for segment in SEGMENTS
        )
        record["gut_parent_mg_l"] = self._species_mass_mg(self.parent_species_id, wall_umol) / patient.gut_volume_l
        record["candidate_screening_model_not_clinically_validated"] = 1.0
        record["structural_limitation_count"] = float(len(self.pathway_parameters.structural_limitations))
        # ``MultiSpeciesPBPKModel`` reports totals for observed central species.
        # Screening also needs the explicitly declared non-systemic metabolic
        # sink so pathway ablations and the parent-moiety ledger can be audited.
        for product in self.pathway_parameters.products:
            total_umol = sum(
                max(float(values[self.state_registry.index(state_name)]), 0.0)
                for state_name in self.state_registry.names_for_species(product.species_id)
            )
            record[f"total_{product.species_id}_umol"] = total_umol
        return record


__all__ = [
    "HepaticPathwayParameters",
    "HepaticPathwayPBPKModel",
    "HepaticProductPathwaySpec",
    "amount_state_name",
    "build_hepatic_pathway_network",
    "build_hepatic_pathway_states",
]
