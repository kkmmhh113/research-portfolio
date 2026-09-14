"""Distributed enalapril-to-enalaprilat µmol PBPK candidate model.

This post-v0.3 model is intentionally parallel to the frozen single-metabolite
adapter.  It separates segmented oral input, liver-zone CES1 formation,
parent/metabolite circulation, renal filtration/secretion, reversible ACE
binding, chemical stoichiometry, and the study observation mapping.

All kinetic values must be supplied from an evidence manifest.  The model has
no fitting callback and never loads a clinical outcome.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import yaml

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
from ..mechanisms import (
    EnalaprilatRenalDisposition,
    FormationKinetics,
    LinearCES1Formation,
    LinearUnboundSecretion,
    SaturableACEBinding,
)
from ..observation import ObservationSpec, predict_observation
from ..pbpk import DoseEvent, DrugParameters
from .multispecies import MultiSpeciesPBPKModel
from .segmented import FormulationParameters


PARENT = "enalapril"
METABOLITE = "enalaprilat"
HYDROLYSIS_FRAGMENT = "enalapril_hydrolysis_fragment"
ABSORPTION_PORT = FluxKey("intestinal_absorption_to_liver", PARENT)


@dataclass(frozen=True)
class EnalaprilMechanismParameters:
    metabolite_molecular_weight_g_mol: float
    parent_fraction_unbound: float
    metabolite_fraction_unbound: float
    metabolite_kp_liver: float
    metabolite_kp_kidney: float
    metabolite_kp_rest: float
    formation: FormationKinetics
    renal_disposition: EnalaprilatRenalDisposition
    ace_binding: SaturableACEBinding | None
    provenance: str

    def __post_init__(self) -> None:
        for label in (
            "metabolite_molecular_weight_g_mol",
            "metabolite_kp_liver",
            "metabolite_kp_kidney",
            "metabolite_kp_rest",
        ):
            value = float(getattr(self, label))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{label} must be finite and > 0")
            object.__setattr__(self, label, value)
        for label in ("parent_fraction_unbound", "metabolite_fraction_unbound"):
            value = float(getattr(self, label))
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{label} must be finite and in [0, 1]")
            object.__setattr__(self, label, value)
        if not callable(getattr(self.formation, "rate_umol_h", None)):
            raise TypeError("formation must implement FormationKinetics")
        if not isinstance(self.renal_disposition, EnalaprilatRenalDisposition):
            raise TypeError("renal_disposition must be EnalaprilatRenalDisposition")
        if self.ace_binding is not None and not isinstance(
            self.ace_binding, SaturableACEBinding
        ):
            raise TypeError("ace_binding must be SaturableACEBinding or None")
        if not self.provenance.strip():
            raise ValueError("enalapril mechanism provenance is required")


def _amount(
    states: AmountStateRegistry,
    vector: np.ndarray,
    state_name: str,
) -> float:
    return max(float(vector[states.index(state_name)]), 0.0)


class _EnalaprilSegmentedGutModule:
    name = "enalapril_segmented_gi"
    required_states = (
        "stomach_solid_enalapril",
        "stomach_dissolved_enalapril",
        "duodenum_lumen_enalapril",
        "jejunum_lumen_enalapril",
        "ileum_lumen_enalapril",
        "feces_enalapril",
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
        formulation = context.formulation
        if not isinstance(formulation, FormulationParameters):
            raise TypeError("enalapril GI module requires FormulationParameters")
        drug = context.drug
        solid = _amount(states, amounts_umol, "stomach_solid_enalapril")
        dissolved = _amount(states, amounts_umol, "stomach_dissolved_enalapril")
        dissolution = formulation.dissolution_rate_h * solid
        emptying = formulation.gastric_emptying_rate_h * dissolved
        derivatives: dict[str, float] = {
            "stomach_solid_enalapril": -dissolution,
            "stomach_dissolved_enalapril": dissolution - emptying,
        }
        incoming = emptying
        absorbed_total = 0.0
        for index, segment in enumerate(("duodenum", "jejunum", "ileum")):
            name = f"{segment}_lumen_enalapril"
            lumen = _amount(states, amounts_umol, name)
            absorbed = (
                drug.fraction_absorbed
                * formulation.segment_absorption_rate_h[index]
                * lumen
            )
            transit = formulation.segment_transit_rate_h[index] * lumen
            derivatives[name] = incoming - absorbed - transit
            absorbed_total += absorbed
            incoming = transit
        derivatives["feces_enalapril"] = incoming
        return AmountModuleResult(
            derivatives_umol_h=derivatives,
            outputs_umol_h={ABSORPTION_PORT: float(absorbed_total)},
        )


class _EnalaprilLiverModule:
    name = "enalapril_three_zone_liver"
    required_states = (
        "central_enalapril",
        "liver_z1_enalapril",
        "liver_z2_enalapril",
        "liver_z3_enalapril",
        "central_enalaprilat_mobile",
        "liver_z1_enalaprilat",
        "liver_z2_enalaprilat",
        "liver_z3_enalaprilat",
        "cumulative_hydrolysis_fragment_umol",
    )
    required_inputs = (ABSORPTION_PORT,)

    def __init__(
        self,
        mechanisms: EnalaprilMechanismParameters,
        reaction_network: OpenReactionNetwork,
    ) -> None:
        self.mechanisms = mechanisms
        self.reaction_network = reaction_network

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
            _amount(states, amounts_umol, "central_enalapril")
            / patient.central_volume_l
        )
        metabolite_central = (
            _amount(states, amounts_umol, "central_enalaprilat_mobile")
            / patient.central_volume_l
        )
        parent_equivalent = np.asarray(
            [
                _amount(states, amounts_umol, f"liver_z{zone}_enalapril")
                / zone_volume
                / drug.kp_liver
                for zone in (1, 2, 3)
            ],
            dtype=np.float64,
        )
        metabolite_equivalent = np.asarray(
            [
                _amount(states, amounts_umol, f"liver_z{zone}_enalaprilat")
                / zone_volume
                / self.mechanisms.metabolite_kp_liver
                for zone in (1, 2, 3)
            ],
            dtype=np.float64,
        )
        activity = (
            patient.enzyme_activity_factor
            * patient.liver_function_fraction
            * context.mechanism_factor("ces1_activity")
        )
        formation = np.asarray(
            [
                self.mechanisms.formation.rate_umol_h(
                    self.mechanisms.parent_fraction_unbound
                    * parent_equivalent[index],
                    activity_factor=activity
                    * drug.normalized_zone_activity[index],
                )
                for index in range(3)
            ],
            dtype=np.float64,
        )
        total_extent = float(formation.sum())
        diagnostic = self.reaction_network.mass_rate_diagnostic(
            {"enalapril_to_enalaprilat": total_extent}
        )
        if abs(diagnostic.closure_error_mg_h) > 1.0e-9:
            raise FloatingPointError("enalapril hydrolysis mass-rate closure failed")

        derivatives: dict[str, float] = {
            "central_enalapril": q_liver * (parent_equivalent[2] - parent_central),
            "liver_z1_enalapril": (
                inputs_umol_h[ABSORPTION_PORT]
                + q_liver * parent_central
                - q_liver * parent_equivalent[0]
                - formation[0]
            ),
            "liver_z2_enalapril": (
                q_liver * (parent_equivalent[0] - parent_equivalent[1])
                - formation[1]
            ),
            "liver_z3_enalapril": (
                q_liver * (parent_equivalent[1] - parent_equivalent[2])
                - formation[2]
            ),
            "central_enalaprilat_mobile": q_liver
            * (metabolite_equivalent[2] - metabolite_central),
            "liver_z1_enalaprilat": (
                q_liver * metabolite_central
                - q_liver * metabolite_equivalent[0]
                + formation[0]
            ),
            "liver_z2_enalaprilat": (
                q_liver * (metabolite_equivalent[0] - metabolite_equivalent[1])
                + formation[1]
            ),
            "liver_z3_enalaprilat": (
                q_liver * (metabolite_equivalent[1] - metabolite_equivalent[2])
                + formation[2]
            ),
            "cumulative_hydrolysis_fragment_umol": total_extent,
        }
        return AmountModuleResult(derivatives_umol_h=derivatives)


class _EnalaprilKidneyModule:
    name = "enalapril_kidney"
    required_states = (
        "central_enalapril",
        "kidney_enalapril",
        "urine_enalapril",
        "central_enalaprilat_mobile",
        "kidney_enalaprilat",
        "urine_enalaprilat_filtration",
        "urine_enalaprilat_secretion",
    )
    required_inputs: tuple[FluxKey, ...] = ()

    def __init__(self, mechanisms: EnalaprilMechanismParameters) -> None:
        self.mechanisms = mechanisms

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
        central_parent = _amount(states, amounts_umol, "central_enalapril") / patient.central_volume_l
        kidney_parent = (
            _amount(states, amounts_umol, "kidney_enalapril")
            / patient.kidney_volume_l
            / drug.kp_kidney
        )
        central_metabolite = (
            _amount(states, amounts_umol, "central_enalaprilat_mobile")
            / patient.central_volume_l
        )
        kidney_metabolite = (
            _amount(states, amounts_umol, "kidney_enalaprilat")
            / patient.kidney_volume_l
            / self.mechanisms.metabolite_kp_kidney
        )
        parent_exchange = patient.renal_flow_l_h * (central_parent - kidney_parent)
        metabolite_exchange = patient.renal_flow_l_h * (
            central_metabolite - kidney_metabolite
        )
        allometric = (patient.body_weight_kg / 70.0) ** 0.75
        renal_factor = patient.renal_function_fraction * allometric
        parent_elimination = drug.renal_clearance_l_h * renal_factor * kidney_parent
        renal_rates = self.mechanisms.renal_disposition.rates_umol_h(
            kidney_metabolite,
            gfr_factor=renal_factor * context.mechanism_factor("enalaprilat_gfr"),
            secretion_factor=renal_factor
            * context.mechanism_factor("enalaprilat_secretion"),
        )
        return AmountModuleResult(
            derivatives_umol_h={
                "central_enalapril": -parent_exchange,
                "kidney_enalapril": parent_exchange - parent_elimination,
                "urine_enalapril": parent_elimination,
                "central_enalaprilat_mobile": -metabolite_exchange,
                "kidney_enalaprilat": metabolite_exchange - renal_rates.total_umol_h,
                "urine_enalaprilat_filtration": renal_rates.filtration_umol_h,
                "urine_enalaprilat_secretion": renal_rates.secretion_umol_h,
            }
        )


class _EnalaprilRestModule:
    name = "enalapril_rest_of_body"
    required_states = (
        "central_enalapril",
        "rest_enalapril",
        "central_enalaprilat_mobile",
        "rest_enalaprilat",
    )
    required_inputs: tuple[FluxKey, ...] = ()

    def __init__(self, mechanisms: EnalaprilMechanismParameters) -> None:
        self.mechanisms = mechanisms

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
        central_parent = _amount(states, amounts_umol, "central_enalapril") / patient.central_volume_l
        rest_parent = (
            _amount(states, amounts_umol, "rest_enalapril")
            / patient.rest_volume_l
            / drug.kp_rest
        )
        central_metabolite = (
            _amount(states, amounts_umol, "central_enalaprilat_mobile")
            / patient.central_volume_l
        )
        rest_metabolite = (
            _amount(states, amounts_umol, "rest_enalaprilat")
            / patient.rest_volume_l
            / self.mechanisms.metabolite_kp_rest
        )
        parent_exchange = patient.rest_flow_l_h * (central_parent - rest_parent)
        metabolite_exchange = patient.rest_flow_l_h * (
            central_metabolite - rest_metabolite
        )
        return AmountModuleResult(
            derivatives_umol_h={
                "central_enalapril": -parent_exchange,
                "rest_enalapril": parent_exchange,
                "central_enalaprilat_mobile": -metabolite_exchange,
                "rest_enalaprilat": metabolite_exchange,
            }
        )


class _EnalaprilACEBindingModule:
    name = "enalaprilat_ace_binding"
    required_states = (
        "central_enalaprilat_mobile",
        "ace_bound_enalaprilat",
    )
    required_inputs: tuple[FluxKey, ...] = ()

    def __init__(self, mechanisms: EnalaprilMechanismParameters) -> None:
        self.mechanisms = mechanisms

    def evaluate(
        self,
        _time_h: float,
        amounts_umol: np.ndarray,
        context: AmountModelContext,
        states: AmountStateRegistry,
        _inputs_umol_h: Mapping[FluxKey, float],
    ) -> AmountModuleResult:
        binding = self.mechanisms.ace_binding
        if binding is None:
            return AmountModuleResult(derivatives_umol_h={})
        patient = context.patient
        mobile_total = (
            _amount(states, amounts_umol, "central_enalaprilat_mobile")
            / patient.central_volume_l
        )
        bound = _amount(states, amounts_umol, "ace_bound_enalaprilat")
        fluxes = binding.fluxes_umol_h(
            self.mechanisms.metabolite_fraction_unbound * mobile_total,
            bound,
            binding_capacity_factor=context.mechanism_factor("ace_capacity"),
        )
        return AmountModuleResult(
            derivatives_umol_h={
                "central_enalaprilat_mobile": -fluxes.net_to_bound_umol_h,
                "ace_bound_enalaprilat": fluxes.net_to_bound_umol_h,
            }
        )


def _build_registry(
    drug: DrugParameters,
    mechanisms: EnalaprilMechanismParameters,
) -> tuple[AmountStateRegistry, OpenReactionNetwork, DualAmountLedger]:
    difference = drug.molecular_weight_g_mol - mechanisms.metabolite_molecular_weight_g_mol
    if difference <= 0.0:
        raise ValueError("enalapril hydrolysis requires parent MW > metabolite MW")
    species = SpeciesRegistry(
        (
            ChemicalSpecies(PARENT, drug.name, drug.molecular_weight_g_mol, drug.smiles),
            ChemicalSpecies(
                METABOLITE,
                "enalaprilat",
                mechanisms.metabolite_molecular_weight_g_mol,
            ),
            ChemicalSpecies(
                "hydrolysis_fragment_counter",
                "cumulative hydrolysis-fragment counter",
                difference,
                mass_surrogate=True,
            ),
        )
    )
    network = OpenReactionNetwork(
        species=(species[PARENT], species[METABOLITE]),
        reactions=(
            OpenReaction(
                reaction_id="enalapril_to_enalaprilat",
                terms=(
                    StoichiometricTerm(PARENT, -1.0),
                    StoichiometricTerm(METABOLITE, 1.0),
                ),
                external_participants=(
                    ExternalParticipant(
                        participant_id=HYDROLYSIS_FRAGMENT,
                        name="parent-derived hydrolysis fragment",
                        coefficient=1.0,
                        molecular_weight_g_mol=difference,
                        role=ExternalParticipantRole.COPRODUCT,
                        provenance=mechanisms.provenance,
                    ),
                ),
                description=mechanisms.provenance,
            ),
        ),
    )
    physical_states = (
        "stomach_solid_enalapril",
        "stomach_dissolved_enalapril",
        "duodenum_lumen_enalapril",
        "jejunum_lumen_enalapril",
        "ileum_lumen_enalapril",
        "central_enalapril",
        "liver_z1_enalapril",
        "liver_z2_enalapril",
        "liver_z3_enalapril",
        "kidney_enalapril",
        "rest_enalapril",
        "urine_enalapril",
        "feces_enalapril",
    )
    metabolite_states = (
        "central_enalaprilat_mobile",
        "liver_z1_enalaprilat",
        "liver_z2_enalaprilat",
        "liver_z3_enalaprilat",
        "kidney_enalaprilat",
        "rest_enalaprilat",
        "ace_bound_enalaprilat",
        "urine_enalaprilat_filtration",
        "urine_enalaprilat_secretion",
    )
    sinks = {
        "urine_enalapril",
        "feces_enalapril",
        "urine_enalaprilat_filtration",
        "urine_enalaprilat_secretion",
        "cumulative_hydrolysis_fragment_umol",
    }
    records = [
        AmountStateSpec(
            name=name,
            species_id=PARENT,
            compartment_id=name.rsplit("_", 1)[0],
            sink=name in sinks,
        )
        for name in physical_states
    ]
    records.extend(
        AmountStateSpec(
            name=name,
            species_id=METABOLITE,
            compartment_id=name.rsplit("_", 1)[0],
            sink=name in sinks,
        )
        for name in metabolite_states
    )
    records.append(
        AmountStateSpec(
            name="cumulative_hydrolysis_fragment_umol",
            species_id="hydrolysis_fragment_counter",
            compartment_id="external_export",
            sink=True,
            accounting_only=True,
        )
    )
    states = AmountStateRegistry(records, species)
    ledger = DualAmountLedger(
        states,
        (
            SpeciesMoiety(PARENT, {"enalapril": 1.0}),
            SpeciesMoiety(METABOLITE, {"enalapril": 1.0}),
        ),
        conserved_moiety_id="enalapril",
        reference_species_id=PARENT,
        external_participants=network.external_participants,
        external_import_states={},
        external_export_states={
            HYDROLYSIS_FRAGMENT: "cumulative_hydrolysis_fragment_umol"
        },
    )
    return states, network, ledger


class EnalaprilMechanismPBPKModel(MultiSpeciesPBPKModel):
    """M1/M2 model; optional ACE binding becomes the evidence-gated M3."""

    def __init__(
        self,
        drug: DrugParameters,
        formulation: FormulationParameters,
        mechanisms: EnalaprilMechanismParameters,
        *,
        observation_spec: ObservationSpec | None = None,
        mechanism_factors: Mapping[str, float] | None = None,
    ) -> None:
        states, network, ledger = _build_registry(drug, mechanisms)
        assembler = AmountSystemAssembler(
            states,
            (
                _EnalaprilSegmentedGutModule(),
                _EnalaprilLiverModule(mechanisms, network),
                _EnalaprilKidneyModule(mechanisms),
                _EnalaprilRestModule(mechanisms),
                _EnalaprilACEBindingModule(mechanisms),
            ),
        )
        self.mechanisms = mechanisms
        self.observation_spec = observation_spec or ObservationSpec(
            observation_id="enalaprilat_plasma_total_default",
            analyte_id=METABOLITE,
            source_matrix="plasma",
            modeled_matrix="plasma",
            quantity="total",
            matrix_bridge="identity",
            fixed_matrix_ratio=None,
            bound_recovery_fraction=1.0,
            residual_model_id=None,
            provenance="generic model-output contract; not a study-specific bridge",
        )
        if self.observation_spec.analyte_id != METABOLITE:
            raise ValueError("enalapril model observation must target enalaprilat")
        suffix = "m3" if mechanisms.ace_binding is not None else "m2"
        super().__init__(
            model_id=f"our_star.enalapril_multispecies.{suffix}.v0.4",
            drug=drug,
            formulation=formulation,
            states=states,
            assembler=assembler,
            reaction_network=network,
            ledger=ledger,
            parent_species_id=PARENT,
            dose_targets={
                ("oral", PARENT): "stomach_dissolved_enalapril",
                ("iv_bolus", PARENT): "central_enalapril",
            },
            central_states={
                PARENT: "central_enalapril",
                METABOLITE: "central_enalaprilat_mobile",
            },
            urine_states={
                PARENT: "urine_enalapril",
                METABOLITE: "urine_enalaprilat_filtration",
            },
            feces_parent_state="feces_enalapril",
            parent_liver_states=(
                "liver_z1_enalapril",
                "liver_z2_enalapril",
                "liver_z3_enalapril",
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
            if event.species_id != PARENT:
                raise ValueError("enalapril model accepts parent dose events only")
            amount_umol = float(
                self.state_registry.species[PARENT].convert_amount(
                    event.amount,
                    event.unit,
                    AmountUnit.UMOL,
                )
            )
            if event.route == "iv_bolus":
                values[self.state_registry.index("central_enalapril")] += amount_umol
                continue
            dissolved = amount_umol * self.formulation.initial_dissolved_fraction
            values[
                self.state_registry.index("stomach_dissolved_enalapril")
            ] += dissolved
            values[self.state_registry.index("stomach_solid_enalapril")] += (
                amount_umol - dissolved
            )
        return values

    def apply_doses(self, state: np.ndarray, events: Sequence[DoseEvent]) -> np.ndarray:
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

    def concentration_record(self, state, patient, drug) -> dict[str, float]:
        record = super().concentration_record(state, patient, drug)
        values = self.state_registry.validate_vector(state)
        filtration = _amount(
            self.state_registry,
            values,
            "urine_enalaprilat_filtration",
        )
        secretion = _amount(
            self.state_registry,
            values,
            "urine_enalaprilat_secretion",
        )
        total_urine_umol = filtration + secretion
        total_urine_mg = float(
            self.state_registry.species[METABOLITE].convert_amount(
                total_urine_umol,
                AmountUnit.UMOL,
                AmountUnit.MG,
            )
        )
        mobile = _amount(
            self.state_registry,
            values,
            "central_enalaprilat_mobile",
        )
        bound = _amount(
            self.state_registry,
            values,
            "ace_bound_enalaprilat",
        )
        observation = predict_observation(
            self.observation_spec,
            mobile_total_amount_umol=mobile,
            bound_amount_umol=bound,
            modeled_volume_l=patient.central_volume_l,
            molecular_weight_g_mol=self.mechanisms.metabolite_molecular_weight_g_mol,
            fraction_unbound=self.mechanisms.metabolite_fraction_unbound,
        )
        record.update(
            {
                "urine_enalaprilat_filtration_umol": filtration,
                "urine_enalaprilat_secretion_umol": secretion,
                "urine_enalaprilat_umol": total_urine_umol,
                "urine_enalaprilat_mg": total_urine_mg,
                "urine_metabolite_mg": total_urine_mg,
                "ace_bound_enalaprilat_umol": bound,
                "observation_enalaprilat_umol_l": observation.concentration_umol_l,
                "observation_enalaprilat_mg_l": observation.concentration_mg_l,
                "observation_uses_proxy_matrix": float(
                    observation.proxy_without_conversion
                ),
                "other_products_mg": record["external_export_mass_mg"],
            }
        )
        return record


def load_enalapril_mechanism_parameters(
    formation_path: str | Path,
    renal_path: str | Path,
    *,
    ace_path: str | Path | None = None,
    metabolite_molecular_weight_g_mol: float = 348.4,
) -> EnalaprilMechanismParameters:
    """Load frozen mechanism configs without accepting outcome-derived values."""

    with Path(formation_path).open("r", encoding="utf-8") as handle:
        formation_payload = yaml.safe_load(handle)["mechanism"]
    with Path(renal_path).open("r", encoding="utf-8") as handle:
        renal_payload = yaml.safe_load(handle)["mechanism"]
    formation_parameters = formation_payload["parameters"]
    renal_parameters = renal_payload["parameters"]
    formation = LinearCES1Formation(
        clearance_l_h=float(
            formation_parameters["clearance_on_unbound_concentration"]["value"]
        ),
        provenance=str(
            formation_parameters["clearance_on_unbound_concentration"]["provenance"]
        ),
    )
    secretion = LinearUnboundSecretion(
        clearance_unbound_l_h=float(
            renal_parameters["secretion_clearance_on_unbound"]["value"]
        ),
        provenance=str(
            renal_parameters["secretion_clearance_on_unbound"]["provenance"]
        ),
    )
    renal = EnalaprilatRenalDisposition(
        fraction_unbound=float(renal_parameters["fraction_unbound"]["value"]),
        gfr_l_h=float(renal_parameters["reference_gfr"]["value"]),
        secretion=secretion,
        provenance=str(renal_parameters["direct_total_renal_clearance"]["provenance"]),
    )
    ace_binding: SaturableACEBinding | None = None
    if ace_path is not None:
        with Path(ace_path).open("r", encoding="utf-8") as handle:
            ace_payload = yaml.safe_load(handle)["mechanism"]
        if ace_payload.get("status") not in {"candidate_disabled", "not_admitted"}:
            raw = ace_payload["parameters"]
            ace_binding = SaturableACEBinding(
                bmax_umol=float(raw["bmax_umol"]["value"]),
                kon_l_umol_h=float(raw["kon_l_umol_h"]["value"]),
                koff_h=float(raw["koff_h"]["value"]),
                provenance=str(ace_payload["provenance"]),
            )
    return EnalaprilMechanismParameters(
        metabolite_molecular_weight_g_mol=metabolite_molecular_weight_g_mol,
        parent_fraction_unbound=float(
            formation_parameters["parent_fraction_unbound"]["value"]
        ),
        metabolite_fraction_unbound=float(
            renal_parameters["fraction_unbound"]["value"]
        ),
        metabolite_kp_liver=1.0,
        metabolite_kp_kidney=1.0,
        metabolite_kp_rest=1.0,
        formation=formation,
        renal_disposition=renal,
        ace_binding=ace_binding,
        provenance=(
            "v0.4 fixed mechanism assembly; no clinical outcome used to construct "
            "this object"
        ),
    )
