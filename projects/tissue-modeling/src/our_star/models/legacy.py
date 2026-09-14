"""Modular assembly that is numerically equivalent to the original PBPK RHS."""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import numpy as np

from ..core import (
    ChemicalSpecies,
    ModelContext,
    SpeciesRegistry,
    StateRegistry,
    StateSpec,
    SystemAssembler,
)
from ..organs import KidneyModule, LegacyGutModule, LiverModule, RestOfBodyModule
from ..pbpk import (
    STATE_NAMES,
    DoseEvent,
    DrugParameters,
    PatientPhysiology,
    apply_doses,
    concentration_record,
)


def build_species_registry(drug: DrugParameters) -> SpeciesRegistry:
    """Create identities for the parent and current lumped mass products."""

    return SpeciesRegistry(
        (
            ChemicalSpecies(
                species_id="parent",
                name=drug.name,
                molecular_weight_g_mol=drug.molecular_weight_g_mol,
                structure=drug.smiles,
            ),
            ChemicalSpecies(
                species_id="lumped_metabolite",
                name=f"{drug.name} lumped metabolite mass",
                molecular_weight_g_mol=drug.molecular_weight_g_mol,
                mass_surrogate=True,
            ),
            ChemicalSpecies(
                species_id="other_products",
                name=f"{drug.name} untracked metabolic product mass",
                molecular_weight_g_mol=drug.molecular_weight_g_mol,
                mass_surrogate=True,
            ),
        )
    )


def _species_for_state(state_name: str) -> str:
    if state_name in {"central_metabolite", "urine_metabolite"}:
        return "lumped_metabolite"
    if state_name == "other_products":
        return "other_products"
    return "parent"


def _compartment_for_state(state_name: str) -> str:
    if state_name.startswith("liver_zone"):
        return state_name.removesuffix("_parent")
    if state_name.endswith("_parent"):
        return state_name.removesuffix("_parent")
    if state_name.endswith("_metabolite"):
        return state_name.removesuffix("_metabolite")
    return state_name


def build_legacy_state_registry(drug: DrugParameters) -> StateRegistry:
    species = build_species_registry(drug)
    sink_states = {
        "urine_parent",
        "urine_metabolite",
        "feces_parent",
        "other_products",
    }
    return StateRegistry(
        (
            StateSpec(
                name=name,
                species_id=_species_for_state(name),
                compartment=_compartment_for_state(name),
                sink=name in sink_states,
            )
            for name in STATE_NAMES
        ),
        species,
    )


@lru_cache(maxsize=64)
def legacy_assembler(drug: DrugParameters) -> SystemAssembler:
    return SystemAssembler(
        build_legacy_state_registry(drug),
        (LegacyGutModule(), LiverModule(), KidneyModule(), RestOfBodyModule()),
    )


def assembled_legacy_rhs(
    time_h: float,
    state: np.ndarray,
    patient: PatientPhysiology,
    drug: DrugParameters,
) -> np.ndarray:
    assembler = legacy_assembler(drug)
    return assembler.rhs(time_h, state, ModelContext(patient=patient, drug=drug))


class LegacyPBPKModel:
    """Simulation adapter for the original-state modular assembly."""

    model_id = "our_star.virtual_phase1_pbpk.v0.2_modular_equivalent"

    def __init__(self, drug: DrugParameters) -> None:
        self.drug = drug
        self.assembler = legacy_assembler(drug)
        self.state_registry = self.assembler.states

    def _validate_drug(self, drug: DrugParameters) -> None:
        if drug != self.drug:
            raise ValueError("LegacyPBPKModel must be rebuilt when drug parameters change")

    def initial_state(self) -> np.ndarray:
        return self.state_registry.zeros()

    def apply_doses(
        self, state: np.ndarray, events: Sequence[DoseEvent]
    ) -> np.ndarray:
        return apply_doses(state, events)

    def rhs(
        self,
        time_h: float,
        state: np.ndarray,
        patient: PatientPhysiology,
        drug: DrugParameters,
    ) -> np.ndarray:
        self._validate_drug(drug)
        return self.assembler.rhs(
            time_h,
            state,
            ModelContext(patient=patient, drug=drug),
        )

    def concentration_record(
        self,
        state: np.ndarray,
        patient: PatientPhysiology,
        drug: DrugParameters,
    ) -> dict[str, float]:
        self._validate_drug(drug)
        return concentration_record(state, patient, drug)
