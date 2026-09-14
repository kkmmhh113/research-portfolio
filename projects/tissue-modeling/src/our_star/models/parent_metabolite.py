"""Reaction-coupled segmented PBPK model with one named circulating metabolite."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from ..chemistry import Reaction, ReactionNetwork, StoichiometricTerm, UntrackedCoproduct
from ..core import ChemicalSpecies, ModelContext, SpeciesRegistry, StateRegistry, StateSpec, SystemAssembler
from ..organs import KidneyModule, LiverModule, RestOfBodyModule, SegmentedGutModule
from ..pbpk import DoseEvent, DrugParameters, PatientPhysiology
from .segmented import FormulationParameters, SEGMENTED_STATE_NAMES, SegmentedPBPKModel


@dataclass(frozen=True)
class NamedMetaboliteParameters:
    """Identity and provenance for a one-to-one parent-to-metabolite reaction.

    The current mass-state solver can represent a product whose molecular
    weight is no greater than the dosed parent. The difference is accumulated
    as a named *parent-derived* coproduct mass pool. This closes administered
    drug mass without pretending that solvent/cofactor atoms are part of the
    administered dose; complete elemental reaction chemistry remains available
    in :mod:`our_star.chemistry` for future molar-state models.
    """

    reaction_id: str
    metabolite_species_id: str
    metabolite_name: str
    metabolite_smiles: str
    metabolite_molecular_weight_g_mol: float
    coproduct_id: str
    coproduct_name: str
    parameter_provenance: str

    def __post_init__(self) -> None:
        for label in (
            "reaction_id",
            "metabolite_species_id",
            "metabolite_name",
            "metabolite_smiles",
            "coproduct_id",
            "coproduct_name",
            "parameter_provenance",
        ):
            if not str(getattr(self, label)).strip():
                raise ValueError(f"{label} is required")
        value = float(self.metabolite_molecular_weight_g_mol)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("metabolite molecular weight must be finite and > 0")
        object.__setattr__(self, "metabolite_molecular_weight_g_mol", value)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "NamedMetaboliteParameters":
        return cls(**dict(raw))


def build_parent_metabolite_network(
    drug: DrugParameters,
    metabolite: NamedMetaboliteParameters,
) -> ReactionNetwork:
    """Build an administered-mass-balanced one-to-one conversion network."""

    difference = drug.molecular_weight_g_mol - metabolite.metabolite_molecular_weight_g_mol
    if difference <= 0.0:
        raise ValueError(
            "the current mg-state adapter requires metabolite MW < parent MW; "
            "use a full molar/cofactor network for mass-gaining conjugates"
        )
    return ReactionNetwork(
        species=(
            ChemicalSpecies(
                species_id="parent",
                name=drug.name,
                molecular_weight_g_mol=drug.molecular_weight_g_mol,
                structure=drug.smiles,
            ),
            ChemicalSpecies(
                species_id=metabolite.metabolite_species_id,
                name=metabolite.metabolite_name,
                molecular_weight_g_mol=metabolite.metabolite_molecular_weight_g_mol,
                structure=metabolite.metabolite_smiles,
            ),
        ),
        reactions=(
            Reaction(
                reaction_id=metabolite.reaction_id,
                terms=(
                    StoichiometricTerm("parent", -1.0),
                    StoichiometricTerm(metabolite.metabolite_species_id, 1.0),
                ),
                untracked_coproducts=(
                    UntrackedCoproduct(
                        coproduct_id=metabolite.coproduct_id,
                        name=metabolite.coproduct_name,
                        molecular_weight_g_mol=difference,
                    ),
                ),
                description=metabolite.parameter_provenance,
            ),
        ),
    )


def _explicit_state_registry(
    drug: DrugParameters,
    metabolite: NamedMetaboliteParameters,
) -> StateRegistry:
    species = SpeciesRegistry(
        (
            ChemicalSpecies(
                species_id="parent",
                name=drug.name,
                molecular_weight_g_mol=drug.molecular_weight_g_mol,
                structure=drug.smiles,
            ),
            ChemicalSpecies(
                species_id=metabolite.metabolite_species_id,
                name=metabolite.metabolite_name,
                molecular_weight_g_mol=metabolite.metabolite_molecular_weight_g_mol,
                structure=metabolite.metabolite_smiles,
            ),
            ChemicalSpecies(
                species_id="reaction_coproduct_mass",
                name=metabolite.coproduct_name,
                molecular_weight_g_mol=drug.molecular_weight_g_mol,
                mass_surrogate=True,
            ),
        )
    )
    sinks = {"urine_parent", "urine_metabolite", "feces_parent", "other_products"}
    records: list[StateSpec] = []
    for name in SEGMENTED_STATE_NAMES:
        if name in {"central_metabolite", "urine_metabolite"}:
            species_id = metabolite.metabolite_species_id
        elif name == "other_products":
            species_id = "reaction_coproduct_mass"
        else:
            species_id = "parent"
        records.append(
            StateSpec(
                name=name,
                species_id=species_id,
                compartment=name.rsplit("_", 1)[0],
                sink=name in sinks,
            )
        )
    return StateRegistry(records, species)


class ReactionCoupledSegmentedPBPKModel:
    """Segmented PBPK dynamics whose metabolic formation uses molar stoichiometry."""

    model_id = "our_star.segmented_parent_metabolite_pbpk.v0.3"

    def __init__(
        self,
        drug: DrugParameters,
        formulation: FormulationParameters,
        metabolite: NamedMetaboliteParameters,
    ) -> None:
        expected_yield = (
            metabolite.metabolite_molecular_weight_g_mol
            / drug.molecular_weight_g_mol
        )
        if not np.isclose(
            drug.metabolite_mass_yield,
            expected_yield,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError(
                "metabolite_mass_yield must equal metabolite_MW / parent_MW "
                "for a one-to-one explicit reaction"
            )
        self.drug = drug
        self.formulation = formulation
        self.metabolite = metabolite
        self.reaction_network = build_parent_metabolite_network(drug, metabolite)
        self.state_registry = _explicit_state_registry(drug, metabolite)
        self.assembler = SystemAssembler(
            self.state_registry,
            (SegmentedGutModule(), LiverModule(), KidneyModule(), RestOfBodyModule()),
        )
        self._record_adapter = SegmentedPBPKModel(drug, formulation)

    def _validate_drug(self, drug: DrugParameters) -> None:
        if drug != self.drug:
            raise ValueError("the reaction-coupled model must be rebuilt when drug parameters change")

    def initial_state(self) -> np.ndarray:
        return self.state_registry.zeros()

    def apply_doses(self, state: np.ndarray, events: Sequence[DoseEvent]) -> np.ndarray:
        return self._record_adapter.apply_doses(state, events)

    def reaction_accounting(self, parent_metabolism_mg_h: float):
        if not np.isfinite(parent_metabolism_mg_h) or parent_metabolism_mg_h < 0.0:
            raise ValueError("parent metabolism rate must be finite and >= 0")
        extent_umol_h = (
            parent_metabolism_mg_h * 1000.0 / self.drug.molecular_weight_g_mol
        )
        rates = {self.metabolite.reaction_id: extent_umol_h}
        return self.reaction_network.derivatives(rates), self.reaction_network.mass_accounting(rates)

    def rhs(
        self,
        time_h: float,
        state: np.ndarray,
        patient: PatientPhysiology,
        drug: DrugParameters,
    ) -> np.ndarray:
        self._validate_drug(drug)
        derivative = self.assembler.rhs(
            time_h,
            state,
            ModelContext(patient=patient, drug=drug, formulation=self.formulation),
        )
        central_metabolite = self.state_registry.index("central_metabolite")
        urine_metabolite = self.state_registry.index("urine_metabolite")
        coproduct = self.state_registry.index("other_products")
        metabolite_elimination_mg_h = float(derivative[urine_metabolite])
        parent_metabolism_mg_h = float(
            derivative[central_metabolite]
            + derivative[urine_metabolite]
            + derivative[coproduct]
        )
        parent_metabolism_mg_h = max(parent_metabolism_mg_h, 0.0)
        reaction_derivatives, diagnostic = self.reaction_accounting(parent_metabolism_mg_h)
        if abs(diagnostic.closure_error_mg_h) > 1.0e-9:
            raise FloatingPointError("reaction-network mass-rate closure failed")
        formed_umol_h = reaction_derivatives.species_umol_h.get(
            self.metabolite.metabolite_species_id, 0.0
        )
        formed_metabolite_mg_h = float(
            self.reaction_network.species[1].convert_amount(
                formed_umol_h, "umol", "mg"
            )
        )
        derivative[central_metabolite] = (
            formed_metabolite_mg_h - metabolite_elimination_mg_h
        )
        derivative[coproduct] = reaction_derivatives.total_untracked_coproduct_mg_h
        return derivative

    def concentration_record(
        self,
        state: np.ndarray,
        patient: PatientPhysiology,
        drug: DrugParameters,
    ) -> dict[str, float]:
        self._validate_drug(drug)
        record = dict(self._record_adapter.concentration_record(state, patient, drug))
        record[f"plasma_{self.metabolite.metabolite_species_id}_mg_l"] = record[
            "plasma_metabolite_mg_l"
        ]
        record[f"urine_{self.metabolite.metabolite_species_id}_mg"] = record[
            "urine_metabolite_mg"
        ]
        record[f"cumulative_{self.metabolite.coproduct_id}_mg"] = record[
            "other_products_mg"
        ]
        record["reaction_static_closure_error_mg_per_umol"] = float(
            self.reaction_network.balance_diagnostics[0].closure_error_mg_per_umol_extent
        )
        return record
