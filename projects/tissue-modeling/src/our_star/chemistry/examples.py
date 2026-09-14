"""Small non-clinical examples for the explicit reaction-network API."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np

from ..core.species import ChemicalSpecies
from .reactions import (
    ElementCount,
    MassAccountingDiagnostic,
    MolecularFormula,
    Reaction,
    ReactionDerivatives,
    ReactionNetwork,
    SpeciesFormula,
    StoichiometricTerm,
    UntrackedCoproduct,
)


@dataclass(frozen=True)
class ParentMetaboliteExampleResult:
    """One exact constant-extent update used to demonstrate mass closure."""

    network: ReactionNetwork
    initial_amounts_umol: Mapping[str, float]
    final_amounts_umol: Mapping[str, float]
    cumulative_untracked_mass_mg: Mapping[str, float]
    derivatives: ReactionDerivatives
    mass_accounting: MassAccountingDiagnostic

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "initial_amounts_umol",
            MappingProxyType(dict(self.initial_amounts_umol)),
        )
        object.__setattr__(
            self,
            "final_amounts_umol",
            MappingProxyType(dict(self.final_amounts_umol)),
        )
        object.__setattr__(
            self,
            "cumulative_untracked_mass_mg",
            MappingProxyType(dict(self.cumulative_untracked_mass_mg)),
        )


def build_minimal_parent_metabolite_network() -> ReactionNetwork:
    """Return a generic 200 -> 180 + 20 g/mol mass-balanced reaction.

    The values are intentionally generic and are not parameters for a drug or
    a clinical model.  The 20 g/mol coproduct is accumulated outside the
    dynamic species vector while remaining explicit in the mass diagnostic.
    """

    parent = ChemicalSpecies(
        species_id="example_parent",
        name="generic parent",
        molecular_weight_g_mol=200.0,
    )
    metabolite = ChemicalSpecies(
        species_id="example_metabolite",
        name="generic metabolite",
        molecular_weight_g_mol=180.0,
    )
    parent_formula = MolecularFormula(
        (ElementCount("C", 10), ElementCount("H", 20))
    )
    metabolite_formula = MolecularFormula(
        (ElementCount("C", 9), ElementCount("H", 18))
    )
    coproduct_formula = MolecularFormula(
        (ElementCount("C", 1), ElementCount("H", 2))
    )
    return ReactionNetwork(
        species=(parent, metabolite),
        reactions=(
            Reaction(
                reaction_id="example_parent_to_metabolite",
                terms=(
                    StoichiometricTerm("example_parent", -1.0),
                    StoichiometricTerm("example_metabolite", 1.0),
                ),
                untracked_coproducts=(
                    UntrackedCoproduct(
                        coproduct_id="example_untracked_coproduct",
                        name="generic untracked coproduct",
                        molecular_weight_g_mol=20.0,
                        formula=coproduct_formula,
                    ),
                ),
                description="Non-clinical molar and mass-accounting example",
            ),
        ),
        species_formulas=(
            SpeciesFormula("example_parent", parent_formula),
            SpeciesFormula("example_metabolite", metabolite_formula),
        ),
    )


def run_minimal_parent_metabolite_example(
    *,
    initial_parent_umol: float = 10.0,
    extent_umol_h: float = 2.0,
    duration_h: float = 3.0,
) -> ParentMetaboliteExampleResult:
    """Apply a constant extent without coupling it to the PBPK model.

    This is a transparent algebraic example, not a numerical ODE integrator.
    It refuses an update that would consume more parent than is available.
    """

    values = (float(initial_parent_umol), float(extent_umol_h), float(duration_h))
    if any(not np.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("example amounts, extent rate, and duration must be finite and >= 0")
    consumed_umol = extent_umol_h * duration_h
    if consumed_umol > initial_parent_umol:
        raise ValueError("example reaction extent would consume more parent than available")

    network = build_minimal_parent_metabolite_network()
    rates = {"example_parent_to_metabolite": extent_umol_h}
    derivatives = network.derivatives(rates)
    initial = {"example_parent": initial_parent_umol, "example_metabolite": 0.0}
    final = {
        species_id: initial[species_id]
        + derivatives.species_umol_h.get(species_id, 0.0) * duration_h
        for species_id in initial
    }
    cumulative = {
        coproduct_id: rate * duration_h
        for coproduct_id, rate in derivatives.untracked_coproduct_mg_h.items()
    }
    return ParentMetaboliteExampleResult(
        network=network,
        initial_amounts_umol=initial,
        final_amounts_umol=final,
        cumulative_untracked_mass_mg=cumulative,
        derivatives=derivatives,
        mass_accounting=network.mass_accounting(rates),
    )
