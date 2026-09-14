"""Species-aware molar reaction networks with explicit mass accounting.

This module deliberately does not alter the mass-based v0.2 PBPK states.  It
provides the chemistry layer needed to introduce named metabolites later: a
reaction extent is expressed in micromoles, while molecular weights from
``ChemicalSpecies`` close the corresponding mass balance.

Only product mass that is explicitly represented by a tracked species or a
named ``UntrackedCoproduct`` is allowed to leave the tracked reaction.  This
makes lumping assumptions visible instead of silently treating unlike
molecular amounts as interchangeable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
import re
from typing import Iterable, Mapping

import numpy as np

from ..core.species import AmountUnit, ChemicalSpecies, SpeciesRegistry


_ELEMENT_SYMBOL = re.compile(r"^[A-Z][a-z]?$")


@dataclass(frozen=True)
class ElementCount:
    """The number of atoms of one element in a molecular formula."""

    element: str
    count: int

    def __post_init__(self) -> None:
        if not _ELEMENT_SYMBOL.fullmatch(self.element):
            raise ValueError(f"invalid element symbol: {self.element!r}")
        if isinstance(self.count, bool) or not isinstance(self.count, (int, np.integer)):
            raise ValueError("element count must be a positive integer")
        if int(self.count) <= 0:
            raise ValueError("element count must be a positive integer")
        object.__setattr__(self, "count", int(self.count))


@dataclass(frozen=True)
class MolecularFormula:
    """Immutable elemental composition for one molecule."""

    elements: tuple[ElementCount, ...]

    def __post_init__(self) -> None:
        elements = tuple(self.elements)
        if not elements:
            raise ValueError("a molecular formula requires at least one element")
        if any(not isinstance(item, ElementCount) for item in elements):
            raise TypeError("molecular formula entries must be ElementCount records")
        symbols = tuple(item.element for item in elements)
        if len(set(symbols)) != len(symbols):
            raise ValueError("an element may appear only once in a molecular formula")
        object.__setattr__(self, "elements", elements)

    @classmethod
    def from_mapping(cls, elements: Mapping[str, int]) -> "MolecularFormula":
        """Build a deterministic formula from ``{"C": 8, "H": 10, ...}``."""

        return cls(
            tuple(
                ElementCount(element=symbol, count=count)
                for symbol, count in sorted(elements.items())
            )
        )

    def as_mapping(self) -> Mapping[str, int]:
        return MappingProxyType({item.element: item.count for item in self.elements})


@dataclass(frozen=True)
class SpeciesFormula:
    """Associate an existing ``ChemicalSpecies`` identifier with a formula."""

    species_id: str
    formula: MolecularFormula

    def __post_init__(self) -> None:
        if not self.species_id.strip():
            raise ValueError("species_id is required for an elemental formula")
        if not isinstance(self.formula, MolecularFormula):
            raise TypeError("formula must be a MolecularFormula")


@dataclass(frozen=True)
class StoichiometricTerm:
    """Signed coefficient for one tracked species.

    Negative coefficients are reactants and positive coefficients are
    products.  The coefficient multiplies a reaction extent in micromoles.
    """

    species_id: str
    coefficient: float

    def __post_init__(self) -> None:
        if not self.species_id.strip():
            raise ValueError("stoichiometric species_id is required")
        coefficient = float(self.coefficient)
        if not np.isfinite(coefficient) or coefficient == 0.0:
            raise ValueError("stoichiometric coefficient must be finite and nonzero")
        object.__setattr__(self, "coefficient", coefficient)


@dataclass(frozen=True)
class UntrackedCoproduct:
    """A named product omitted from dynamic species states.

    ``molecular_weight_g_mol`` and ``coefficient`` make the omitted mass per
    micromole of reaction extent explicit.  A formula is optional; when all
    participating molecules have formulas, the network also enforces exact
    elemental balance.
    """

    coproduct_id: str
    name: str
    molecular_weight_g_mol: float
    coefficient: float = 1.0
    formula: MolecularFormula | None = None

    def __post_init__(self) -> None:
        if not self.coproduct_id.strip():
            raise ValueError("coproduct_id is required")
        if not self.name.strip():
            raise ValueError("coproduct name is required")
        molecular_weight = float(self.molecular_weight_g_mol)
        if not np.isfinite(molecular_weight) or molecular_weight <= 0.0:
            raise ValueError("coproduct molecular weight must be finite and > 0")
        coefficient = float(self.coefficient)
        if not np.isfinite(coefficient) or coefficient <= 0.0:
            raise ValueError("coproduct coefficient must be finite and > 0")
        if self.formula is not None and not isinstance(self.formula, MolecularFormula):
            raise TypeError("coproduct formula must be a MolecularFormula when supplied")
        object.__setattr__(self, "molecular_weight_g_mol", molecular_weight)
        object.__setattr__(self, "coefficient", coefficient)

    @property
    def mass_mg_per_umol_extent(self) -> float:
        """Coproduct mass generated by one micromole of reaction extent."""

        return self.coefficient * self.molecular_weight_g_mol / 1000.0


@dataclass(frozen=True)
class Reaction:
    """Immutable signed stoichiometry and declared untracked coproducts."""

    reaction_id: str
    terms: tuple[StoichiometricTerm, ...]
    untracked_coproducts: tuple[UntrackedCoproduct, ...] = ()
    description: str | None = None

    def __post_init__(self) -> None:
        if not self.reaction_id.strip():
            raise ValueError("reaction_id is required")
        terms = tuple(self.terms)
        coproducts = tuple(self.untracked_coproducts)
        if not terms:
            raise ValueError("a reaction requires tracked stoichiometric terms")
        if any(not isinstance(term, StoichiometricTerm) for term in terms):
            raise TypeError("reaction terms must be StoichiometricTerm records")
        if any(not isinstance(item, UntrackedCoproduct) for item in coproducts):
            raise TypeError(
                "untracked products must be UntrackedCoproduct records"
            )
        species_ids = tuple(term.species_id for term in terms)
        if len(set(species_ids)) != len(species_ids):
            raise ValueError("a tracked species may appear only once per reaction")
        coproduct_ids = tuple(item.coproduct_id for item in coproducts)
        if len(set(coproduct_ids)) != len(coproduct_ids):
            raise ValueError("an untracked coproduct may appear only once per reaction")
        if not any(term.coefficient < 0.0 for term in terms):
            raise ValueError("a reaction requires at least one tracked reactant")
        if not any(term.coefficient > 0.0 for term in terms) and not coproducts:
            raise ValueError("a reaction requires at least one product")
        if self.description is not None and not self.description.strip():
            raise ValueError("reaction description cannot be blank")
        object.__setattr__(self, "terms", terms)
        object.__setattr__(self, "untracked_coproducts", coproducts)


@dataclass(frozen=True)
class ReactionExtentRate:
    """A nonnegative reaction extent rate in micromoles per hour."""

    reaction_id: str
    extent_umol_h: float

    def __post_init__(self) -> None:
        if not self.reaction_id.strip():
            raise ValueError("reaction_id is required for an extent rate")
        extent = float(self.extent_umol_h)
        if not np.isfinite(extent) or extent < 0.0:
            raise ValueError("reaction extent rate must be finite and >= 0 umol/h")
        object.__setattr__(self, "extent_umol_h", extent)


@dataclass(frozen=True)
class ReactionBalanceDiagnostic:
    """Static molecular and optional elemental balance for one reaction."""

    reaction_id: str
    tracked_mass_delta_mg_per_umol_extent: float
    untracked_coproduct_mass_mg_per_umol_extent: float
    closure_error_mg_per_umol_extent: float
    elemental_balance_checked: bool
    elemental_residuals: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "elemental_residuals",
            MappingProxyType(dict(self.elemental_residuals)),
        )


@dataclass(frozen=True)
class ReactionDerivatives:
    """Sparse molar derivatives plus cumulative coproduct-mass rates."""

    species_umol_h: Mapping[str, float]
    untracked_coproduct_mg_h: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "species_umol_h",
            MappingProxyType(dict(self.species_umol_h)),
        )
        object.__setattr__(
            self,
            "untracked_coproduct_mg_h",
            MappingProxyType(dict(self.untracked_coproduct_mg_h)),
        )

    @property
    def total_untracked_coproduct_mg_h(self) -> float:
        """Derivative of a cumulative untracked-coproduct mass state."""

        return float(sum(self.untracked_coproduct_mg_h.values()))


@dataclass(frozen=True)
class MassAccountingDiagnostic:
    """Dynamic mass-rate closure for supplied reaction extent rates."""

    tracked_mass_rate_mg_h: float
    untracked_coproduct_mass_rate_mg_h: float
    closure_error_mg_h: float
    reaction_closure_error_mg_h: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "reaction_closure_error_mg_h",
            MappingProxyType(dict(self.reaction_closure_error_mg_h)),
        )


@dataclass(frozen=True)
class ReactionNetwork:
    """Validated, immutable collection of species and molar reactions."""

    species: tuple[ChemicalSpecies, ...] | SpeciesRegistry
    reactions: tuple[Reaction, ...]
    species_formulas: tuple[SpeciesFormula, ...] = ()
    mass_balance_atol_mg_per_umol: float = 1e-9
    elemental_balance_atol: float = 1e-9
    _species_by_id: Mapping[str, ChemicalSpecies] = field(
        init=False, repr=False, compare=False
    )
    _reaction_by_id: Mapping[str, Reaction] = field(
        init=False, repr=False, compare=False
    )
    _formula_by_species_id: Mapping[str, MolecularFormula] = field(
        init=False, repr=False, compare=False
    )
    _balances: tuple[ReactionBalanceDiagnostic, ...] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        species = tuple(self.species)
        reactions = tuple(self.reactions)
        formulas = tuple(self.species_formulas)
        if not species:
            raise ValueError("a reaction network requires at least one species")
        if not reactions:
            raise ValueError("a reaction network requires at least one reaction")
        if any(not isinstance(record, ChemicalSpecies) for record in species):
            raise TypeError("reaction-network species must be ChemicalSpecies records")
        if any(not isinstance(reaction, Reaction) for reaction in reactions):
            raise TypeError("reaction-network reactions must be Reaction records")
        if any(not isinstance(record, SpeciesFormula) for record in formulas):
            raise TypeError("species formulas must be SpeciesFormula records")
        species_by_id = {record.species_id: record for record in species}
        if len(species_by_id) != len(species):
            raise ValueError("reaction-network species_id values must be unique")
        reaction_by_id = {reaction.reaction_id: reaction for reaction in reactions}
        if len(reaction_by_id) != len(reactions):
            raise ValueError("reaction_id values must be unique")
        formula_by_id = {record.species_id: record.formula for record in formulas}
        if len(formula_by_id) != len(formulas):
            raise ValueError("each species may have at most one elemental formula")
        unknown_formula_species = sorted(set(formula_by_id) - set(species_by_id))
        if unknown_formula_species:
            raise ValueError(
                f"elemental formulas reference unknown species: {unknown_formula_species}"
            )
        mass_tolerance = float(self.mass_balance_atol_mg_per_umol)
        element_tolerance = float(self.elemental_balance_atol)
        if not np.isfinite(mass_tolerance) or mass_tolerance < 0.0:
            raise ValueError("mass balance tolerance must be finite and >= 0")
        if not np.isfinite(element_tolerance) or element_tolerance < 0.0:
            raise ValueError("elemental balance tolerance must be finite and >= 0")

        coproduct_definitions: dict[str, UntrackedCoproduct] = {}
        balances: list[ReactionBalanceDiagnostic] = []
        for reaction in reactions:
            balances.append(
                self._validate_reaction(
                    reaction,
                    species_by_id,
                    formula_by_id,
                    coproduct_definitions,
                    mass_tolerance,
                    element_tolerance,
                )
            )

        object.__setattr__(self, "species", species)
        object.__setattr__(self, "reactions", reactions)
        object.__setattr__(self, "species_formulas", formulas)
        object.__setattr__(self, "mass_balance_atol_mg_per_umol", mass_tolerance)
        object.__setattr__(self, "elemental_balance_atol", element_tolerance)
        object.__setattr__(self, "_species_by_id", MappingProxyType(species_by_id))
        object.__setattr__(self, "_reaction_by_id", MappingProxyType(reaction_by_id))
        object.__setattr__(self, "_formula_by_species_id", MappingProxyType(formula_by_id))
        object.__setattr__(self, "_balances", tuple(balances))

    @staticmethod
    def _validate_reaction(
        reaction: Reaction,
        species_by_id: Mapping[str, ChemicalSpecies],
        formula_by_id: Mapping[str, MolecularFormula],
        coproduct_definitions: dict[str, UntrackedCoproduct],
        mass_tolerance: float,
        element_tolerance: float,
    ) -> ReactionBalanceDiagnostic:
        term_ids = {term.species_id for term in reaction.terms}
        unknown_species = sorted(term_ids - set(species_by_id))
        if unknown_species:
            raise ValueError(
                f"reaction {reaction.reaction_id} references unknown species: "
                f"{unknown_species}"
            )
        surrogates = sorted(
            species_id
            for species_id in term_ids
            if species_by_id[species_id].mass_surrogate
        )
        if surrogates:
            raise ValueError(
                f"reaction {reaction.reaction_id} cannot use mass-surrogate species "
                f"in a molar network: {surrogates}"
            )
        for coproduct in reaction.untracked_coproducts:
            if coproduct.coproduct_id in species_by_id:
                raise ValueError(
                    f"untracked coproduct {coproduct.coproduct_id} collides with a "
                    "tracked species_id"
                )
            previous = coproduct_definitions.get(coproduct.coproduct_id)
            if previous is not None and previous != coproduct:
                raise ValueError(
                    f"inconsistent definitions for untracked coproduct "
                    f"{coproduct.coproduct_id}"
                )
            coproduct_definitions[coproduct.coproduct_id] = coproduct

        tracked_mass_delta = sum(
            term.coefficient
            * species_by_id[term.species_id].convert_amount(
                1.0, AmountUnit.UMOL, AmountUnit.MG
            )
            for term in reaction.terms
        )
        untracked_mass = sum(
            coproduct.mass_mg_per_umol_extent
            for coproduct in reaction.untracked_coproducts
        )
        closure_error = tracked_mass_delta + untracked_mass
        if abs(closure_error) > mass_tolerance:
            raise ValueError(
                f"reaction {reaction.reaction_id} is not mass balanced: "
                f"tracked delta {tracked_mass_delta:.12g} plus declared untracked "
                f"coproduct {untracked_mass:.12g} = {closure_error:.12g} "
                "mg/umol extent"
            )

        tracked_formulas = [formula_by_id.get(term.species_id) for term in reaction.terms]
        coproduct_formulas = [item.formula for item in reaction.untracked_coproducts]
        supplied_formula_count = sum(formula is not None for formula in tracked_formulas)
        supplied_formula_count += sum(formula is not None for formula in coproduct_formulas)
        molecule_count = len(tracked_formulas) + len(coproduct_formulas)
        elemental_residuals: dict[str, float] = {}
        elemental_checked = supplied_formula_count > 0
        if elemental_checked:
            if supplied_formula_count != molecule_count:
                raise ValueError(
                    f"reaction {reaction.reaction_id} has partial elemental formulas; "
                    "provide formulas for every tracked species and coproduct or none"
                )
            for term, formula in zip(reaction.terms, tracked_formulas, strict=True):
                assert formula is not None
                for element, count in formula.as_mapping().items():
                    elemental_residuals[element] = (
                        elemental_residuals.get(element, 0.0)
                        + term.coefficient * count
                    )
            for coproduct, formula in zip(
                reaction.untracked_coproducts, coproduct_formulas, strict=True
            ):
                assert formula is not None
                for element, count in formula.as_mapping().items():
                    elemental_residuals[element] = (
                        elemental_residuals.get(element, 0.0)
                        + coproduct.coefficient * count
                    )
            residual_failures = {
                element: residual
                for element, residual in elemental_residuals.items()
                if abs(residual) > element_tolerance
            }
            if residual_failures:
                raise ValueError(
                    f"reaction {reaction.reaction_id} is not element balanced: "
                    f"{residual_failures}"
                )

        return ReactionBalanceDiagnostic(
            reaction_id=reaction.reaction_id,
            tracked_mass_delta_mg_per_umol_extent=float(tracked_mass_delta),
            untracked_coproduct_mass_mg_per_umol_extent=float(untracked_mass),
            closure_error_mg_per_umol_extent=float(closure_error),
            elemental_balance_checked=elemental_checked,
            elemental_residuals=elemental_residuals,
        )

    @property
    def species_ids(self) -> tuple[str, ...]:
        return tuple(record.species_id for record in self.species)

    @property
    def reaction_ids(self) -> tuple[str, ...]:
        return tuple(reaction.reaction_id for reaction in self.reactions)

    @property
    def balance_diagnostics(self) -> tuple[ReactionBalanceDiagnostic, ...]:
        return self._balances

    def _normalise_extent_rates(
        self,
        extent_rates: Mapping[str, float] | Iterable[ReactionExtentRate],
    ) -> Mapping[str, float]:
        if isinstance(extent_rates, Mapping):
            records = tuple(
                ReactionExtentRate(reaction_id=reaction_id, extent_umol_h=value)
                for reaction_id, value in extent_rates.items()
            )
        else:
            records = tuple(extent_rates)
            if any(not isinstance(record, ReactionExtentRate) for record in records):
                raise TypeError(
                    "extent rates must be a mapping or ReactionExtentRate records"
                )
        identifiers = tuple(record.reaction_id for record in records)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("a reaction extent rate may be supplied only once")
        unknown = sorted(set(identifiers) - set(self._reaction_by_id))
        if unknown:
            raise ValueError(f"extent rates reference unknown reactions: {unknown}")
        return MappingProxyType(
            {record.reaction_id: record.extent_umol_h for record in records}
        )

    def derivatives(
        self,
        extent_rates: Mapping[str, float] | Iterable[ReactionExtentRate],
    ) -> ReactionDerivatives:
        """Compute sparse species derivatives from nonnegative extent rates.

        Returned tracked-species values have units of ``umol/h``.  Untracked
        coproduct values have units of ``mg/h`` and can be integrated as
        cumulative mass-accounting states.
        """

        rates = self._normalise_extent_rates(extent_rates)
        species_derivatives: dict[str, float] = {}
        coproduct_derivatives: dict[str, float] = {}
        for reaction in self.reactions:
            extent = rates.get(reaction.reaction_id, 0.0)
            if extent == 0.0:
                continue
            for term in reaction.terms:
                species_derivatives[term.species_id] = (
                    species_derivatives.get(term.species_id, 0.0)
                    + term.coefficient * extent
                )
            for coproduct in reaction.untracked_coproducts:
                coproduct_derivatives[coproduct.coproduct_id] = (
                    coproduct_derivatives.get(coproduct.coproduct_id, 0.0)
                    + coproduct.mass_mg_per_umol_extent * extent
                )
        species_derivatives = {
            species_id: value
            for species_id, value in species_derivatives.items()
            if value != 0.0
        }
        coproduct_derivatives = {
            coproduct_id: value
            for coproduct_id, value in coproduct_derivatives.items()
            if value != 0.0
        }
        return ReactionDerivatives(species_derivatives, coproduct_derivatives)

    def mass_accounting(
        self,
        extent_rates: Mapping[str, float] | Iterable[ReactionExtentRate],
    ) -> MassAccountingDiagnostic:
        """Report mass-rate closure for one set of reaction extent rates."""

        rates = self._normalise_extent_rates(extent_rates)
        derivatives = self.derivatives(rates)
        tracked_mass_rate = sum(
            self._species_by_id[species_id].convert_amount(
                rate, AmountUnit.UMOL, AmountUnit.MG
            )
            for species_id, rate in derivatives.species_umol_h.items()
        )
        untracked_mass_rate = derivatives.total_untracked_coproduct_mg_h
        balance_by_reaction = {
            balance.reaction_id: (
                balance.closure_error_mg_per_umol_extent
                * rates.get(balance.reaction_id, 0.0)
            )
            for balance in self._balances
            if rates.get(balance.reaction_id, 0.0) != 0.0
        }
        return MassAccountingDiagnostic(
            tracked_mass_rate_mg_h=float(tracked_mass_rate),
            untracked_coproduct_mass_rate_mg_h=float(untracked_mass_rate),
            closure_error_mg_h=float(tracked_mass_rate + untracked_mass_rate),
            reaction_closure_error_mg_h=balance_by_reaction,
        )
