"""Open-system reaction chemistry and dual PBPK amount ledgers.

Drug conjugation is an open chemical system: a glucuronide can weigh more than
the administered parent because an endogenous cofactor supplied additional
atoms.  This module therefore keeps two independent invariants:

* a drug-moiety ledger, reported as parent-equivalent mass; and
* an augmented molecular ledger that includes signed external imports/exports.

The historical v0.3 :mod:`our_star.chemistry.reactions` API remains unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Iterable, Mapping

import numpy as np

from ..core.amount_state import AmountStateRegistry
from ..core.species import AmountUnit, ChemicalSpecies, SpeciesRegistry
from .reactions import MolecularFormula, SpeciesFormula, StoichiometricTerm


# Conventional atomic weights, used only to make configured formula-derived
# molecular weights internally self-consistent.  They are not fitted values.
_ATOMIC_WEIGHT_G_MOL = MappingProxyType(
    {
        "C": 12.011,
        "H": 1.008,
        "N": 14.007,
        "O": 15.999,
        "P": 30.973761998,
        "S": 32.06,
    }
)


def formula_molecular_weight_g_mol(formula: MolecularFormula) -> float:
    """Return a deterministic average molecular weight for a supported formula."""

    unknown = sorted(set(formula.as_mapping()) - set(_ATOMIC_WEIGHT_G_MOL))
    if unknown:
        raise ValueError(f"no configured atomic weight for elements: {unknown}")
    return float(
        sum(
            _ATOMIC_WEIGHT_G_MOL[element] * count
            for element, count in formula.as_mapping().items()
        )
    )


class ExternalParticipantRole(StrEnum):
    COFACTOR = "cofactor"
    NET_TRANSFER_GROUP = "net_transfer_group"
    COPRODUCT = "coproduct"
    ENDOGENOUS_CARRIER = "endogenous_carrier"


@dataclass(frozen=True)
class ExternalParticipant:
    """One unmodeled reactant or product in an open-system reaction.

    The sign follows ordinary stoichiometry: negative is imported/consumed by
    the tracked reaction and positive is exported/produced.
    """

    participant_id: str
    name: str
    coefficient: float
    molecular_weight_g_mol: float
    role: ExternalParticipantRole | str
    formula: MolecularFormula | None = None
    provenance: str = ""

    def __post_init__(self) -> None:
        if not self.participant_id.strip():
            raise ValueError("external participant_id is required")
        if not self.name.strip():
            raise ValueError("external participant name is required")
        coefficient = float(self.coefficient)
        molecular_weight = float(self.molecular_weight_g_mol)
        if not np.isfinite(coefficient) or coefficient == 0.0:
            raise ValueError("external participant coefficient must be finite and nonzero")
        if not np.isfinite(molecular_weight) or molecular_weight <= 0.0:
            raise ValueError("external participant molecular weight must be finite and > 0")
        if self.formula is not None and not isinstance(self.formula, MolecularFormula):
            raise TypeError("external participant formula must be MolecularFormula")
        if not self.provenance.strip():
            raise ValueError("external participant provenance is required")
        object.__setattr__(self, "coefficient", coefficient)
        object.__setattr__(self, "molecular_weight_g_mol", molecular_weight)
        object.__setattr__(self, "role", ExternalParticipantRole(self.role))

    @property
    def signed_mass_mg_per_umol_extent(self) -> float:
        return self.coefficient * self.molecular_weight_g_mol / 1000.0


@dataclass(frozen=True)
class OpenReaction:
    """Tracked stoichiometry plus signed untracked reaction participants."""

    reaction_id: str
    terms: tuple[StoichiometricTerm, ...]
    external_participants: tuple[ExternalParticipant, ...] = ()
    description: str = ""

    def __post_init__(self) -> None:
        if not self.reaction_id.strip():
            raise ValueError("open reaction_id is required")
        terms = tuple(self.terms)
        participants = tuple(self.external_participants)
        if not terms:
            raise ValueError("an open reaction requires tracked terms")
        if any(not isinstance(term, StoichiometricTerm) for term in terms):
            raise TypeError("open reaction terms must be StoichiometricTerm records")
        if any(not isinstance(item, ExternalParticipant) for item in participants):
            raise TypeError(
                "external participants must be ExternalParticipant records"
            )
        species_ids = tuple(term.species_id for term in terms)
        participant_ids = tuple(item.participant_id for item in participants)
        if len(set(species_ids)) != len(species_ids):
            raise ValueError("a tracked species may appear only once per open reaction")
        if len(set(participant_ids)) != len(participant_ids):
            raise ValueError(
                "an external participant may appear only once per open reaction"
            )
        all_coefficients = tuple(term.coefficient for term in terms) + tuple(
            item.coefficient for item in participants
        )
        if not any(value < 0.0 for value in all_coefficients):
            raise ValueError("an open reaction requires at least one reactant")
        if not any(value > 0.0 for value in all_coefficients):
            raise ValueError("an open reaction requires at least one product")
        if not self.description.strip():
            raise ValueError("open reaction description is required")
        object.__setattr__(self, "terms", terms)
        object.__setattr__(self, "external_participants", participants)


@dataclass(frozen=True)
class OpenReactionBalance:
    reaction_id: str
    tracked_mass_delta_mg_per_umol: float
    external_mass_delta_mg_per_umol: float
    closure_error_mg_per_umol: float
    elemental_balance_checked: bool
    elemental_residuals: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "elemental_residuals",
            MappingProxyType(dict(self.elemental_residuals)),
        )


@dataclass(frozen=True)
class OpenReactionDerivatives:
    species_umol_h: Mapping[str, float]
    external_participant_umol_h: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "species_umol_h", MappingProxyType(dict(self.species_umol_h))
        )
        object.__setattr__(
            self,
            "external_participant_umol_h",
            MappingProxyType(dict(self.external_participant_umol_h)),
        )

    @property
    def external_import_umol_h(self) -> Mapping[str, float]:
        return MappingProxyType(
            {
                participant_id: -rate
                for participant_id, rate in self.external_participant_umol_h.items()
                if rate < 0.0
            }
        )

    @property
    def external_export_umol_h(self) -> Mapping[str, float]:
        return MappingProxyType(
            {
                participant_id: rate
                for participant_id, rate in self.external_participant_umol_h.items()
                if rate > 0.0
            }
        )


@dataclass(frozen=True)
class OpenMassRateDiagnostic:
    tracked_mass_rate_mg_h: float
    external_mass_rate_mg_h: float
    closure_error_mg_h: float


class OpenReactionNetwork:
    """Validated chemical network that permits signed endogenous participants."""

    def __init__(
        self,
        species: Iterable[ChemicalSpecies] | SpeciesRegistry,
        reactions: Iterable[OpenReaction],
        *,
        species_formulas: Iterable[SpeciesFormula] = (),
        mass_balance_atol_mg_per_umol: float = 1.0e-9,
        elemental_balance_atol: float = 1.0e-9,
    ) -> None:
        records = tuple(species)
        reaction_records = tuple(reactions)
        formulas = tuple(species_formulas)
        if not records or not reaction_records:
            raise ValueError("open reaction network requires species and reactions")
        self.species = SpeciesRegistry(records)
        if any(not isinstance(item, OpenReaction) for item in reaction_records):
            raise TypeError("open network reactions must be OpenReaction records")
        reaction_ids = tuple(item.reaction_id for item in reaction_records)
        if len(set(reaction_ids)) != len(reaction_ids):
            raise ValueError("open reaction_id values must be unique")
        formula_by_species = {item.species_id: item.formula for item in formulas}
        if len(formula_by_species) != len(formulas):
            raise ValueError("each species may have at most one formula")
        unknown_formula_species = sorted(set(formula_by_species) - set(self.species.ids))
        if unknown_formula_species:
            raise ValueError(
                f"formulas reference unknown open-network species: {unknown_formula_species}"
            )
        mass_tolerance = float(mass_balance_atol_mg_per_umol)
        element_tolerance = float(elemental_balance_atol)
        if not np.isfinite(mass_tolerance) or mass_tolerance < 0.0:
            raise ValueError("mass balance tolerance must be finite and >= 0")
        if not np.isfinite(element_tolerance) or element_tolerance < 0.0:
            raise ValueError("elemental balance tolerance must be finite and >= 0")

        participants: dict[str, ExternalParticipant] = {}
        balances: list[OpenReactionBalance] = []
        for reaction in reaction_records:
            balances.append(
                self._validate_reaction(
                    reaction,
                    formula_by_species,
                    participants,
                    mass_tolerance,
                    element_tolerance,
                )
            )
        self.reactions = reaction_records
        self._reaction_by_id = MappingProxyType(
            {reaction.reaction_id: reaction for reaction in reaction_records}
        )
        self.external_participants = MappingProxyType(participants)
        self.balance_diagnostics = tuple(balances)

    def _validate_reaction(
        self,
        reaction: OpenReaction,
        formula_by_species: Mapping[str, MolecularFormula],
        participants: dict[str, ExternalParticipant],
        mass_tolerance: float,
        element_tolerance: float,
    ) -> OpenReactionBalance:
        unknown = sorted(
            {term.species_id for term in reaction.terms} - set(self.species.ids)
        )
        if unknown:
            raise ValueError(
                f"open reaction {reaction.reaction_id} references unknown species: {unknown}"
            )
        surrogates = sorted(
            term.species_id
            for term in reaction.terms
            if self.species[term.species_id].mass_surrogate
        )
        if surrogates:
            raise ValueError(
                f"open reaction {reaction.reaction_id} cannot use mass surrogates: "
                f"{surrogates}"
            )
        for participant in reaction.external_participants:
            if participant.participant_id in self.species:
                raise ValueError(
                    f"external participant {participant.participant_id} collides "
                    "with tracked species"
                )
            previous = participants.get(participant.participant_id)
            if previous is not None:
                same_identity = (
                    previous.name == participant.name
                    and previous.molecular_weight_g_mol
                    == participant.molecular_weight_g_mol
                    and previous.role == participant.role
                    and previous.formula == participant.formula
                    and previous.provenance == participant.provenance
                )
                if not same_identity:
                    raise ValueError(
                        f"inconsistent external participant definition: "
                        f"{participant.participant_id}"
                    )
            else:
                participants[participant.participant_id] = participant

        tracked_mass = sum(
            term.coefficient
            * self.species[term.species_id].convert_amount(
                1.0, AmountUnit.UMOL, AmountUnit.MG
            )
            for term in reaction.terms
        )
        external_mass = sum(
            item.signed_mass_mg_per_umol_extent
            for item in reaction.external_participants
        )
        closure = float(tracked_mass + external_mass)
        if abs(closure) > mass_tolerance:
            raise ValueError(
                f"open reaction {reaction.reaction_id} is not mass balanced: "
                f"tracked {tracked_mass:.12g} + external {external_mass:.12g} "
                f"= {closure:.12g} mg/umol"
            )

        tracked_formulas = [
            formula_by_species.get(term.species_id) for term in reaction.terms
        ]
        external_formulas = [item.formula for item in reaction.external_participants]
        supplied = sum(item is not None for item in tracked_formulas + external_formulas)
        molecule_count = len(tracked_formulas) + len(external_formulas)
        residuals: dict[str, float] = {}
        checked = supplied > 0
        if checked:
            if supplied != molecule_count:
                raise ValueError(
                    f"open reaction {reaction.reaction_id} has partial elemental formulas"
                )
            for term, formula in zip(reaction.terms, tracked_formulas, strict=True):
                assert formula is not None
                for element, count in formula.as_mapping().items():
                    residuals[element] = residuals.get(element, 0.0) + term.coefficient * count
            for participant, formula in zip(
                reaction.external_participants, external_formulas, strict=True
            ):
                assert formula is not None
                for element, count in formula.as_mapping().items():
                    residuals[element] = (
                        residuals.get(element, 0.0)
                        + participant.coefficient * count
                    )
            failures = {
                element: value
                for element, value in residuals.items()
                if abs(value) > element_tolerance
            }
            if failures:
                raise ValueError(
                    f"open reaction {reaction.reaction_id} is not element balanced: "
                    f"{failures}"
                )
        return OpenReactionBalance(
            reaction_id=reaction.reaction_id,
            tracked_mass_delta_mg_per_umol=float(tracked_mass),
            external_mass_delta_mg_per_umol=float(external_mass),
            closure_error_mg_per_umol=closure,
            elemental_balance_checked=checked,
            elemental_residuals=residuals,
        )

    def _rates(self, extent_rates: Mapping[str, float]) -> Mapping[str, float]:
        unknown = sorted(set(extent_rates) - set(self._reaction_by_id))
        if unknown:
            raise ValueError(f"extent rates reference unknown open reactions: {unknown}")
        rates = {reaction_id: float(value) for reaction_id, value in extent_rates.items()}
        invalid = {
            key: value
            for key, value in rates.items()
            if not np.isfinite(value) or value < 0.0
        }
        if invalid:
            raise ValueError(
                f"open reaction extent rates must be finite and >= 0: {invalid}"
            )
        return MappingProxyType(rates)

    def derivatives(self, extent_rates: Mapping[str, float]) -> OpenReactionDerivatives:
        rates = self._rates(extent_rates)
        species_rates: dict[str, float] = {}
        external_rates: dict[str, float] = {}
        for reaction in self.reactions:
            extent = rates.get(reaction.reaction_id, 0.0)
            if extent == 0.0:
                continue
            for term in reaction.terms:
                species_rates[term.species_id] = (
                    species_rates.get(term.species_id, 0.0)
                    + term.coefficient * extent
                )
            for participant in reaction.external_participants:
                external_rates[participant.participant_id] = (
                    external_rates.get(participant.participant_id, 0.0)
                    + participant.coefficient * extent
                )
        return OpenReactionDerivatives(
            species_umol_h={key: value for key, value in species_rates.items() if value},
            external_participant_umol_h={
                key: value for key, value in external_rates.items() if value
            },
        )

    def mass_rate_diagnostic(
        self, extent_rates: Mapping[str, float]
    ) -> OpenMassRateDiagnostic:
        derivatives = self.derivatives(extent_rates)
        tracked_mass = sum(
            self.species[species_id].convert_amount(
                rate, AmountUnit.UMOL, AmountUnit.MG
            )
            for species_id, rate in derivatives.species_umol_h.items()
        )
        external_mass = sum(
            self.external_participants[participant_id].molecular_weight_g_mol
            * rate
            / 1000.0
            for participant_id, rate in derivatives.external_participant_umol_h.items()
        )
        return OpenMassRateDiagnostic(
            tracked_mass_rate_mg_h=float(tracked_mass),
            external_mass_rate_mg_h=float(external_mass),
            closure_error_mg_h=float(tracked_mass + external_mass),
        )


@dataclass(frozen=True)
class SpeciesMoiety:
    """Conserved-moiety count carried by one molecule of a tracked species."""

    species_id: str
    moieties: Mapping[str, float]

    def __post_init__(self) -> None:
        if not self.species_id.strip():
            raise ValueError("moiety species_id is required")
        values = {str(key): float(value) for key, value in self.moieties.items()}
        if not values or any(
            not key.strip() or not np.isfinite(value) or value < 0.0
            for key, value in values.items()
        ):
            raise ValueError("moiety counts require names and finite values >= 0")
        object.__setattr__(self, "moieties", MappingProxyType(values))


@dataclass(frozen=True)
class DualLedgerReport:
    moiety_umol: Mapping[str, float]
    drug_equivalent_mass_mg: float
    tracked_molecular_mass_mg: float
    external_import_mass_mg: float
    external_export_mass_mg: float
    augmented_accounted_mass_mg: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "moiety_umol", MappingProxyType(dict(self.moiety_umol)))


class DualAmountLedger:
    """Compute parent-equivalent and open-system molecular accounting."""

    def __init__(
        self,
        states: AmountStateRegistry,
        species_moieties: Iterable[SpeciesMoiety],
        *,
        conserved_moiety_id: str,
        reference_species_id: str,
        external_participants: Mapping[str, ExternalParticipant],
        external_import_states: Mapping[str, str],
        external_export_states: Mapping[str, str],
    ) -> None:
        if not conserved_moiety_id.strip():
            raise ValueError("conserved_moiety_id is required")
        if reference_species_id not in states.species:
            raise ValueError("reference species is not registered")
        by_species = {item.species_id: item for item in species_moieties}
        physical_species = {
            states.spec(name).species_id for name in states.physical_names
        }
        missing = sorted(physical_species - set(by_species))
        if missing:
            raise ValueError(f"physical species lack moiety definitions: {missing}")
        if any(conserved_moiety_id not in item.moieties for item in by_species.values()):
            raise ValueError(
                f"all physical species must define moiety {conserved_moiety_id}"
            )
        unknown_participants = sorted(
            (set(external_import_states) | set(external_export_states))
            - set(external_participants)
        )
        if unknown_participants:
            raise ValueError(
                f"ledger states reference unknown external participants: "
                f"{unknown_participants}"
            )
        accounting_names = set(states.accounting_names)
        for mapping_name, mapping in (
            ("external import", external_import_states),
            ("external export", external_export_states),
        ):
            invalid = sorted(set(mapping.values()) - accounting_names)
            if invalid:
                raise ValueError(
                    f"{mapping_name} states must be registered accounting-only states: "
                    f"{invalid}"
                )
        self.states = states
        self._moieties = MappingProxyType(by_species)
        self.conserved_moiety_id = conserved_moiety_id
        self.reference_species_id = reference_species_id
        self.external_participants = MappingProxyType(dict(external_participants))
        self.external_import_states = MappingProxyType(dict(external_import_states))
        self.external_export_states = MappingProxyType(dict(external_export_states))

    def report(self, vector: np.ndarray) -> DualLedgerReport:
        values = self.states.validate_vector(vector)
        moiety_totals: dict[str, float] = {}
        tracked_mass = 0.0
        for state_name in self.states.physical_names:
            spec = self.states.spec(state_name)
            amount_umol = max(float(values[self.states.index(state_name)]), 0.0)
            tracked_mass += float(
                self.states.species[spec.species_id].convert_amount(
                    amount_umol, AmountUnit.UMOL, AmountUnit.MG
                )
            )
            for moiety_id, coefficient in self._moieties[spec.species_id].moieties.items():
                moiety_totals[moiety_id] = (
                    moiety_totals.get(moiety_id, 0.0)
                    + coefficient * amount_umol
                )

        external_import_mass = sum(
            max(float(values[self.states.index(state_name)]), 0.0)
            * self.external_participants[participant_id].molecular_weight_g_mol
            / 1000.0
            for participant_id, state_name in self.external_import_states.items()
        )
        external_export_mass = sum(
            max(float(values[self.states.index(state_name)]), 0.0)
            * self.external_participants[participant_id].molecular_weight_g_mol
            / 1000.0
            for participant_id, state_name in self.external_export_states.items()
        )
        conserved_umol = moiety_totals.get(self.conserved_moiety_id, 0.0)
        reference_mw = self.states.species[
            self.reference_species_id
        ].molecular_weight_g_mol
        drug_equivalent_mass = conserved_umol * reference_mw / 1000.0
        augmented_mass = tracked_mass + external_export_mass - external_import_mass
        return DualLedgerReport(
            moiety_umol=moiety_totals,
            drug_equivalent_mass_mg=float(drug_equivalent_mass),
            tracked_molecular_mass_mg=float(tracked_mass),
            external_import_mass_mg=float(external_import_mass),
            external_export_mass_mg=float(external_export_mass),
            augmented_accounted_mass_mg=float(augmented_mass),
        )
