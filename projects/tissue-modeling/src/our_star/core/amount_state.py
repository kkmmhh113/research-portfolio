"""Micromole amount states for multi-species PBPK systems.

The v0.3 solver deliberately keeps its historical milligram state vector.
This module is a parallel, opt-in state contract for reactions between named
species with different molecular weights.  Every dynamic value is expressed
in micromoles; mass conversion happens only at dose and observation/accounting
boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .species import AmountUnit, SpeciesRegistry


@dataclass(frozen=True)
class AmountStateSpec:
    """Metadata for one scalar micromole state.

    ``accounting_only`` states hold cumulative external reaction participants
    (for example, a glucuronyl group imported from an endogenous cofactor).
    They are part of the solver vector so the augmented molecular ledger is
    reproducible, but are excluded from drug-moiety and tracked-drug mass.
    """

    name: str
    species_id: str
    compartment_id: str
    unit: AmountUnit = AmountUnit.UMOL
    sink: bool = False
    accounting_only: bool = False

    def __post_init__(self) -> None:
        for label in ("name", "species_id", "compartment_id"):
            if not str(getattr(self, label)).strip():
                raise ValueError(f"{label} is required")
        unit = AmountUnit(self.unit)
        if unit != AmountUnit.UMOL:
            raise ValueError("multi-species amount states must use umol")
        object.__setattr__(self, "unit", unit)


@dataclass(frozen=True, order=True)
class FluxKey:
    """Species-aware connection between independently assembled modules."""

    connection_id: str
    species_id: str

    def __post_init__(self) -> None:
        if not self.connection_id.strip():
            raise ValueError("connection_id is required")
        if not self.species_id.strip():
            raise ValueError("species_id is required")


@dataclass(frozen=True)
class SpeciesDoseEvent:
    """Dose event whose basis species and amount unit are explicit."""

    time_h: float
    species_id: str
    amount: float
    unit: AmountUnit | str = AmountUnit.MG
    route: str = "oral"
    dose_basis_id: str = "active_moiety"

    def __post_init__(self) -> None:
        time_h = float(self.time_h)
        amount = float(self.amount)
        if not np.isfinite(time_h) or time_h < 0.0:
            raise ValueError("dose time must be finite and >= 0")
        if not np.isfinite(amount) or amount <= 0.0:
            raise ValueError("dose amount must be finite and > 0")
        if not self.species_id.strip():
            raise ValueError("dose species_id is required")
        if self.route not in {"oral", "iv_bolus"}:
            raise ValueError("dose route must be 'oral' or 'iv_bolus'")
        if not self.dose_basis_id.strip():
            raise ValueError("dose_basis_id is required")
        object.__setattr__(self, "time_h", time_h)
        object.__setattr__(self, "amount", amount)
        object.__setattr__(self, "unit", AmountUnit(self.unit))


class AmountStateRegistry:
    """Immutable-by-convention mapping from names to µmol solver positions."""

    def __init__(
        self,
        states: Iterable[AmountStateSpec],
        species: SpeciesRegistry,
    ) -> None:
        records = tuple(states)
        if not records:
            raise ValueError("at least one amount state is required")
        if any(not isinstance(record, AmountStateSpec) for record in records):
            raise TypeError("amount states must be AmountStateSpec records")
        names = tuple(record.name for record in records)
        if len(set(names)) != len(names):
            raise ValueError("amount state names must be unique")
        unknown = sorted(
            {record.species_id for record in records if record.species_id not in species}
        )
        if unknown:
            raise ValueError(f"amount states reference unknown species: {unknown}")
        self._records = records
        self._indices = {name: index for index, name in enumerate(names)}
        self.species = species

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self):
        return iter(self._records)

    def index(self, state_name: str) -> int:
        try:
            return self._indices[state_name]
        except KeyError as error:
            raise KeyError(f"unregistered amount state: {state_name}") from error

    def spec(self, state_name: str) -> AmountStateSpec:
        return self._records[self.index(state_name)]

    def zeros(self) -> np.ndarray:
        return np.zeros(len(self), dtype=np.float64)

    def validate_vector(self, vector: np.ndarray) -> np.ndarray:
        values = np.asarray(vector, dtype=np.float64)
        if values.shape != (len(self),):
            raise ValueError(f"expected state shape {(len(self),)}, got {values.shape}")
        if not np.all(np.isfinite(values)):
            raise FloatingPointError("amount state contains non-finite values")
        return values

    def amount_umol(self, vector: np.ndarray, state_name: str) -> float:
        values = self.validate_vector(vector)
        return float(values[self.index(state_name)])

    def names_for_species(
        self,
        species_id: str,
        *,
        include_accounting: bool = False,
    ) -> tuple[str, ...]:
        if species_id not in self.species:
            raise KeyError(f"unregistered chemical species: {species_id}")
        return tuple(
            record.name
            for record in self._records
            if record.species_id == species_id
            and (include_accounting or not record.accounting_only)
        )

    def names_for_compartment(self, compartment_id: str) -> tuple[str, ...]:
        return tuple(
            record.name
            for record in self._records
            if record.compartment_id == compartment_id
        )

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(record.name for record in self._records)

    @property
    def sinks(self) -> tuple[str, ...]:
        return tuple(record.name for record in self._records if record.sink)

    @property
    def physical_names(self) -> tuple[str, ...]:
        return tuple(
            record.name for record in self._records if not record.accounting_only
        )

    @property
    def accounting_names(self) -> tuple[str, ...]:
        return tuple(
            record.name for record in self._records if record.accounting_only
        )
