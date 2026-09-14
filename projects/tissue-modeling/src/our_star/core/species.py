"""Chemical-species identities and explicit amount-unit conversion.

The first PBPK milestone remains mass based in order to preserve its validated
mass-balance behaviour.  A molecular registry is introduced now so future
metabolites can be represented in molar units without silently adding unlike
quantities.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Iterable

import numpy as np


class AmountUnit(StrEnum):
    """Supported amount units for model states."""

    MG = "mg"
    UMOL = "umol"


@dataclass(frozen=True)
class ChemicalSpecies:
    """One conserved chemical species or an explicitly labelled mass pool."""

    species_id: str
    name: str
    molecular_weight_g_mol: float
    structure: str | None = None
    mass_surrogate: bool = False

    def __post_init__(self) -> None:
        if not self.species_id.strip():
            raise ValueError("species_id is required")
        if not self.name.strip():
            raise ValueError("species name is required")
        if not np.isfinite(self.molecular_weight_g_mol) or self.molecular_weight_g_mol <= 0:
            raise ValueError("molecular_weight_g_mol must be finite and > 0")
        if self.structure is not None and not self.structure.strip():
            raise ValueError("structure cannot be blank when supplied")

    def convert_amount(
        self,
        value: float | np.ndarray,
        from_unit: AmountUnit | str,
        to_unit: AmountUnit | str,
    ) -> float | np.ndarray:
        """Convert mass and molar amount using this species' molecular weight."""

        source = AmountUnit(from_unit)
        target = AmountUnit(to_unit)
        values = np.asarray(value, dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError("amount must be finite")
        if source == target:
            converted = values.copy()
        elif source == AmountUnit.MG and target == AmountUnit.UMOL:
            converted = values * 1000.0 / self.molecular_weight_g_mol
        elif source == AmountUnit.UMOL and target == AmountUnit.MG:
            converted = values * self.molecular_weight_g_mol / 1000.0
        else:  # pragma: no cover - exhaustive guard for future enum members
            raise ValueError(f"unsupported conversion: {source} -> {target}")
        return float(converted) if converted.ndim == 0 else converted


class SpeciesRegistry:
    """Immutable-by-convention lookup of unique chemical species."""

    def __init__(self, species: Iterable[ChemicalSpecies]) -> None:
        records = tuple(species)
        by_id = {record.species_id: record for record in records}
        if len(by_id) != len(records):
            raise ValueError("species_id values must be unique")
        if not records:
            raise ValueError("at least one chemical species is required")
        self._records = records
        self._by_id = by_id

    def __iter__(self):
        return iter(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, species_id: str) -> bool:
        return species_id in self._by_id

    def __getitem__(self, species_id: str) -> ChemicalSpecies:
        try:
            return self._by_id[species_id]
        except KeyError as error:
            raise KeyError(f"unregistered chemical species: {species_id}") from error

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(record.species_id for record in self._records)
