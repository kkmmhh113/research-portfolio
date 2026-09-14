"""Named state-vector registry used by every organ module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .species import AmountUnit, SpeciesRegistry


@dataclass(frozen=True)
class StateSpec:
    """Metadata for one scalar amount state."""

    name: str
    species_id: str
    compartment: str
    unit: AmountUnit = AmountUnit.MG
    sink: bool = False

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("state name is required")
        if not self.species_id.strip():
            raise ValueError("state species_id is required")
        if not self.compartment.strip():
            raise ValueError("state compartment is required")
        object.__setattr__(self, "unit", AmountUnit(self.unit))


class StateRegistry:
    """Maps stable state names to dense solver-vector positions."""

    def __init__(
        self,
        states: Iterable[StateSpec],
        species: SpeciesRegistry,
    ) -> None:
        records = tuple(states)
        names = tuple(record.name for record in records)
        if not records:
            raise ValueError("at least one state is required")
        if len(set(names)) != len(names):
            raise ValueError("state names must be unique")
        unknown = sorted({record.species_id for record in records if record.species_id not in species})
        if unknown:
            raise ValueError(f"states reference unknown species: {unknown}")
        non_mass = [record.name for record in records if record.unit != AmountUnit.MG]
        if non_mass:
            raise ValueError(
                "the current mass-conserving solver requires mg states; "
                f"convert at module boundaries: {non_mass}"
            )
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
            raise KeyError(f"unregistered model state: {state_name}") from error

    def spec(self, state_name: str) -> StateSpec:
        return self._records[self.index(state_name)]

    def zeros(self) -> np.ndarray:
        return np.zeros(len(self), dtype=np.float64)

    def amount(self, vector: np.ndarray, state_name: str) -> float:
        values = np.asarray(vector, dtype=np.float64)
        if values.shape != (len(self),):
            raise ValueError(f"expected state shape {(len(self),)}, got {values.shape}")
        return float(values[self.index(state_name)])

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(record.name for record in self._records)

    @property
    def sinks(self) -> tuple[str, ...]:
        return tuple(record.name for record in self._records if record.sink)
