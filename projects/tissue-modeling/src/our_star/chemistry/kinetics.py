"""Small, unit-explicit kinetic laws for reaction extent rates."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Protocol, Sequence

import numpy as np


def _finite_nonnegative(value: float, label: str) -> float:
    converted = float(value)
    if not np.isfinite(converted) or converted < 0.0:
        raise ValueError(f"{label} must be finite and >= 0")
    return converted


class ConcentrationKineticLaw(Protocol):
    """Rate law mapping µmol/L to a µmol/h reaction extent."""

    def rate_umol_h(self, concentration_umol_l: float) -> float: ...


@dataclass(frozen=True)
class MichaelisMentenKinetics:
    """Capacity-limited extent rate with explicit µmol units."""

    vmax_umol_h: float
    km_umol_l: float

    def __post_init__(self) -> None:
        vmax = _finite_nonnegative(self.vmax_umol_h, "vmax_umol_h")
        km = float(self.km_umol_l)
        if not np.isfinite(km) or km <= 0.0:
            raise ValueError("km_umol_l must be finite and > 0")
        object.__setattr__(self, "vmax_umol_h", vmax)
        object.__setattr__(self, "km_umol_l", km)

    def rate_umol_h(self, concentration_umol_l: float) -> float:
        concentration = _finite_nonnegative(
            concentration_umol_l, "concentration_umol_l"
        )
        if concentration == 0.0 or self.vmax_umol_h == 0.0:
            return 0.0
        return float(
            self.vmax_umol_h
            * concentration
            / (self.km_umol_l + concentration)
        )


@dataclass(frozen=True)
class LinearClearanceKinetics:
    """First-order concentration clearance expressed as L/h × µmol/L."""

    clearance_l_h: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "clearance_l_h",
            _finite_nonnegative(self.clearance_l_h, "clearance_l_h"),
        )

    def rate_umol_h(self, concentration_umol_l: float) -> float:
        concentration = _finite_nonnegative(
            concentration_umol_l, "concentration_umol_l"
        )
        return float(self.clearance_l_h * concentration)


@dataclass(frozen=True)
class ScaledKineticPathway:
    """Named reaction law multiplied by a named activity factor."""

    reaction_id: str
    mechanism_id: str
    law: ConcentrationKineticLaw
    reference_scale: float = 1.0

    def __post_init__(self) -> None:
        if not self.reaction_id.strip():
            raise ValueError("kinetic pathway reaction_id is required")
        if not self.mechanism_id.strip():
            raise ValueError("kinetic pathway mechanism_id is required")
        scale = _finite_nonnegative(self.reference_scale, "reference_scale")
        if not callable(getattr(self.law, "rate_umol_h", None)):
            raise TypeError("kinetic pathway law must implement rate_umol_h")
        object.__setattr__(self, "reference_scale", scale)

    def rate_umol_h(
        self,
        concentration_umol_l: float,
        *,
        mechanism_factor: float = 1.0,
        shared_scale: float = 1.0,
    ) -> float:
        factor = _finite_nonnegative(mechanism_factor, "mechanism_factor")
        shared = _finite_nonnegative(shared_scale, "shared_scale")
        rate = (
            self.law.rate_umol_h(concentration_umol_l)
            * self.reference_scale
            * factor
            * shared
        )
        if not np.isfinite(rate) or rate < 0.0:  # pragma: no cover - custom-law guard
            raise FloatingPointError("kinetic law produced invalid reaction extent")
        return float(rate)


def evaluate_competing_pathways(
    concentration_umol_l: float,
    pathways: Sequence[ScaledKineticPathway],
    *,
    mechanism_factors: Mapping[str, float] | None = None,
    shared_scale: float = 1.0,
) -> Mapping[str, float]:
    """Evaluate pathways consuming the same nonnegative concentration pool.

    The function intentionally returns independent instantaneous extent rates;
    competition occurs because their stoichiometric derivatives are summed on
    the same parent state.  No pathway fraction is frozen over concentration.
    """

    concentration = _finite_nonnegative(
        concentration_umol_l, "concentration_umol_l"
    )
    records = tuple(pathways)
    if not records:
        raise ValueError("at least one competing pathway is required")
    reaction_ids = tuple(item.reaction_id for item in records)
    if len(set(reaction_ids)) != len(reaction_ids):
        raise ValueError("competing pathway reaction_id values must be unique")
    factors = dict(mechanism_factors or {})
    rates = {
        pathway.reaction_id: pathway.rate_umol_h(
            concentration,
            mechanism_factor=factors.get(pathway.mechanism_id, 1.0),
            shared_scale=shared_scale,
        )
        for pathway in records
    }
    return MappingProxyType(rates)
