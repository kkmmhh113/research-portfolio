"""Mechanistic filtration and active-secretion fluxes in molar units."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


def _nonnegative(value: float, label: str) -> float:
    converted = float(value)
    if not np.isfinite(converted) or converted < 0.0:
        raise ValueError(f"{label} must be finite and >= 0")
    return converted


def _positive(value: float, label: str) -> float:
    converted = float(value)
    if not np.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{label} must be finite and > 0")
    return converted


class SecretionKinetics(Protocol):
    """Protocol for secretion laws acting on unbound concentration."""

    def rate_umol_h(
        self,
        unbound_concentration_umol_l: float,
        *,
        activity_factor: float = 1.0,
    ) -> float: ...


@dataclass(frozen=True)
class LinearUnboundSecretion:
    clearance_unbound_l_h: float
    provenance: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "clearance_unbound_l_h",
            _nonnegative(self.clearance_unbound_l_h, "clearance_unbound_l_h"),
        )
        if not self.provenance.strip():
            raise ValueError("secretion provenance is required")

    def rate_umol_h(
        self,
        unbound_concentration_umol_l: float,
        *,
        activity_factor: float = 1.0,
    ) -> float:
        concentration = _nonnegative(
            unbound_concentration_umol_l,
            "unbound_concentration_umol_l",
        )
        activity = _nonnegative(activity_factor, "activity_factor")
        return self.clearance_unbound_l_h * concentration * activity


@dataclass(frozen=True)
class SaturableUnboundSecretion:
    vmax_umol_h: float
    km_umol_l: float
    provenance: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "vmax_umol_h", _positive(self.vmax_umol_h, "vmax_umol_h"))
        object.__setattr__(self, "km_umol_l", _positive(self.km_umol_l, "km_umol_l"))
        if not self.provenance.strip():
            raise ValueError("secretion provenance is required")

    def rate_umol_h(
        self,
        unbound_concentration_umol_l: float,
        *,
        activity_factor: float = 1.0,
    ) -> float:
        concentration = _nonnegative(
            unbound_concentration_umol_l,
            "unbound_concentration_umol_l",
        )
        activity = _nonnegative(activity_factor, "activity_factor")
        if concentration == 0.0:
            return 0.0
        return (
            activity
            * self.vmax_umol_h
            * concentration
            / (self.km_umol_l + concentration)
        )


@dataclass(frozen=True)
class RenalDispositionFluxes:
    filtration_umol_h: float
    secretion_umol_h: float

    @property
    def total_umol_h(self) -> float:
        return self.filtration_umol_h + self.secretion_umol_h


@dataclass(frozen=True)
class EnalaprilatRenalDisposition:
    """Split enalaprilat elimination into filtration and secretion."""

    fraction_unbound: float
    gfr_l_h: float
    secretion: SecretionKinetics | None
    provenance: str

    def __post_init__(self) -> None:
        fraction = float(self.fraction_unbound)
        if not np.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
            raise ValueError("fraction_unbound must be finite and in [0, 1]")
        object.__setattr__(self, "fraction_unbound", fraction)
        object.__setattr__(self, "gfr_l_h", _positive(self.gfr_l_h, "gfr_l_h"))
        if self.secretion is not None and not hasattr(self.secretion, "rate_umol_h"):
            raise TypeError("secretion must implement SecretionKinetics")
        if not self.provenance.strip():
            raise ValueError("renal-disposition provenance is required")

    def rates_umol_h(
        self,
        total_plasma_concentration_umol_l: float,
        *,
        gfr_factor: float = 1.0,
        secretion_factor: float = 1.0,
    ) -> RenalDispositionFluxes:
        total = _nonnegative(
            total_plasma_concentration_umol_l,
            "total_plasma_concentration_umol_l",
        )
        gfr_scale = _nonnegative(gfr_factor, "gfr_factor")
        secretion_scale = _nonnegative(secretion_factor, "secretion_factor")
        filtration = self.fraction_unbound * self.gfr_l_h * gfr_scale * total
        unbound = self.fraction_unbound * total
        secretion = (
            0.0
            if self.secretion is None
            else self.secretion.rate_umol_h(
                unbound,
                activity_factor=secretion_scale,
            )
        )
        return RenalDispositionFluxes(
            filtration_umol_h=float(filtration),
            secretion_umol_h=float(secretion),
        )
