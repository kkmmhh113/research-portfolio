"""CES1-mediated parent-to-metabolite formation kinetics in molar units."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


def _nonnegative_finite(value: float, *, label: str) -> float:
    converted = float(value)
    if not np.isfinite(converted) or converted < 0.0:
        raise ValueError(f"{label} must be finite and >= 0")
    return converted


def _positive_finite(value: float, *, label: str) -> float:
    converted = float(value)
    if not np.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{label} must be finite and > 0")
    return converted


class FormationKinetics(Protocol):
    """Protocol for a nonnegative molar parent-to-metabolite flux."""

    def rate_umol_h(
        self,
        unbound_parent_umol_l: float,
        *,
        activity_factor: float = 1.0,
    ) -> float: ...


@dataclass(frozen=True)
class LinearCES1Formation:
    """Evidence-gated linear CES1 formation clearance.

    ``clearance_l_h`` acts on an unbound parent concentration in micromoles
    per litre, so the returned formation extent has units micromoles per hour.
    """

    clearance_l_h: float
    provenance: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "clearance_l_h",
            _positive_finite(self.clearance_l_h, label="clearance_l_h"),
        )
        if not self.provenance.strip():
            raise ValueError("formation provenance is required")

    def rate_umol_h(
        self,
        unbound_parent_umol_l: float,
        *,
        activity_factor: float = 1.0,
    ) -> float:
        concentration = _nonnegative_finite(
            unbound_parent_umol_l,
            label="unbound_parent_umol_l",
        )
        activity = _nonnegative_finite(activity_factor, label="activity_factor")
        return self.clearance_l_h * concentration * activity


@dataclass(frozen=True)
class CES1SubstrateInhibition:
    """Substrate-inhibition candidate that must pass mechanism admission.

    The law is ``Vmax*C / (Km + C + C**2/Ki)``.  It is deliberately a
    separate candidate rather than an automatic replacement for the linear
    law because ``Vmax``, ``Km``, and ``Ki`` are frequently confounded by a
    single oral parent/metabolite curve.
    """

    vmax_umol_h: float
    km_umol_l: float
    ki_umol_l: float
    provenance: str

    def __post_init__(self) -> None:
        for label in ("vmax_umol_h", "km_umol_l", "ki_umol_l"):
            object.__setattr__(
                self,
                label,
                _positive_finite(getattr(self, label), label=label),
            )
        if not self.provenance.strip():
            raise ValueError("formation provenance is required")

    def rate_umol_h(
        self,
        unbound_parent_umol_l: float,
        *,
        activity_factor: float = 1.0,
    ) -> float:
        concentration = _nonnegative_finite(
            unbound_parent_umol_l,
            label="unbound_parent_umol_l",
        )
        activity = _nonnegative_finite(activity_factor, label="activity_factor")
        if concentration == 0.0:
            return 0.0
        denominator = (
            self.km_umol_l
            + concentration
            + concentration**2 / self.ki_umol_l
        )
        return activity * self.vmax_umol_h * concentration / denominator
