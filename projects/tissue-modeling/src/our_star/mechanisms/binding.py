"""Explicit saturable ACE binding for mobile enalaprilat."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _finite_nonnegative(value: float, label: str) -> float:
    converted = float(value)
    if not np.isfinite(converted) or converted < 0.0:
        raise ValueError(f"{label} must be finite and >= 0")
    return converted


@dataclass(frozen=True)
class ACEBindingFluxes:
    association_umol_h: float
    dissociation_umol_h: float

    @property
    def net_to_bound_umol_h(self) -> float:
        return self.association_umol_h - self.dissociation_umol_h


@dataclass(frozen=True)
class SaturableACEBinding:
    """Mass-conserving binding to a finite ACE-site amount.

    ``kon_l_umol_h`` multiplies concentration (umol/L) by unoccupied site
    amount (umol). ``koff_h`` multiplies bound ligand amount. The bound state
    represents ligand amount, not total receptor/protein mass.
    """

    bmax_umol: float
    kon_l_umol_h: float
    koff_h: float
    provenance: str

    def __post_init__(self) -> None:
        for label in ("bmax_umol", "kon_l_umol_h", "koff_h"):
            value = float(getattr(self, label))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{label} must be finite and > 0")
            object.__setattr__(self, label, value)
        if not self.provenance.strip():
            raise ValueError("binding provenance is required")

    def fluxes_umol_h(
        self,
        unbound_mobile_concentration_umol_l: float,
        bound_amount_umol: float,
        *,
        binding_capacity_factor: float = 1.0,
    ) -> ACEBindingFluxes:
        concentration = _finite_nonnegative(
            unbound_mobile_concentration_umol_l,
            "unbound_mobile_concentration_umol_l",
        )
        bound = _finite_nonnegative(bound_amount_umol, "bound_amount_umol")
        factor = float(binding_capacity_factor)
        if not np.isfinite(factor) or factor <= 0.0:
            raise ValueError("binding_capacity_factor must be finite and > 0")
        capacity = self.bmax_umol * factor
        tolerance = max(1.0, capacity) * 1.0e-12
        if bound > capacity + tolerance:
            raise ValueError("bound amount exceeds the available ACE capacity")
        unoccupied = max(capacity - bound, 0.0)
        association = self.kon_l_umol_h * concentration * unoccupied
        dissociation = self.koff_h * bound
        return ACEBindingFluxes(
            association_umol_h=float(association),
            dissociation_umol_h=float(dissociation),
        )
