"""Unit-safe simulation result for post-v0.3 molar PBPK models.

The frozen v0.3 simulator stores every raw state matrix in a historical field
named ``states_mg``.  Multi-species v0.4 models deliberately use µmol states,
so exposing that legacy field to new callers would make a silent unit mistake
too easy.  This adapter keeps the proven dose-discontinuity/ODE implementation
but returns an explicitly named, read-only ``states_umol`` matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ..core.species import AmountUnit
from ..pbpk import DoseEvent, DrugParameters, PatientPhysiology
from ..virtual_trial import SolverSettings, simulate_patient
from .multispecies import MultiSpeciesPBPKModel


@dataclass(frozen=True)
class AmountPatientSimulation:
    """One molar-state PBPK run with an explicit state unit boundary."""

    patient: PatientPhysiology
    times_h: np.ndarray
    states_umol: np.ndarray
    trajectory: pd.DataFrame
    summary: Mapping[str, Any]
    model_id: str
    state_amount_unit: AmountUnit = AmountUnit.UMOL

    def __post_init__(self) -> None:
        times = np.asarray(self.times_h, dtype=np.float64).copy()
        states = np.asarray(self.states_umol, dtype=np.float64).copy()
        if times.ndim != 1 or times.size == 0:
            raise ValueError("times_h must be a non-empty vector")
        if states.ndim != 2 or states.shape[0] != times.size:
            raise ValueError("states_umol must have one row per output time")
        if np.any(~np.isfinite(times)) or np.any(np.diff(times) <= 0.0):
            raise ValueError("times_h must be finite and strictly increasing")
        if np.any(~np.isfinite(states)) or np.any(states < -1.0e-8):
            raise ValueError("states_umol must be finite and nonnegative")
        if not isinstance(self.trajectory, pd.DataFrame) or self.trajectory.empty:
            raise ValueError("trajectory must be a non-empty DataFrame")
        if self.trajectory.shape[0] != times.size:
            raise ValueError("trajectory must have one row per output time")
        if not self.model_id.strip():
            raise ValueError("model_id is required")
        times.setflags(write=False)
        states.setflags(write=False)
        object.__setattr__(self, "times_h", times)
        object.__setattr__(self, "states_umol", states)
        object.__setattr__(self, "summary", MappingProxyType(dict(self.summary)))
        object.__setattr__(self, "state_amount_unit", AmountUnit(self.state_amount_unit))


def simulate_amount_patient(
    patient: PatientPhysiology,
    drug: DrugParameters,
    doses: Sequence[DoseEvent],
    *,
    model: MultiSpeciesPBPKModel,
    duration_h: float,
    output_interval_h: float = 0.25,
    solver: SolverSettings | None = None,
) -> AmountPatientSimulation:
    """Run a multi-species model without leaking the legacy mass field name."""

    if not isinstance(model, MultiSpeciesPBPKModel):
        raise TypeError("model must be a MultiSpeciesPBPKModel")
    if AmountUnit(model.state_amount_unit) != AmountUnit.UMOL:
        raise ValueError("multi-species amount simulation requires µmol states")
    legacy_result = simulate_patient(
        patient,
        drug,
        tuple(doses),
        duration_h=duration_h,
        output_interval_h=output_interval_h,
        solver=solver,
        model=model,
    )
    return AmountPatientSimulation(
        patient=legacy_result.patient,
        times_h=legacy_result.times_h,
        states_umol=legacy_result.states_mg,
        trajectory=legacy_result.trajectory,
        summary=legacy_result.summary,
        model_id=legacy_result.model_id,
    )


__all__ = ["AmountPatientSimulation", "simulate_amount_patient"]
