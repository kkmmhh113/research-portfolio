"""Simulation-facing interface implemented by assembled PBPK systems."""

from __future__ import annotations

from typing import Mapping, Protocol, Sequence

import numpy as np

from ..pbpk import DoseEvent, DrugParameters, PatientPhysiology


class PBPKSimulationModel(Protocol):
    model_id: str

    def initial_state(self) -> np.ndarray: ...

    def apply_doses(
        self, state: np.ndarray, events: Sequence[DoseEvent]
    ) -> np.ndarray: ...

    def rhs(
        self,
        time_h: float,
        state: np.ndarray,
        patient: PatientPhysiology,
        drug: DrugParameters,
    ) -> np.ndarray: ...

    def concentration_record(
        self,
        state: np.ndarray,
        patient: PatientPhysiology,
        drug: DrugParameters,
    ) -> Mapping[str, float]: ...
