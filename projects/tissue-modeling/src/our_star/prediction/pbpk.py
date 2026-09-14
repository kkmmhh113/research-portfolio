"""Bridge deterministic v0.4 virtual subjects to PBPK endpoint callbacks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from ..models.amount_simulation import simulate_amount_patient
from ..models.multispecies import MultiSpeciesPBPKModel
from ..pbpk import DoseEvent, DrugParameters, PatientPhysiology
from ..population.adapters import materialize_patient_physiology, mechanism_factors
from ..population.specification import VirtualSubject
from ..virtual_trial import SolverSettings


EndpointStatistic = Literal["value_at_time", "cmax", "tmax", "auc_last", "final"]


@dataclass(frozen=True)
class TrajectoryEndpoint:
    quantity: str
    column: str
    statistic: EndpointStatistic
    time_h: float | None = None

    def __post_init__(self) -> None:
        if not self.quantity.strip() or not self.column.strip():
            raise ValueError("endpoint quantity and column are required")
        if self.statistic not in {
            "value_at_time",
            "cmax",
            "tmax",
            "auc_last",
            "final",
        }:
            raise ValueError("unsupported endpoint statistic")
        if self.statistic == "value_at_time":
            if self.time_h is None or not np.isfinite(self.time_h) or self.time_h < 0.0:
                raise ValueError("value_at_time requires finite time_h >= 0")
            object.__setattr__(self, "time_h", float(self.time_h))
        elif self.time_h is not None:
            raise ValueError("time_h is only valid for value_at_time")


def evaluate_trajectory_endpoint(
    trajectory: pd.DataFrame,
    endpoint: TrajectoryEndpoint,
) -> float:
    if "time_h" not in trajectory or endpoint.column not in trajectory:
        raise ValueError("trajectory lacks endpoint time or value column")
    times = trajectory["time_h"].to_numpy(dtype=np.float64)
    values = trajectory[endpoint.column].to_numpy(dtype=np.float64)
    if (
        times.ndim != 1
        or times.size == 0
        or values.shape != times.shape
        or np.any(~np.isfinite(times))
        or np.any(~np.isfinite(values))
        or np.any(np.diff(times) <= 0.0)
    ):
        raise ValueError("trajectory endpoint inputs must be finite and ordered")
    if endpoint.statistic == "value_at_time":
        assert endpoint.time_h is not None
        if endpoint.time_h < times[0] or endpoint.time_h > times[-1]:
            raise ValueError("endpoint time lies outside the simulated trajectory")
        result = np.interp(endpoint.time_h, times, values)
    elif endpoint.statistic == "cmax":
        result = np.max(values)
    elif endpoint.statistic == "tmax":
        result = times[int(np.argmax(values))]
    elif endpoint.statistic == "auc_last":
        result = np.trapezoid(values, times)
    else:
        result = values[-1]
    result = float(result)
    if not np.isfinite(result):
        raise FloatingPointError("endpoint calculation produced a non-finite value")
    return result


ModelFactory = Callable[[Mapping[str, float]], MultiSpeciesPBPKModel]


def build_pbpk_simulation_callback(
    *,
    reference_patient: PatientPhysiology,
    drug: DrugParameters,
    doses: Sequence[DoseEvent],
    model_factory: ModelFactory,
    endpoints: Sequence[TrajectoryEndpoint],
    duration_h: float,
    output_interval_h: float = 0.25,
    solver: SolverSettings | None = None,
):
    """Create the callback consumed by ``generate_predictive_replicates``.

    Random-effect draws are already frozen in each ``VirtualSubject``.  The
    callback does not inject residual noise; model discrepancy and assay noise
    belong to the separate prediction-interval calibration layer.
    """

    if not isinstance(reference_patient, PatientPhysiology):
        raise TypeError("reference_patient must be PatientPhysiology")
    if not isinstance(drug, DrugParameters):
        raise TypeError("drug must be DrugParameters")
    dose_records = tuple(doses)
    if not dose_records:
        raise ValueError("at least one dose is required")
    endpoint_records = tuple(endpoints)
    if not endpoint_records:
        raise ValueError("at least one endpoint is required")
    if any(not isinstance(item, TrajectoryEndpoint) for item in endpoint_records):
        raise TypeError("endpoints must be TrajectoryEndpoint records")
    quantities = tuple(item.quantity for item in endpoint_records)
    if len(quantities) != len(set(quantities)):
        raise ValueError("endpoint quantities must be unique")
    if not callable(model_factory):
        raise TypeError("model_factory must be callable")
    if not np.isfinite(duration_h) or duration_h <= 0.0:
        raise ValueError("duration_h must be finite and > 0")

    def simulate(
        subject: VirtualSubject,
        _replicate_index: int,
        _rng: np.random.Generator,
    ) -> Mapping[str, float]:
        patient = materialize_patient_physiology(subject, reference_patient)
        model = model_factory(mechanism_factors(subject))
        if not isinstance(model, MultiSpeciesPBPKModel):
            raise TypeError(
                "v0.4 PBPK prediction callbacks require a molar "
                "MultiSpeciesPBPKModel"
            )
        result = simulate_amount_patient(
            patient,
            drug,
            dose_records,
            duration_h=float(duration_h),
            output_interval_h=output_interval_h,
            solver=solver,
            model=model,
        )
        return {
            endpoint.quantity: evaluate_trajectory_endpoint(
                result.trajectory,
                endpoint,
            )
            for endpoint in endpoint_records
        }

    return simulate


__all__ = [
    "EndpointStatistic",
    "ModelFactory",
    "TrajectoryEndpoint",
    "build_pbpk_simulation_callback",
    "evaluate_trajectory_endpoint",
]
