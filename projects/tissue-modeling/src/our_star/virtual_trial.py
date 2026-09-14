"""Simulation and artifact utilities for the virtual Phase I PBPK model."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml
from scipy.integrate import solve_ivp

from .models.base import PBPKSimulationModel
from .models.legacy import LegacyPBPKModel
from .models.segmented import FormulationParameters
from .pbpk import (
    DoseEvent,
    DrugParameters,
    PatientPhysiology,
)
from .virtual_population import PopulationSpecification, population_records


@dataclass(frozen=True)
class SolverSettings:
    method: str = "BDF"
    relative_tolerance: float = 1e-8
    absolute_tolerance: float = 1e-10
    max_step_h: float = 0.1


@dataclass
class PatientSimulation:
    patient: PatientPhysiology
    times_h: np.ndarray
    states_mg: np.ndarray
    trajectory: pd.DataFrame
    summary: dict[str, Any]
    model_id: str


@dataclass
class PopulationSimulation:
    drug: DrugParameters
    doses: tuple[DoseEvent, ...]
    patients: list[PatientPhysiology]
    trajectories: pd.DataFrame
    patient_summaries: pd.DataFrame
    cohort_trajectory: pd.DataFrame
    cohort_summary: dict[str, Any]
    model_id: str


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a mapping in {path}")
    return payload


def load_drug(path: str | Path) -> DrugParameters:
    payload = load_yaml(path)
    return DrugParameters.from_mapping(payload["drug"])


def load_formulation(path: str | Path) -> FormulationParameters:
    payload = load_yaml(path)
    return FormulationParameters.from_mapping(payload["formulation"])


def load_population_specification(path: str | Path) -> PopulationSpecification:
    payload = load_yaml(path)
    return PopulationSpecification.from_mapping(payload["population"])


def load_trial_doses(path: str | Path) -> tuple[DoseEvent, ...]:
    payload = load_yaml(path)
    raw_doses = payload["trial"]["doses"]
    return tuple(DoseEvent(**raw) for raw in raw_doses)


def _output_grid(duration_h: float, interval_h: float) -> np.ndarray:
    if duration_h <= 0 or interval_h <= 0:
        raise ValueError("duration_h and interval_h must be positive")
    count = int(np.floor(duration_h / interval_h))
    grid = np.arange(count + 1, dtype=np.float64) * interval_h
    if grid[-1] < duration_h - 1e-12:
        grid = np.append(grid, duration_h)
    else:
        grid[-1] = duration_h
    return grid


def _group_doses(doses: Sequence[DoseEvent], duration_h: float) -> dict[float, list[DoseEvent]]:
    grouped: dict[float, list[DoseEvent]] = {}
    for dose in doses:
        if dose.time_h > duration_h:
            raise ValueError(f"dose at {dose.time_h} h is beyond trial duration {duration_h} h")
        grouped.setdefault(float(dose.time_h), []).append(dose)
    return grouped


def simulate_patient(
    patient: PatientPhysiology,
    drug: DrugParameters,
    doses: Sequence[DoseEvent],
    *,
    duration_h: float,
    output_interval_h: float = 0.25,
    solver: SolverSettings | None = None,
    model: PBPKSimulationModel | None = None,
) -> PatientSimulation:
    """Simulate one patient with exact dose discontinuities."""

    if not doses:
        raise ValueError("at least one dose is required")
    selected_model = model or LegacyPBPKModel(drug)
    settings = solver or SolverSettings()
    dose_map = _group_doses(doses, duration_h)
    regular_times = _output_grid(duration_h, output_interval_h)
    boundaries = sorted(set([0.0, duration_h, *dose_map.keys()]))

    state = selected_model.initial_state()
    delivered_mg = 0.0
    recorded_times: list[float] = []
    recorded_states: list[np.ndarray] = []
    delivered_at_record: list[float] = []

    if 0.0 in dose_map:
        state = selected_model.apply_doses(state, dose_map[0.0])
        delivered_mg += sum(event.amount_mg for event in dose_map[0.0])
    recorded_times.append(0.0)
    recorded_states.append(state.copy())
    delivered_at_record.append(delivered_mg)

    current_time = 0.0
    for boundary in boundaries[1:]:
        interior = regular_times[
            (regular_times > current_time + 1e-12) & (regular_times < boundary - 1e-12)
        ]
        evaluation_times = np.append(interior, boundary)
        solution = solve_ivp(
            fun=lambda time_h, values: selected_model.rhs(
                time_h, values, patient, drug
            ),
            t_span=(current_time, boundary),
            y0=state,
            t_eval=evaluation_times,
            method=settings.method,
            rtol=settings.relative_tolerance,
            atol=settings.absolute_tolerance,
            max_step=settings.max_step_h,
        )
        if not solution.success:
            raise RuntimeError(
                f"PBPK solver failed for {patient.patient_id} at {boundary} h: "
                f"{solution.message}"
            )
        if not np.all(np.isfinite(solution.y)):
            raise FloatingPointError(f"non-finite PBPK state for {patient.patient_id}")
        minimum = float(solution.y.min())
        if minimum < -1e-6:
            raise FloatingPointError(
                f"negative amount {minimum:.3e} mg for {patient.patient_id}"
            )

        for time_h, values in zip(solution.t[:-1], solution.y.T[:-1], strict=True):
            recorded_times.append(float(time_h))
            recorded_states.append(np.maximum(values, 0.0))
            delivered_at_record.append(delivered_mg)

        state = np.maximum(solution.y[:, -1], 0.0)
        current_time = float(boundary)
        if boundary in dose_map:
            state = selected_model.apply_doses(state, dose_map[boundary])
            delivered_mg += sum(event.amount_mg for event in dose_map[boundary])
        recorded_times.append(current_time)
        recorded_states.append(state.copy())
        delivered_at_record.append(delivered_mg)

    times = np.asarray(recorded_times, dtype=np.float64)
    states = np.vstack(recorded_states)
    delivered = np.asarray(delivered_at_record, dtype=np.float64)
    if np.any(np.diff(times) <= 0):
        raise RuntimeError("simulation output times must be strictly increasing")

    records: list[dict[str, Any]] = []
    for time_h, values, cumulative_dose in zip(times, states, delivered, strict=True):
        record = {
            "patient_id": patient.patient_id,
            "time_h": float(time_h),
            **selected_model.concentration_record(values, patient, drug),
            "delivered_dose_mg": float(cumulative_dose),
        }
        record["mass_balance_error_mg"] = record["tracked_mass_mg"] - cumulative_dose
        records.append(record)
    trajectory = pd.DataFrame.from_records(records)
    summary = summarize_patient_trajectory(trajectory, patient)
    return PatientSimulation(
        patient=patient,
        times_h=times,
        states_mg=states,
        trajectory=trajectory,
        summary=summary,
        model_id=selected_model.model_id,
    )


def summarize_patient_trajectory(
    trajectory: pd.DataFrame,
    patient: PatientPhysiology,
) -> dict[str, Any]:
    times = trajectory["time_h"].to_numpy(dtype=np.float64)
    concentration = trajectory["plasma_parent_mg_l"].to_numpy(dtype=np.float64)
    peak_index = int(np.argmax(concentration))
    auc = float(np.trapezoid(concentration, times))
    terminal_half_life = _terminal_half_life_h(times, concentration, peak_index)
    return {
        "patient_id": patient.patient_id,
        "age_years": patient.age_years,
        "sex": patient.sex,
        "body_weight_kg": patient.body_weight_kg,
        "genotype_label": patient.genotype_label,
        "enzyme_activity_factor": patient.enzyme_activity_factor,
        "liver_function_fraction": patient.liver_function_fraction,
        "renal_function_fraction": patient.renal_function_fraction,
        "cmax_mg_l": float(concentration[peak_index]),
        "tmax_h": float(times[peak_index]),
        "auc_last_mg_h_l": auc,
        "terminal_half_life_h": terminal_half_life,
        "max_abs_mass_balance_error_mg": float(
            trajectory["mass_balance_error_mg"].abs().max()
        ),
        "final_urine_parent_mg": float(trajectory["urine_parent_mg"].iloc[-1]),
        "final_urine_metabolite_mg": float(
            trajectory["urine_metabolite_mg"].iloc[-1]
        ),
        "final_feces_parent_mg": float(trajectory["feces_parent_mg"].iloc[-1]),
    }


def _terminal_half_life_h(
    times_h: np.ndarray,
    concentration_mg_l: np.ndarray,
    peak_index: int,
) -> float:
    post_peak = np.arange(len(times_h)) > peak_index
    positive = concentration_mg_l > max(float(concentration_mg_l.max()) * 1e-5, 1e-12)
    candidates = np.flatnonzero(post_peak & positive)
    if len(candidates) < 4:
        return float("nan")
    terminal = candidates[max(0, len(candidates) - max(4, len(candidates) // 3)) :]
    slope, _intercept = np.polyfit(
        times_h[terminal], np.log(concentration_mg_l[terminal]), 1
    )
    if slope >= 0:
        return float("nan")
    return float(np.log(2.0) / -slope)


def simulate_population(
    patients: Sequence[PatientPhysiology],
    drug: DrugParameters,
    doses: Sequence[DoseEvent],
    *,
    duration_h: float,
    output_interval_h: float = 0.25,
    solver: SolverSettings | None = None,
    model: PBPKSimulationModel | None = None,
) -> PopulationSimulation:
    if not patients:
        raise ValueError("at least one patient is required")
    selected_model = model or LegacyPBPKModel(drug)
    patient_results = [
        simulate_patient(
            patient,
            drug,
            doses,
            duration_h=duration_h,
            output_interval_h=output_interval_h,
            solver=solver,
            model=selected_model,
        )
        for patient in patients
    ]
    trajectories = pd.concat(
        [result.trajectory for result in patient_results], ignore_index=True
    )
    summaries = pd.DataFrame.from_records([result.summary for result in patient_results])
    cohort_trajectory = summarize_cohort_trajectory(trajectories)
    cohort_summary = {
        "drug": drug.name,
        "patient_count": len(patients),
        "dose_count": len(doses),
        "total_nominal_dose_mg": float(sum(dose.amount_mg for dose in doses)),
        "cmax_mg_l": _distribution_summary(summaries["cmax_mg_l"]),
        "auc_last_mg_h_l": _distribution_summary(summaries["auc_last_mg_h_l"]),
        "terminal_half_life_h": _distribution_summary(
            summaries["terminal_half_life_h"].dropna()
        ),
        "max_mass_balance_error_mg": float(
            summaries["max_abs_mass_balance_error_mg"].max()
        ),
    }
    return PopulationSimulation(
        drug=drug,
        doses=tuple(doses),
        patients=list(patients),
        trajectories=trajectories,
        patient_summaries=summaries,
        cohort_trajectory=cohort_trajectory,
        cohort_summary=cohort_summary,
        model_id=selected_model.model_id,
    )


def _distribution_summary(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {key: float("nan") for key in ("mean", "p05", "p50", "p95")}
    return {
        "mean": float(np.mean(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
    }


def summarize_cohort_trajectory(trajectories: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "plasma_parent_mg_l",
        "plasma_metabolite_mg_l",
        "liver_zone1_parent_mg_l",
        "liver_zone2_parent_mg_l",
        "liver_zone3_parent_mg_l",
    ]
    rows: list[dict[str, float]] = []
    for time_h, group in trajectories.groupby("time_h", sort=True):
        row: dict[str, float] = {"time_h": float(time_h)}
        for metric in metrics:
            values = group[metric].to_numpy(dtype=np.float64)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_p05"] = float(np.quantile(values, 0.05))
            row[f"{metric}_p50"] = float(np.quantile(values, 0.50))
            row[f"{metric}_p95"] = float(np.quantile(values, 0.95))
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_population_artifacts(
    result: PopulationSimulation,
    output_directory: str | Path,
    *,
    run_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "trajectories": output / "patient_trajectories.csv",
        "patient_summaries": output / "patient_pk_summaries.csv",
        "cohort_trajectory": output / "cohort_trajectory.csv",
        "patients": output / "virtual_patients.csv",
        "summary": output / "cohort_summary.json",
    }
    result.trajectories.to_csv(paths["trajectories"], index=False)
    result.patient_summaries.to_csv(paths["patient_summaries"], index=False)
    result.cohort_trajectory.to_csv(paths["cohort_trajectory"], index=False)
    pd.DataFrame.from_records(population_records(result.patients)).to_csv(
        paths["patients"], index=False
    )
    with paths["summary"].open("w", encoding="utf-8") as handle:
        json.dump(result.cohort_summary, handle, indent=2, sort_keys=True, allow_nan=True)

    manifest = {
        "model": result.model_id,
        "clinical_status": "research_only_not_validated_for_clinical_decisions",
        "drug": asdict(result.drug),
        "doses": [asdict(dose) for dose in result.doses],
        "cohort_summary": result.cohort_summary,
        "run_metadata": dict(run_metadata or {}),
        "files": {
            key: {"path": path.name, "sha256": _sha256(path)}
            for key, path in paths.items()
        },
    }
    manifest_path = output / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=True)
    return manifest
