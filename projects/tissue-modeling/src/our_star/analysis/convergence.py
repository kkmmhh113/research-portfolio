"""Numerical-convergence checks for the reduced PBPK simulation interface.

The utilities in this module deliberately operate on synthetic/reference
simulations.  They do not load, inspect, or score clinical observations.  A
convergence result is therefore an implementation-verification result, not a
claim of biological or clinical predictive validity.

The production simulator clips small negative solver states at recorded output
times.  This module mirrors that behaviour for reported concentrations while
also retaining the minimum *raw* (pre-clamp) state returned by the integrator.
That distinction prevents clipping from hiding a positivity problem.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.integrate import solve_ivp

from ..models.base import PBPKSimulationModel
from ..pbpk import DoseEvent, DrugParameters, PatientPhysiology


REPORT_SCHEMA_VERSION = "paper_b.solver_convergence.v1"
_RUNTIME_KEY = "runtime_seconds_observed"


@dataclass(frozen=True)
class ConvergenceThresholds:
    """Prespecified numerical-development targets and invalidity boundary."""

    endpoint_relative_difference_maximum: float = 1.0e-3
    curve_fraction_of_reference_cmax_maximum: float = 1.0e-3
    curve_absolute_floor_mg_l: float = 1.0e-10
    maximum_mass_balance_error_mg: float = 1.0e-5
    minimum_pre_clamp_state_mg: float = -1.0e-6
    maximum_dose_jump_error_mg: float = 1.0e-10
    prospective_endpoint_invalidity_boundary: float = 1.0e-2

    def __post_init__(self) -> None:
        positive = (
            "endpoint_relative_difference_maximum",
            "curve_fraction_of_reference_cmax_maximum",
            "curve_absolute_floor_mg_l",
            "maximum_mass_balance_error_mg",
            "maximum_dose_jump_error_mg",
            "prospective_endpoint_invalidity_boundary",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0")
            object.__setattr__(self, name, value)
        minimum = float(self.minimum_pre_clamp_state_mg)
        if not np.isfinite(minimum) or minimum > 0.0:
            raise ValueError("minimum_pre_clamp_state_mg must be finite and <= 0")
        object.__setattr__(self, "minimum_pre_clamp_state_mg", minimum)
        if (
            self.prospective_endpoint_invalidity_boundary
            <= self.endpoint_relative_difference_maximum
        ):
            raise ValueError(
                "prospective invalidity boundary must exceed the development target"
            )

    def to_record(self) -> dict[str, float]:
        return {
            "curve_absolute_floor_mg_l": self.curve_absolute_floor_mg_l,
            "curve_fraction_of_reference_cmax_maximum": (
                self.curve_fraction_of_reference_cmax_maximum
            ),
            "endpoint_relative_difference_maximum": (
                self.endpoint_relative_difference_maximum
            ),
            "maximum_dose_jump_error_mg": self.maximum_dose_jump_error_mg,
            "maximum_mass_balance_error_mg": self.maximum_mass_balance_error_mg,
            "minimum_pre_clamp_state_mg": self.minimum_pre_clamp_state_mg,
            "prospective_endpoint_invalidity_boundary": (
                self.prospective_endpoint_invalidity_boundary
            ),
        }


@dataclass(frozen=True)
class SolverProtocolSetting:
    """One solver/output-grid setting in a convergence matrix."""

    label: str
    method: str
    relative_tolerance: float
    absolute_tolerance: float
    max_step_h: float
    output_interval_h: float
    comparison_reference_label: str
    required_for_case_pass: bool
    role: str

    def __post_init__(self) -> None:
        if not self.label.strip() or not self.comparison_reference_label.strip():
            raise ValueError("solver labels must be non-empty")
        if self.method not in {"BDF", "Radau"}:
            raise ValueError("convergence protocol supports BDF and Radau")
        if self.role not in {
            "loose_diagnostic",
            "nominal",
            "tight_reference",
            "cross_solver",
            "output_interval_sensitivity",
        }:
            raise ValueError(f"unsupported convergence role: {self.role}")
        for name in (
            "relative_tolerance",
            "absolute_tolerance",
            "max_step_h",
            "output_interval_h",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0")
            object.__setattr__(self, name, value)

    def to_record(self) -> dict[str, Any]:
        return {
            "absolute_tolerance": self.absolute_tolerance,
            "comparison_reference_label": self.comparison_reference_label,
            "label": self.label,
            "max_step_h": self.max_step_h,
            "method": self.method,
            "output_interval_h": self.output_interval_h,
            "relative_tolerance": self.relative_tolerance,
            "required_for_case_pass": self.required_for_case_pass,
            "role": self.role,
        }


@dataclass(frozen=True)
class ConvergenceCase:
    """A clinical-data-free representative PBPK simulation."""

    case_id: str
    model: PBPKSimulationModel
    patient: PatientPhysiology
    drug: DrugParameters
    doses: tuple[DoseEvent, ...]
    duration_h: float
    output_interval_h: float
    nominal_max_step_h: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.case_id.strip():
            raise ValueError("case_id is required")
        doses = tuple(self.doses)
        if not doses:
            raise ValueError("at least one dose is required")
        if any(dose.time_h > self.duration_h for dose in doses):
            raise ValueError("dose time may not exceed case duration")
        object.__setattr__(self, "doses", doses)
        for name in ("duration_h", "output_interval_h", "nominal_max_step_h"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0")
            object.__setattr__(self, name, value)
        metadata = dict(self.metadata)
        # Fail before a long simulation if metadata cannot enter strict JSON.
        json.dumps(metadata, sort_keys=True, allow_nan=False)
        object.__setattr__(self, "metadata", metadata)


@dataclass
class _CompletedRun:
    times_h: np.ndarray
    concentration_mg_l: np.ndarray
    auc_last_mg_h_l: float
    cmax_mg_l: float
    tmax_h: float
    maximum_abs_mass_balance_error_mg: float
    minimum_pre_clamp_state_mg: float
    finite_states: bool
    dose_jumps: list[dict[str, float | int]]
    maximum_abs_dose_jump_error_mg: float
    maximum_single_clip_mass_mg: float
    solver_counters: dict[str, int]


def default_solver_protocol(
    *, nominal_max_step_h: float, output_interval_h: float
) -> tuple[SolverProtocolSetting, ...]:
    """Return the prespecified loose/nominal/tight/Radau/grid matrix."""

    if not np.isfinite(nominal_max_step_h) or nominal_max_step_h <= 0.0:
        raise ValueError("nominal_max_step_h must be finite and > 0")
    if not np.isfinite(output_interval_h) or output_interval_h <= 0.0:
        raise ValueError("output_interval_h must be finite and > 0")
    return (
        SolverProtocolSetting(
            label="loose_bdf",
            method="BDF",
            relative_tolerance=1.0e-6,
            absolute_tolerance=1.0e-8,
            max_step_h=2.0 * nominal_max_step_h,
            output_interval_h=output_interval_h,
            comparison_reference_label="tight_bdf",
            required_for_case_pass=False,
            role="loose_diagnostic",
        ),
        SolverProtocolSetting(
            label="nominal_bdf",
            method="BDF",
            relative_tolerance=1.0e-8,
            absolute_tolerance=1.0e-10,
            max_step_h=nominal_max_step_h,
            output_interval_h=output_interval_h,
            comparison_reference_label="tight_bdf",
            required_for_case_pass=True,
            role="nominal",
        ),
        SolverProtocolSetting(
            label="tight_bdf",
            method="BDF",
            relative_tolerance=1.0e-10,
            absolute_tolerance=1.0e-12,
            max_step_h=0.5 * nominal_max_step_h,
            output_interval_h=output_interval_h,
            comparison_reference_label="tight_bdf",
            required_for_case_pass=True,
            role="tight_reference",
        ),
        SolverProtocolSetting(
            label="cross_solver_radau",
            method="Radau",
            relative_tolerance=1.0e-9,
            absolute_tolerance=1.0e-11,
            max_step_h=0.5 * nominal_max_step_h,
            output_interval_h=output_interval_h,
            comparison_reference_label="tight_bdf",
            required_for_case_pass=True,
            role="cross_solver",
        ),
        SolverProtocolSetting(
            label="nominal_bdf_half_output_interval",
            method="BDF",
            relative_tolerance=1.0e-8,
            absolute_tolerance=1.0e-10,
            max_step_h=nominal_max_step_h,
            output_interval_h=0.5 * output_interval_h,
            comparison_reference_label="nominal_bdf",
            required_for_case_pass=True,
            role="output_interval_sensitivity",
        ),
    )


def _output_grid(duration_h: float, interval_h: float) -> np.ndarray:
    count = int(np.floor(duration_h / interval_h))
    grid = np.arange(count + 1, dtype=np.float64) * interval_h
    if grid[-1] < duration_h - 1.0e-12:
        grid = np.append(grid, duration_h)
    else:
        grid[-1] = duration_h
    return grid


def _group_doses(
    doses: Sequence[DoseEvent], duration_h: float
) -> dict[float, list[DoseEvent]]:
    grouped: dict[float, list[DoseEvent]] = {}
    for dose in doses:
        if dose.time_h > duration_h:
            raise ValueError("dose time may not exceed simulation duration")
        grouped.setdefault(float(dose.time_h), []).append(dose)
    return grouped


def _apply_dose_and_record_jump(
    model: PBPKSimulationModel,
    state: np.ndarray,
    events: Sequence[DoseEvent],
    *,
    time_h: float,
) -> tuple[np.ndarray, dict[str, float | int]]:
    expected = float(sum(event.amount_mg for event in events))
    before = float(np.sum(state))
    updated = np.asarray(model.apply_doses(state, events), dtype=np.float64)
    after = float(np.sum(updated))
    actual = after - before
    return updated, {
        "actual_jump_mg": actual,
        "dose_event_count": len(events),
        "error_mg": actual - expected,
        "expected_jump_mg": expected,
        "time_h": float(time_h),
    }


def _simulate_case(
    case: ConvergenceCase, setting: SolverProtocolSetting
) -> _CompletedRun:
    dose_map = _group_doses(case.doses, case.duration_h)
    regular_times = _output_grid(case.duration_h, setting.output_interval_h)
    boundaries = sorted({0.0, case.duration_h, *dose_map.keys()})

    state = np.asarray(case.model.initial_state(), dtype=np.float64)
    if state.ndim != 1 or state.size == 0:
        raise ValueError("model initial_state must be a non-empty vector")
    raw_minimum = float(np.min(state))
    finite_states = bool(np.all(np.isfinite(state)))
    maximum_single_clip_mass_mg = 0.0
    delivered_mg = 0.0
    dose_jumps: list[dict[str, float | int]] = []
    recorded_times: list[float] = []
    recorded_states: list[np.ndarray] = []
    delivered_at_record: list[float] = []
    counters = {"accepted_steps": 0, "nfev": 0, "njev": 0, "nlu": 0}

    if 0.0 in dose_map:
        state, jump = _apply_dose_and_record_jump(
            case.model, state, dose_map[0.0], time_h=0.0
        )
        dose_jumps.append(jump)
        delivered_mg += float(jump["expected_jump_mg"])
    raw_minimum = min(raw_minimum, float(np.min(state)))
    finite_states = finite_states and bool(np.all(np.isfinite(state)))
    recorded_times.append(0.0)
    recorded_states.append(np.maximum(state, 0.0))
    delivered_at_record.append(delivered_mg)

    current_time = 0.0
    for boundary in boundaries[1:]:
        solution = solve_ivp(
            fun=lambda time_h, values: case.model.rhs(
                time_h, values, case.patient, case.drug
            ),
            t_span=(current_time, boundary),
            y0=state,
            method=setting.method,
            rtol=setting.relative_tolerance,
            atol=setting.absolute_tolerance,
            max_step=setting.max_step_h,
            dense_output=True,
        )
        counters["accepted_steps"] += max(int(solution.t.size) - 1, 0)
        counters["nfev"] += int(solution.nfev)
        counters["njev"] += int(solution.njev)
        counters["nlu"] += int(solution.nlu)
        if not solution.success:
            raise RuntimeError(
                f"{setting.method} failed at {boundary:g} h: {solution.message}"
            )
        if solution.sol is None:
            raise RuntimeError("solver did not return the requested dense solution")
        internal = np.asarray(solution.y, dtype=np.float64)
        finite_states = finite_states and bool(np.all(np.isfinite(internal)))
        if internal.size:
            raw_minimum = min(raw_minimum, float(np.min(internal)))

        interior = regular_times[
            (regular_times > current_time + 1.0e-12)
            & (regular_times < boundary - 1.0e-12)
        ]
        evaluation_times = np.append(interior, boundary)
        evaluated = np.asarray(solution.sol(evaluation_times), dtype=np.float64)
        finite_states = finite_states and bool(np.all(np.isfinite(evaluated)))
        raw_minimum = min(raw_minimum, float(np.min(evaluated)))

        for time_h, raw_values in zip(
            evaluation_times[:-1], evaluated.T[:-1], strict=True
        ):
            clipped = np.maximum(raw_values, 0.0)
            maximum_single_clip_mass_mg = max(
                maximum_single_clip_mass_mg,
                float(np.sum(clipped) - np.sum(raw_values)),
            )
            recorded_times.append(float(time_h))
            recorded_states.append(clipped)
            delivered_at_record.append(delivered_mg)

        raw_endpoint = evaluated[:, -1]
        state = np.maximum(raw_endpoint, 0.0)
        maximum_single_clip_mass_mg = max(
            maximum_single_clip_mass_mg,
            float(np.sum(state) - np.sum(raw_endpoint)),
        )
        current_time = float(boundary)
        if boundary in dose_map:
            state, jump = _apply_dose_and_record_jump(
                case.model, state, dose_map[boundary], time_h=boundary
            )
            dose_jumps.append(jump)
            delivered_mg += float(jump["expected_jump_mg"])
        recorded_times.append(current_time)
        recorded_states.append(state.copy())
        delivered_at_record.append(delivered_mg)

    times = np.asarray(recorded_times, dtype=np.float64)
    if np.any(np.diff(times) <= 0.0):
        raise RuntimeError("convergence output times must be strictly increasing")
    states = np.vstack(recorded_states)
    delivered = np.asarray(delivered_at_record, dtype=np.float64)
    concentrations: list[float] = []
    mass_residuals: list[float] = []
    for values, cumulative_dose in zip(states, delivered, strict=True):
        record = case.model.concentration_record(values, case.patient, case.drug)
        concentration = float(record["plasma_parent_mg_l"])
        tracked_mass = float(record["tracked_mass_mg"])
        concentrations.append(concentration)
        mass_residuals.append(tracked_mass - float(cumulative_dose))
    concentration_array = np.asarray(concentrations, dtype=np.float64)
    mass_array = np.asarray(mass_residuals, dtype=np.float64)
    finite_states = finite_states and bool(
        np.all(np.isfinite(states))
        and np.all(np.isfinite(concentration_array))
        and np.all(np.isfinite(mass_array))
    )
    if not finite_states:
        raise FloatingPointError("non-finite solver state, concentration, or mass")
    peak_index = int(np.argmax(concentration_array))
    jump_errors = [abs(float(record["error_mg"])) for record in dose_jumps]
    return _CompletedRun(
        times_h=times,
        concentration_mg_l=concentration_array,
        auc_last_mg_h_l=float(np.trapezoid(concentration_array, times)),
        cmax_mg_l=float(concentration_array[peak_index]),
        tmax_h=float(times[peak_index]),
        maximum_abs_mass_balance_error_mg=float(np.max(np.abs(mass_array))),
        minimum_pre_clamp_state_mg=raw_minimum,
        finite_states=finite_states,
        dose_jumps=dose_jumps,
        maximum_abs_dose_jump_error_mg=max(jump_errors, default=0.0),
        maximum_single_clip_mass_mg=maximum_single_clip_mass_mg,
        solver_counters=counters,
    )


def _run_record(
    case: ConvergenceCase, setting: SolverProtocolSetting
) -> tuple[dict[str, Any], _CompletedRun | None]:
    started = perf_counter()
    try:
        completed = _simulate_case(case, setting)
    except Exception as exc:  # The report must preserve solver failure, not abort it.
        runtime = perf_counter() - started
        return (
            {
                "error": {"message": str(exc), "type": type(exc).__name__},
                "finite_states": False,
                "setting": setting.to_record(),
                "solver_success": False,
                _RUNTIME_KEY: float(runtime),
            },
            None,
        )
    runtime = perf_counter() - started
    record: dict[str, Any] = {
        "dose_jumps": completed.dose_jumps,
        "finite_states": completed.finite_states,
        "metrics": {
            "auc_last_mg_h_l": completed.auc_last_mg_h_l,
            "cmax_mg_l": completed.cmax_mg_l,
            "tmax_h": completed.tmax_h,
        },
        "numerical_diagnostics": {
            "maximum_single_clip_mass_mg": completed.maximum_single_clip_mass_mg,
            "maximum_abs_dose_jump_error_mg": (
                completed.maximum_abs_dose_jump_error_mg
            ),
            "maximum_abs_mass_balance_error_mg": (
                completed.maximum_abs_mass_balance_error_mg
            ),
            "minimum_pre_clamp_state_mg": completed.minimum_pre_clamp_state_mg,
        },
        _RUNTIME_KEY: float(runtime),
        "setting": setting.to_record(),
        "solver_counters": completed.solver_counters,
        "solver_success": True,
    }
    return record, completed


def _relative_difference(value: float, reference: float, floor: float) -> float:
    return float(abs(value - reference) / max(abs(reference), floor))


def _gate_record(
    passed: bool, *, value: Any, criterion: str
) -> dict[str, Any]:
    return {"criterion": criterion, "passed": bool(passed), "value": value}


def _comparison_and_gates(
    run: _CompletedRun | None,
    reference: _CompletedRun | None,
    *,
    setting: SolverProtocolSetting,
    case: ConvergenceCase,
    thresholds: ConvergenceThresholds,
) -> tuple[dict[str, Any] | None, dict[str, Any], list[str]]:
    if run is None or reference is None:
        gates = {
            "all_development_targets_passed": False,
            "solver_success": _gate_record(
                False, value=False, criterion="solver must complete"
            ),
        }
        return None, gates, ["unresolved_solver_failure"]

    common_times = _output_grid(case.duration_h, case.output_interval_h)
    curve = np.interp(common_times, run.times_h, run.concentration_mg_l)
    reference_curve = np.interp(
        common_times, reference.times_h, reference.concentration_mg_l
    )
    max_curve_difference = float(np.max(np.abs(curve - reference_curve)))
    curve_limit = max(
        thresholds.curve_absolute_floor_mg_l,
        thresholds.curve_fraction_of_reference_cmax_maximum
        * reference.cmax_mg_l,
    )
    auc_difference = _relative_difference(
        run.auc_last_mg_h_l,
        reference.auc_last_mg_h_l,
        thresholds.curve_absolute_floor_mg_l * case.duration_h,
    )
    cmax_difference = _relative_difference(
        run.cmax_mg_l,
        reference.cmax_mg_l,
        thresholds.curve_absolute_floor_mg_l,
    )
    tmax_difference = float(abs(run.tmax_h - reference.tmax_h))
    comparison = {
        "absolute_max_mass_error_difference_mg": float(
            abs(
                run.maximum_abs_mass_balance_error_mg
                - reference.maximum_abs_mass_balance_error_mg
            )
        ),
        "auc_last_relative_difference_fraction": auc_difference,
        "cmax_relative_difference_fraction": cmax_difference,
        "common_grid_interval_h": case.output_interval_h,
        "maximum_absolute_curve_difference_mg_l": max_curve_difference,
        "maximum_curve_difference_fraction_of_reference_cmax": float(
            max_curve_difference
            / max(reference.cmax_mg_l, thresholds.curve_absolute_floor_mg_l)
        ),
        "reference_label": setting.comparison_reference_label,
        "tmax_absolute_difference_h": tmax_difference,
    }
    gates = {
        "auc_last_convergence": _gate_record(
            auc_difference <= thresholds.endpoint_relative_difference_maximum,
            value=auc_difference,
            criterion=(
                "<= endpoint_relative_difference_maximum"
            ),
        ),
        "cmax_convergence": _gate_record(
            cmax_difference <= thresholds.endpoint_relative_difference_maximum,
            value=cmax_difference,
            criterion="<= endpoint_relative_difference_maximum",
        ),
        "curve_convergence": _gate_record(
            max_curve_difference <= curve_limit,
            value=max_curve_difference,
            criterion=f"<= {curve_limit:.17g} mg/L",
        ),
        "dose_jump_conservation": _gate_record(
            run.maximum_abs_dose_jump_error_mg
            <= thresholds.maximum_dose_jump_error_mg,
            value=run.maximum_abs_dose_jump_error_mg,
            criterion="<= maximum_dose_jump_error_mg",
        ),
        "finite_states": _gate_record(
            run.finite_states, value=run.finite_states, criterion="must be true"
        ),
        "mass_balance": _gate_record(
            run.maximum_abs_mass_balance_error_mg
            <= thresholds.maximum_mass_balance_error_mg,
            value=run.maximum_abs_mass_balance_error_mg,
            criterion="<= maximum_mass_balance_error_mg",
        ),
        "pre_clamp_positivity": _gate_record(
            run.minimum_pre_clamp_state_mg
            >= thresholds.minimum_pre_clamp_state_mg,
            value=run.minimum_pre_clamp_state_mg,
            criterion=">= minimum_pre_clamp_state_mg",
        ),
        "solver_success": _gate_record(
            True, value=True, criterion="solver must complete"
        ),
        "tmax_resolution": _gate_record(
            tmax_difference <= case.output_interval_h,
            value=tmax_difference,
            criterion="<= nominal output interval",
        ),
    }
    gates["all_development_targets_passed"] = all(
        bool(record["passed"])
        for record in gates.values()
        if isinstance(record, dict) and "passed" in record
    )

    invalidity_reasons: list[str] = []
    if not run.finite_states:
        invalidity_reasons.append("nonfinite_state")
    if run.minimum_pre_clamp_state_mg < thresholds.minimum_pre_clamp_state_mg:
        invalidity_reasons.append("unresolved_negative_state")
    if (
        run.maximum_abs_mass_balance_error_mg
        > thresholds.maximum_mass_balance_error_mg
    ):
        invalidity_reasons.append("unresolved_mass_balance_error")
    if (
        run.maximum_abs_dose_jump_error_mg
        > thresholds.maximum_dose_jump_error_mg
    ):
        invalidity_reasons.append("unresolved_dose_jump_error")
    if (
        auc_difference
        >= thresholds.prospective_endpoint_invalidity_boundary
    ):
        invalidity_reasons.append("auc_endpoint_difference_at_least_1_percent")
    if (
        cmax_difference
        >= thresholds.prospective_endpoint_invalidity_boundary
    ):
        invalidity_reasons.append("cmax_endpoint_difference_at_least_1_percent")
    return comparison, gates, invalidity_reasons


def _case_record(case: ConvergenceCase) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "dose_events": [
            {
                "amount_mg": dose.amount_mg,
                "route": dose.route,
                "time_h": dose.time_h,
            }
            for dose in case.doses
        ],
        "drug_name": case.drug.name,
        "duration_h": case.duration_h,
        "metadata": dict(case.metadata),
        "model_id": case.model.model_id,
        "nominal_max_step_h": case.nominal_max_step_h,
        "output_interval_h": case.output_interval_h,
        "patient_id": case.patient.patient_id,
    }


def _without_observed_runtime(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_observed_runtime(item)
            for key, item in value.items()
            if key != _RUNTIME_KEY and key != "scientific_payload_sha256"
        }
    if isinstance(value, list):
        return [_without_observed_runtime(item) for item in value]
    return value


def scientific_payload_sha256(report: Mapping[str, Any]) -> str:
    """Hash deterministic scientific content, excluding wall-clock runtimes."""

    payload = _without_observed_runtime(dict(report))
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def run_solver_convergence(
    cases: Sequence[ConvergenceCase],
    *,
    thresholds: ConvergenceThresholds | None = None,
) -> dict[str, Any]:
    """Run and evaluate the prespecified numerical-convergence matrix.

    Each case obtains its own max-step and output-grid values but shares the
    same tolerance policy.  All run failures remain explicit in the returned
    report.  ``overall_passed`` excludes the deliberately loose diagnostic.
    """

    case_sequence = tuple(cases)
    if not case_sequence:
        raise ValueError("at least one convergence case is required")
    case_ids = tuple(case.case_id for case in case_sequence)
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("convergence case IDs must be unique")
    limits = thresholds or ConvergenceThresholds()
    case_reports: list[dict[str, Any]] = []

    for case in case_sequence:
        settings = default_solver_protocol(
            nominal_max_step_h=case.nominal_max_step_h,
            output_interval_h=case.output_interval_h,
        )
        records: dict[str, dict[str, Any]] = {}
        completed: dict[str, _CompletedRun | None] = {}
        for setting in settings:
            record, result = _run_record(case, setting)
            records[setting.label] = record
            completed[setting.label] = result

        prospective_reasons: list[str] = []
        for setting in settings:
            reference = completed.get(setting.comparison_reference_label)
            comparison, gates, invalidity = _comparison_and_gates(
                completed[setting.label],
                reference,
                setting=setting,
                case=case,
                thresholds=limits,
            )
            records[setting.label]["comparison"] = comparison
            records[setting.label]["gates"] = gates
            records[setting.label]["prospective_invalidity"] = {
                "reasons": invalidity,
                "triggered": bool(invalidity),
            }
            if setting.required_for_case_pass:
                prospective_reasons.extend(
                    f"{setting.label}:{reason}" for reason in invalidity
                )

        required_labels = tuple(
            setting.label for setting in settings if setting.required_for_case_pass
        )
        failed_required = [
            label
            for label in required_labels
            if not bool(records[label]["gates"]["all_development_targets_passed"])
        ]
        case_reports.append(
            {
                "case": _case_record(case),
                "failed_required_run_labels": failed_required,
                "overall_passed": not failed_required and not prospective_reasons,
                "prospective_invalidity": {
                    "reasons": prospective_reasons,
                    "triggered": bool(prospective_reasons),
                },
                "required_run_labels": list(required_labels),
                "runs": [records[setting.label] for setting in settings],
            }
        )

    report: dict[str, Any] = {
        "cases": case_reports,
        "interpretation": {
            "clinical_validation": False,
            "runtime_note": (
                "Observed wall-clock runtime is machine-dependent and excluded "
                "from scientific_payload_sha256."
            ),
            "scope": "numerical implementation verification on synthetic/reference inputs",
        },
        "overall_passed": all(record["overall_passed"] for record in case_reports),
        "schema_version": REPORT_SCHEMA_VERSION,
        "thresholds": limits.to_record(),
    }
    report["scientific_payload_sha256"] = scientific_payload_sha256(report)
    return report


def convergence_report_json(report: Mapping[str, Any]) -> str:
    """Serialize a strict, stably ordered convergence report."""

    return json.dumps(
        dict(report),
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"


def write_convergence_report(report: Mapping[str, Any], path: str | Path) -> Path:
    """Write a report to any explicit project or temporary filesystem path."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(convergence_report_json(report), encoding="utf-8")
    return output
