"""Deterministic fitting for transparent empirical PK comparators.

This module accepts already-canonicalized development curves.  It does not
open raw or external outcomes.  Each curve contributes its mean squared log10
residual to the objective, so dense curves cannot dominate sparse curves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import qmc

from ..comparators.compartmental import (
    ORAL_ONE_COMPARTMENT_PARAMETER_NAMES,
    OralOneCompartmentBounds,
    OralOneCompartmentParameters,
    oral_one_compartment_concentration,
)


@dataclass(frozen=True)
class OralConcentrationCurve:
    """One canonical oral parent-concentration curve used for development fit.

    Exact zero-valued predose rows at ``time_h == 0`` are retained in the
    record but explicitly excluded from log fitting.  Other zero-valued rows
    are rejected rather than silently changing a canonicalization decision.
    """

    curve_id: str
    times_h: np.ndarray
    observed_mg_l: np.ndarray
    dose_mg: float

    def __post_init__(self) -> None:
        if not isinstance(self.curve_id, str) or not self.curve_id.strip():
            raise ValueError("curve_id must be a non-empty string")
        if self.curve_id != self.curve_id.strip():
            raise ValueError("curve_id must not contain surrounding whitespace")
        times = np.asarray(self.times_h, dtype=np.float64).copy()
        observed = np.asarray(self.observed_mg_l, dtype=np.float64).copy()
        if times.ndim != 1 or times.size == 0 or observed.shape != times.shape:
            raise ValueError(
                "times_h and observed_mg_l must be matched non-empty vectors"
            )
        if np.any(~np.isfinite(times)) or np.any(times < 0.0):
            raise ValueError("times_h must be finite and nonnegative")
        if np.any(np.diff(times) <= 0.0):
            raise ValueError("times_h must be strictly increasing and unique")
        if np.any(~np.isfinite(observed)) or np.any(observed < 0.0):
            raise ValueError("observed_mg_l must be finite and nonnegative")
        invalid_zero = (observed == 0.0) & (times != 0.0)
        if np.any(invalid_zero):
            raise ValueError(
                "zero observations are allowed only for an explicit time-zero predose row"
            )
        if not np.any(observed > 0.0):
            raise ValueError("each curve must contain at least one positive observation")
        dose = float(self.dose_mg)
        if not np.isfinite(dose) or dose <= 0.0:
            raise ValueError("dose_mg must be finite and > 0")
        times.setflags(write=False)
        observed.setflags(write=False)
        object.__setattr__(self, "times_h", times)
        object.__setattr__(self, "observed_mg_l", observed)
        object.__setattr__(self, "dose_mg", dose)

    @property
    def fit_mask(self) -> np.ndarray:
        return self.observed_mg_l > 0.0

    @property
    def fitted_observation_count(self) -> int:
        return int(np.count_nonzero(self.fit_mask))

    @property
    def excluded_predose_count(self) -> int:
        return int(self.observed_mg_l.size - self.fitted_observation_count)


@dataclass(frozen=True)
class MultistartSettings:
    """Frozen optimizer settings for a deterministic log-parameter fit."""

    start_count: int = 20
    seed: int = 20260902
    maximum_function_evaluations: int = 2000
    ftol: float = 1.0e-10
    xtol: float = 1.0e-10
    gtol: float = 1.0e-10
    boundary_relative_tolerance: float = 1.0e-6
    best_objective_absolute_tolerance: float = 1.0e-10

    def __post_init__(self) -> None:
        if isinstance(self.start_count, bool) or not isinstance(
            self.start_count, (int, np.integer)
        ):
            raise TypeError("start_count must be an integer")
        if self.start_count < 20:
            raise ValueError("oral one-compartment fitting requires at least 20 starts")
        if isinstance(self.seed, bool) or not isinstance(self.seed, (int, np.integer)):
            raise TypeError("seed must be an integer")
        if isinstance(self.maximum_function_evaluations, bool) or not isinstance(
            self.maximum_function_evaluations, (int, np.integer)
        ):
            raise TypeError("maximum_function_evaluations must be an integer")
        if self.maximum_function_evaluations <= 0:
            raise ValueError("maximum_function_evaluations must be > 0")
        for name in (
            "ftol",
            "xtol",
            "gtol",
            "boundary_relative_tolerance",
            "best_objective_absolute_tolerance",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class FitStartDiagnostic:
    """Complete outcome retained for one deterministic optimizer start."""

    start_index: int
    initial_parameters: OralOneCompartmentParameters
    success: bool
    status: int
    message: str
    function_evaluations: int
    jacobian_evaluations: int | None
    objective: float | None
    optimality: float | None
    fitted_parameters: OralOneCompartmentParameters | None
    boundary_hits: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "start_index": self.start_index,
            "initial_parameters": self.initial_parameters.to_dict(),
            "success": self.success,
            "status": self.status,
            "message": self.message,
            "function_evaluations": self.function_evaluations,
            "jacobian_evaluations": self.jacobian_evaluations,
            "objective": self.objective,
            "optimality": self.optimality,
            "fitted_parameters": (
                None
                if self.fitted_parameters is None
                else self.fitted_parameters.to_dict()
            ),
            "boundary_hits": list(self.boundary_hits),
        }


@dataclass(frozen=True)
class OralOneCompartmentFit:
    """Selected fit plus audit diagnostics from every optimizer start."""

    parameters: OralOneCompartmentParameters
    bounds: OralOneCompartmentBounds
    objective: float
    selected_start_index: int
    start_diagnostics: tuple[FitStartDiagnostic, ...]
    curve_mean_squared_log10_error: tuple[tuple[str, float], ...]
    curve_fitted_observation_count: tuple[tuple[str, int], ...]
    curve_excluded_predose_count: tuple[tuple[str, int], ...]
    concentration_floor_mg_l: float
    settings: MultistartSettings

    @property
    def converged_start_count(self) -> int:
        return sum(diagnostic.success for diagnostic in self.start_diagnostics)

    @property
    def convergence_fraction(self) -> float:
        return self.converged_start_count / len(self.start_diagnostics)

    @property
    def near_best_start_count(self) -> int:
        threshold = self.objective + self.settings.best_objective_absolute_tolerance
        return sum(
            diagnostic.success
            and diagnostic.objective is not None
            and diagnostic.objective <= threshold
            for diagnostic in self.start_diagnostics
        )

    @property
    def near_best_parameter_log_span(self) -> tuple[tuple[str, float], ...]:
        """Log-parameter range across converged, objective-equivalent starts."""

        threshold = self.objective + self.settings.best_objective_absolute_tolerance
        values = np.asarray(
            [
                diagnostic.fitted_parameters.as_array()
                for diagnostic in self.start_diagnostics
                if diagnostic.success
                and diagnostic.objective is not None
                and diagnostic.objective <= threshold
                and diagnostic.fitted_parameters is not None
            ],
            dtype=np.float64,
        )
        spans = np.ptp(np.log(values), axis=0)
        return tuple(
            (name, float(span))
            for name, span in zip(
                ORAL_ONE_COMPARTMENT_PARAMETER_NAMES, spans, strict=True
            )
        )

    @property
    def flip_flop_ambiguity_detected(self) -> bool:
        """Whether equivalent starts support both ``ka > ke`` and ``ka < ke``."""

        threshold = self.objective + self.settings.best_objective_absolute_tolerance
        signs = [
            np.sign(
                diagnostic.fitted_parameters.ka_h
                - diagnostic.fitted_parameters.ke_h
            )
            for diagnostic in self.start_diagnostics
            if diagnostic.success
            and diagnostic.objective is not None
            and diagnostic.objective <= threshold
            and diagnostic.fitted_parameters is not None
        ]
        return bool(
            any(sign > 0.0 for sign in signs)
            and any(sign < 0.0 for sign in signs)
        )

    def predict(self, times_h: np.ndarray, *, dose_mg: float) -> np.ndarray:
        return oral_one_compartment_concentration(
            times_h, dose_mg=dose_mg, parameters=self.parameters
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "model": "oral_one_compartment_apparent_f_over_v",
            "parameter_names": list(ORAL_ONE_COMPARTMENT_PARAMETER_NAMES),
            "parameters": self.parameters.to_dict(),
            "bounds": self.bounds.to_dict(),
            "objective_equal_curve_mean_squared_log10_error": self.objective,
            "selected_start_index": self.selected_start_index,
            "converged_start_count": self.converged_start_count,
            "start_count": len(self.start_diagnostics),
            "convergence_fraction": self.convergence_fraction,
            "near_best_start_count": self.near_best_start_count,
            "near_best_parameter_log_span": dict(
                self.near_best_parameter_log_span
            ),
            "flip_flop_ambiguity_detected": self.flip_flop_ambiguity_detected,
            "curve_mean_squared_log10_error": dict(
                self.curve_mean_squared_log10_error
            ),
            "curve_fitted_observation_count": dict(
                self.curve_fitted_observation_count
            ),
            "curve_excluded_predose_count": dict(
                self.curve_excluded_predose_count
            ),
            "concentration_floor_mg_l": self.concentration_floor_mg_l,
            "optimizer": {
                "algorithm": "scipy.optimize.least_squares_trf_linear_loss",
                "parameterization": "natural_log_positive_parameters",
                "start_design": "seeded_latin_hypercube_in_log_bounds",
                "seed": int(self.settings.seed),
                "maximum_function_evaluations": int(
                    self.settings.maximum_function_evaluations
                ),
                "ftol": self.settings.ftol,
                "xtol": self.settings.xtol,
                "gtol": self.settings.gtol,
            },
            "start_diagnostics": [
                diagnostic.to_dict() for diagnostic in self.start_diagnostics
            ],
            "interpretation_boundary": (
                "scale_l_inv is apparent F/V; F and V are not separately estimated; "
                "ka/ke labels require caution when flip_flop_ambiguity_detected is true"
            ),
        }


def _validated_curves(
    curves: Sequence[OralConcentrationCurve],
) -> tuple[OralConcentrationCurve, ...]:
    records = tuple(curves)
    if not records:
        raise ValueError("at least one oral concentration curve is required")
    if any(not isinstance(curve, OralConcentrationCurve) for curve in records):
        raise TypeError("curves must contain only OralConcentrationCurve records")
    identifiers = tuple(curve.curve_id for curve in records)
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("curve_id values must be unique")
    if sum(curve.fitted_observation_count for curve in records) < 3:
        raise ValueError("at least three positive observations are required")
    return records


def equal_curve_log10_residuals(
    curves: Sequence[OralConcentrationCurve],
    parameters: OralOneCompartmentParameters,
    *,
    concentration_floor_mg_l: float = 1.0e-8,
) -> np.ndarray:
    """Build residuals whose squared norm is the macro mean curve MSE.

    For each included point the raw residual is exactly
    ``log10(max(pred, floor) / max(obs, floor))``.  Dividing a curve's residuals
    by ``sqrt(n_points * n_curves)`` makes the squared vector norm equal the
    arithmetic mean of the per-curve mean squared log10 residuals.
    """

    records = _validated_curves(curves)
    if not isinstance(parameters, OralOneCompartmentParameters):
        raise TypeError("parameters must be OralOneCompartmentParameters")
    floor = float(concentration_floor_mg_l)
    if not np.isfinite(floor) or floor <= 0.0:
        raise ValueError("concentration_floor_mg_l must be finite and > 0")
    curve_count = len(records)
    residual_blocks: list[np.ndarray] = []
    for curve in records:
        mask = curve.fit_mask
        predicted = oral_one_compartment_concentration(
            curve.times_h[mask], dose_mg=curve.dose_mg, parameters=parameters
        )
        observed = curve.observed_mg_l[mask]
        log_residual = np.log10(
            np.maximum(predicted, floor) / np.maximum(observed, floor)
        )
        residual_blocks.append(
            log_residual / np.sqrt(log_residual.size * curve_count)
        )
    return np.concatenate(residual_blocks)


def _curve_objectives(
    curves: tuple[OralConcentrationCurve, ...],
    parameters: OralOneCompartmentParameters,
    floor: float,
) -> tuple[tuple[str, float], ...]:
    records: list[tuple[str, float]] = []
    for curve in curves:
        mask = curve.fit_mask
        predicted = oral_one_compartment_concentration(
            curve.times_h[mask], dose_mg=curve.dose_mg, parameters=parameters
        )
        log_residual = np.log10(
            np.maximum(predicted, floor)
            / np.maximum(curve.observed_mg_l[mask], floor)
        )
        records.append((curve.curve_id, float(np.mean(log_residual**2))))
    return tuple(records)


def _boundary_hits(
    log_values: np.ndarray,
    bounds: OralOneCompartmentBounds,
    relative_tolerance: float,
) -> tuple[str, ...]:
    lower = bounds.log_lower
    upper = bounds.log_upper
    tolerance = relative_tolerance * (upper - lower)
    hits: list[str] = []
    for index, name in enumerate(ORAL_ONE_COMPARTMENT_PARAMETER_NAMES):
        if log_values[index] - lower[index] <= tolerance[index]:
            hits.append(f"{name}:lower")
        if upper[index] - log_values[index] <= tolerance[index]:
            hits.append(f"{name}:upper")
    return tuple(hits)


def _log_parameter_starts(
    bounds: OralOneCompartmentBounds,
    settings: MultistartSettings,
) -> np.ndarray:
    design = qmc.LatinHypercube(
        d=len(ORAL_ONE_COMPARTMENT_PARAMETER_NAMES),
        scramble=True,
        seed=int(settings.seed),
    ).random(n=int(settings.start_count))
    return qmc.scale(design, bounds.log_lower, bounds.log_upper)


def fit_oral_one_compartment(
    curves: Sequence[OralConcentrationCurve],
    *,
    bounds: OralOneCompartmentBounds,
    concentration_floor_mg_l: float = 1.0e-8,
    settings: MultistartSettings | None = None,
) -> OralOneCompartmentFit:
    """Fit a shared oral one-compartment comparator on log parameters.

    All starts are deterministic for a given ``MultistartSettings`` record.
    Every start is retained, including optimizer failures.  The selected result
    is the lowest finite objective among starts whose optimizer reports
    success.  If none succeeds, the function fails closed with ``RuntimeError``.
    """

    records = _validated_curves(curves)
    if not isinstance(bounds, OralOneCompartmentBounds):
        raise TypeError("bounds must be OralOneCompartmentBounds")
    floor = float(concentration_floor_mg_l)
    if not np.isfinite(floor) or floor <= 0.0:
        raise ValueError("concentration_floor_mg_l must be finite and > 0")
    optimizer_settings = settings or MultistartSettings()
    if not isinstance(optimizer_settings, MultistartSettings):
        raise TypeError("settings must be MultistartSettings")

    def residual_function(log_values: np.ndarray) -> np.ndarray:
        parameters = OralOneCompartmentParameters.from_array(np.exp(log_values))
        return equal_curve_log10_residuals(
            records,
            parameters,
            concentration_floor_mg_l=floor,
        )

    diagnostics: list[FitStartDiagnostic] = []
    for start_index, initial_log_values in enumerate(
        _log_parameter_starts(bounds, optimizer_settings)
    ):
        initial_parameters = OralOneCompartmentParameters.from_array(
            np.exp(initial_log_values)
        )
        try:
            optimization = least_squares(
                residual_function,
                initial_log_values,
                bounds=(bounds.log_lower, bounds.log_upper),
                method="trf",
                loss="linear",
                max_nfev=int(optimizer_settings.maximum_function_evaluations),
                ftol=optimizer_settings.ftol,
                xtol=optimizer_settings.xtol,
                gtol=optimizer_settings.gtol,
            )
            finite = bool(
                np.all(np.isfinite(optimization.x))
                and np.isfinite(optimization.cost)
                and np.isfinite(optimization.optimality)
            )
            success = bool(optimization.success and finite)
            fitted = (
                OralOneCompartmentParameters.from_array(np.exp(optimization.x))
                if finite
                else None
            )
            objective = (
                float(np.dot(optimization.fun, optimization.fun))
                if finite and np.all(np.isfinite(optimization.fun))
                else None
            )
            hits = (
                _boundary_hits(
                    optimization.x,
                    bounds,
                    optimizer_settings.boundary_relative_tolerance,
                )
                if finite
                else ()
            )
            diagnostics.append(
                FitStartDiagnostic(
                    start_index=start_index,
                    initial_parameters=initial_parameters,
                    success=success,
                    status=int(optimization.status),
                    message=str(optimization.message),
                    function_evaluations=int(optimization.nfev),
                    jacobian_evaluations=(
                        None
                        if optimization.njev is None
                        else int(optimization.njev)
                    ),
                    objective=objective,
                    optimality=(
                        float(optimization.optimality) if finite else None
                    ),
                    fitted_parameters=fitted,
                    boundary_hits=hits,
                )
            )
        except (FloatingPointError, RuntimeError, ValueError) as exc:
            diagnostics.append(
                FitStartDiagnostic(
                    start_index=start_index,
                    initial_parameters=initial_parameters,
                    success=False,
                    status=-1,
                    message=f"{type(exc).__name__}: {exc}",
                    function_evaluations=0,
                    jacobian_evaluations=None,
                    objective=None,
                    optimality=None,
                    fitted_parameters=None,
                    boundary_hits=(),
                )
            )

    candidates = [
        diagnostic
        for diagnostic in diagnostics
        if diagnostic.success
        and diagnostic.objective is not None
        and diagnostic.fitted_parameters is not None
    ]
    if not candidates:
        raise RuntimeError("all oral one-compartment optimizer starts failed")
    selected = min(candidates, key=lambda item: (item.objective, item.start_index))
    assert selected.fitted_parameters is not None
    assert selected.objective is not None
    bounds.validate(selected.fitted_parameters)
    curve_objectives = _curve_objectives(
        records, selected.fitted_parameters, floor
    )
    macro_objective = float(np.mean([value for _, value in curve_objectives]))
    if not np.isclose(
        selected.objective,
        macro_objective,
        rtol=1.0e-10,
        atol=1.0e-14,
    ):
        raise RuntimeError("weighted residual objective is internally inconsistent")

    return OralOneCompartmentFit(
        parameters=selected.fitted_parameters,
        bounds=bounds,
        objective=macro_objective,
        selected_start_index=selected.start_index,
        start_diagnostics=tuple(diagnostics),
        curve_mean_squared_log10_error=curve_objectives,
        curve_fitted_observation_count=tuple(
            (curve.curve_id, curve.fitted_observation_count) for curve in records
        ),
        curve_excluded_predose_count=tuple(
            (curve.curve_id, curve.excluded_predose_count) for curve in records
        ),
        concentration_floor_mg_l=floor,
        settings=optimizer_settings,
    )
