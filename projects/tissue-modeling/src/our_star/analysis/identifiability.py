"""Finite-difference and SVD-based local identifiability diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .parameters import ParameterSpace


Evaluator = Callable[[np.ndarray], np.ndarray]


def _evaluate(evaluator: Evaluator, values: np.ndarray, expected_size: int | None) -> np.ndarray:
    result = np.asarray(evaluator(values.copy()), dtype=np.float64)
    if result.ndim == 0:
        result = result.reshape(1)
    if result.ndim != 1 or result.size == 0:
        raise ValueError("evaluator must return a non-empty scalar or 1D array")
    if expected_size is not None and result.size != expected_size:
        raise ValueError("evaluator output size changed during finite differences")
    if np.any(~np.isfinite(result)):
        raise FloatingPointError("evaluator returned a non-finite output")
    return result


@dataclass(frozen=True)
class JacobianResult:
    parameter_names: tuple[str, ...]
    output_names: tuple[str, ...]
    point: np.ndarray
    baseline: np.ndarray
    jacobian: np.ndarray
    step_sizes: np.ndarray
    schemes: tuple[str, ...]
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray

    def __post_init__(self) -> None:
        parameter_count = len(self.parameter_names)
        output_count = len(self.output_names)
        arrays = {
            "point": (self.point, (parameter_count,)),
            "baseline": (self.baseline, (output_count,)),
            "jacobian": (self.jacobian, (output_count, parameter_count)),
            "step_sizes": (self.step_sizes, (parameter_count,)),
            "lower_bounds": (self.lower_bounds, (parameter_count,)),
            "upper_bounds": (self.upper_bounds, (parameter_count,)),
        }
        for name, (raw, shape) in arrays.items():
            array = np.asarray(raw, dtype=np.float64).copy()
            if array.shape != shape or np.any(~np.isfinite(array)):
                raise ValueError(f"{name} must be a finite array with shape {shape}")
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        if len(self.schemes) != parameter_count or any(
            scheme not in {"central", "forward", "backward"} for scheme in self.schemes
        ):
            raise ValueError("invalid finite-difference schemes")
        if np.any(self.step_sizes <= 0.0):
            raise ValueError("finite-difference step sizes must be positive")


def finite_difference_jacobian(
    evaluator: Evaluator,
    space: ParameterSpace,
    *,
    point: np.ndarray | None = None,
    output_names: Sequence[str] | None = None,
    relative_step: float = 1e-4,
) -> JacobianResult:
    """Estimate an output-by-parameter Jacobian within explicit bounds.

    Central differences are preferred.  At or near a bound, a full-sized
    one-sided difference is used; the selected scheme is retained in the result
    for auditability.
    """

    if not callable(evaluator):
        raise TypeError("evaluator must be callable")
    if not isinstance(space, ParameterSpace):
        raise TypeError("space must be a ParameterSpace")
    if not np.isfinite(relative_step) or not 0.0 < relative_step < 0.5:
        raise ValueError("relative_step must be finite and in (0, 0.5)")
    values = space.nominal if point is None else space.validate_values(point, label="point")
    baseline = _evaluate(evaluator, values, None)
    if output_names is None:
        labels = tuple(f"output_{index}" for index in range(baseline.size))
    else:
        labels = tuple(output_names)
        if len(labels) != baseline.size:
            raise ValueError("output_names do not match evaluator output size")
        if any(not isinstance(name, str) or not name.strip() for name in labels):
            raise ValueError("output_names must be non-empty strings")
        if len(set(labels)) != len(labels):
            raise ValueError("output_names must be unique")

    jacobian = np.empty((baseline.size, len(space)), dtype=np.float64)
    step_sizes = np.empty(len(space), dtype=np.float64)
    schemes: list[str] = []
    lower = space.lower_bounds
    upper = space.upper_bounds
    widths = upper - lower
    for index in range(len(space)):
        requested = relative_step * widths[index]
        left = values[index] - lower[index]
        right = upper[index] - values[index]
        numerical_floor = 64.0 * np.finfo(float).eps * max(1.0, abs(values[index]))

        if left >= requested and right >= requested:
            step = requested
            plus = values.copy()
            minus = values.copy()
            plus[index] += step
            minus[index] -= step
            derivative = (
                _evaluate(evaluator, plus, baseline.size)
                - _evaluate(evaluator, minus, baseline.size)
            ) / (2.0 * step)
            scheme = "central"
        elif right >= requested:
            step = requested
            plus = values.copy()
            plus[index] += step
            derivative = (_evaluate(evaluator, plus, baseline.size) - baseline) / step
            scheme = "forward"
        elif left >= requested:
            step = requested
            minus = values.copy()
            minus[index] -= step
            derivative = (baseline - _evaluate(evaluator, minus, baseline.size)) / step
            scheme = "backward"
        elif left > numerical_floor and right > numerical_floor:
            step = min(left, right)
            plus = values.copy()
            minus = values.copy()
            plus[index] += step
            minus[index] -= step
            derivative = (
                _evaluate(evaluator, plus, baseline.size)
                - _evaluate(evaluator, minus, baseline.size)
            ) / (2.0 * step)
            scheme = "central"
        elif right > numerical_floor:
            step = right
            plus = values.copy()
            plus[index] += step
            derivative = (_evaluate(evaluator, plus, baseline.size) - baseline) / step
            scheme = "forward"
        elif left > numerical_floor:
            step = left
            minus = values.copy()
            minus[index] -= step
            derivative = (baseline - _evaluate(evaluator, minus, baseline.size)) / step
            scheme = "backward"
        else:
            raise FloatingPointError(
                f"no numerically resolvable finite-difference step for {space.names[index]}"
            )
        if np.any(~np.isfinite(derivative)):
            raise FloatingPointError(
                f"non-finite derivative for parameter {space.names[index]}"
            )
        jacobian[:, index] = derivative
        step_sizes[index] = step
        schemes.append(scheme)

    return JacobianResult(
        parameter_names=space.names,
        output_names=labels,
        point=values,
        baseline=baseline,
        jacobian=jacobian,
        step_sizes=step_sizes,
        schemes=tuple(schemes),
        lower_bounds=lower,
        upper_bounds=upper,
    )


@dataclass(frozen=True)
class IdentifiabilityDiagnostics:
    parameter_names: tuple[str, ...]
    output_names: tuple[str, ...]
    normalized_jacobian: np.ndarray
    singular_values: np.ndarray
    right_singular_vectors: np.ndarray
    null_space_vectors: np.ndarray
    effective_rank: int
    parameter_count: int
    singular_value_threshold: float
    condition_number: float
    locally_identifiable: bool

    def __post_init__(self) -> None:
        for name in (
            "normalized_jacobian",
            "singular_values",
            "right_singular_vectors",
            "null_space_vectors",
        ):
            array = np.asarray(getattr(self, name), dtype=np.float64).copy()
            if np.any(~np.isfinite(array)):
                raise ValueError(f"{name} must contain only finite values")
            array.setflags(write=False)
            object.__setattr__(self, name, array)

    def weakest_direction(self) -> list[dict[str, float | int | str]]:
        """Rank parameters contributing to the weakest local direction."""

        vector = self.right_singular_vectors[-1]
        order = np.argsort(-np.abs(vector), kind="stable")
        return [
            {
                "rank": rank,
                "parameter": self.parameter_names[index],
                "loading": float(vector[index]),
                "absolute_loading": float(abs(vector[index])),
            }
            for rank, index in enumerate(order, start=1)
        ]


def identifiability_diagnostics(
    result: JacobianResult,
    *,
    parameter_scales: np.ndarray | None = None,
    output_scales: np.ndarray | None = None,
    relative_singular_value_threshold: float = 1e-6,
) -> IdentifiabilityDiagnostics:
    """Diagnose local practical identifiability using a dimensionless SVD."""

    if not isinstance(result, JacobianResult):
        raise TypeError("result must be a JacobianResult")
    if not np.isfinite(relative_singular_value_threshold) or not (
        0.0 < relative_singular_value_threshold < 1.0
    ):
        raise ValueError("relative_singular_value_threshold must be in (0, 1)")
    parameter_count = len(result.parameter_names)
    output_count = len(result.output_names)
    if parameter_scales is None:
        p_scales = result.upper_bounds - result.lower_bounds
    else:
        p_scales = np.asarray(parameter_scales, dtype=np.float64)
    if output_scales is None:
        o_scales = np.maximum(np.abs(result.baseline), 1.0)
    else:
        o_scales = np.asarray(output_scales, dtype=np.float64)
    if p_scales.shape != (parameter_count,) or np.any(~np.isfinite(p_scales)) or np.any(
        p_scales <= 0.0
    ):
        raise ValueError("parameter_scales must be finite, positive, and correctly sized")
    if o_scales.shape != (output_count,) or np.any(~np.isfinite(o_scales)) or np.any(
        o_scales <= 0.0
    ):
        raise ValueError("output_scales must be finite, positive, and correctly sized")

    normalized = result.jacobian * p_scales[np.newaxis, :] / o_scales[:, np.newaxis]
    _left, singular_values, right = np.linalg.svd(normalized, full_matrices=True)
    largest = float(singular_values[0]) if singular_values.size else 0.0
    threshold = largest * relative_singular_value_threshold
    effective_rank = int(np.count_nonzero(singular_values > threshold)) if largest > 0 else 0
    locally_identifiable = effective_rank == parameter_count
    if locally_identifiable:
        condition = float(singular_values[0] / singular_values[parameter_count - 1])
    else:
        condition = float("inf")
    null_space = right[effective_rank:, :]
    return IdentifiabilityDiagnostics(
        parameter_names=result.parameter_names,
        output_names=result.output_names,
        normalized_jacobian=normalized,
        singular_values=singular_values,
        right_singular_vectors=right,
        null_space_vectors=null_space,
        effective_rank=effective_rank,
        parameter_count=parameter_count,
        singular_value_threshold=threshold,
        condition_number=condition,
        locally_identifiable=locally_identifiable,
    )
