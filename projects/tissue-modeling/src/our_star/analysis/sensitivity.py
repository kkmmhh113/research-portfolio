"""Partial rank correlation for multivariable PBPK sensitivity analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import stats


def _validated_names(names: Sequence[str], count: int, label: str) -> tuple[str, ...]:
    result = tuple(names)
    if len(result) != count:
        raise ValueError(f"{label} must contain exactly {count} names")
    if any(not isinstance(name, str) or not name.strip() for name in result):
        raise ValueError(f"{label} must contain non-empty strings")
    if len(set(result)) != len(result):
        raise ValueError(f"{label} must be unique")
    return result


@dataclass(frozen=True)
class SensitivityResult:
    """PRCC coefficients and two-sided approximate significance values."""

    parameter_names: tuple[str, ...]
    output_names: tuple[str, ...]
    coefficients: np.ndarray
    p_values: np.ndarray
    sample_count: int
    degrees_of_freedom: int

    def __post_init__(self) -> None:
        expected = (len(self.parameter_names), len(self.output_names))
        coefficients = np.asarray(self.coefficients, dtype=np.float64).copy()
        p_values = np.asarray(self.p_values, dtype=np.float64).copy()
        if coefficients.shape != expected or p_values.shape != expected:
            raise ValueError("sensitivity matrices do not match their names")
        if np.any(~np.isfinite(coefficients)) or np.any(~np.isfinite(p_values)):
            raise ValueError("sensitivity results must be finite")
        if np.any(np.abs(coefficients) > 1.0 + 1e-12):
            raise ValueError("correlations must lie in [-1, 1]")
        if np.any((p_values < 0.0) | (p_values > 1.0)):
            raise ValueError("p-values must lie in [0, 1]")
        coefficients.setflags(write=False)
        p_values.setflags(write=False)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "p_values", p_values)

    def ranked(self, output_name: str) -> list[dict[str, float | int | str]]:
        """Return stable descending absolute-effect ranking for one output."""

        try:
            output_index = self.output_names.index(output_name)
        except ValueError as exc:
            raise KeyError(f"unknown sensitivity output: {output_name}") from exc
        order = np.argsort(
            -np.abs(self.coefficients[:, output_index]), kind="stable"
        )
        return [
            {
                "rank": rank,
                "parameter": self.parameter_names[index],
                "output": output_name,
                "prcc": float(self.coefficients[index, output_index]),
                "absolute_prcc": float(abs(self.coefficients[index, output_index])),
                "p_value": float(self.p_values[index, output_index]),
            }
            for rank, index in enumerate(order, start=1)
        ]


def partial_rank_correlation(
    samples: np.ndarray,
    outputs: np.ndarray,
    *,
    parameter_names: Sequence[str] | None = None,
    output_names: Sequence[str] | None = None,
) -> SensitivityResult:
    """Compute partial rank correlation coefficients (PRCC).

    Each ranked parameter and ranked output is residualized against all other
    ranked parameters before their Pearson correlation is calculated.  The
    implementation rejects constant or rank-deficient inputs rather than
    returning misleading NaNs.  Approximate p-values use the conventional
    partial-correlation t statistic and are descriptive, not multiplicity
    corrected confirmatory inference.
    """

    x = np.asarray(samples, dtype=np.float64)
    y = np.asarray(outputs, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] == 0:
        raise ValueError("samples must be a non-empty 2D matrix")
    if y.ndim == 1:
        y = y[:, np.newaxis]
    if y.ndim != 2 or y.shape[1] == 0:
        raise ValueError("outputs must be a non-empty 1D or 2D array")
    if x.shape[0] != y.shape[0]:
        raise ValueError("samples and outputs must contain the same number of rows")
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
        raise ValueError("samples and outputs must contain only finite values")
    row_count, parameter_count = x.shape
    degrees_of_freedom = row_count - parameter_count - 1
    if degrees_of_freedom <= 0:
        raise ValueError(
            "PRCC requires at least parameter_count + 2 samples"
        )

    parameter_labels = _validated_names(
        (
            tuple(f"parameter_{i}" for i in range(parameter_count))
            if parameter_names is None
            else parameter_names
        ),
        parameter_count,
        "parameter_names",
    )
    output_labels = _validated_names(
        (
            tuple(f"output_{i}" for i in range(y.shape[1]))
            if output_names is None
            else output_names
        ),
        y.shape[1],
        "output_names",
    )
    if any(np.ptp(x[:, index]) == 0.0 for index in range(parameter_count)):
        raise ValueError("PRCC is undefined for a constant parameter column")
    if any(np.ptp(y[:, index]) == 0.0 for index in range(y.shape[1])):
        raise ValueError("PRCC is undefined for a constant output column")

    ranked_x = np.column_stack(
        [stats.rankdata(x[:, index], method="average") for index in range(parameter_count)]
    )
    ranked_y = np.column_stack(
        [stats.rankdata(y[:, index], method="average") for index in range(y.shape[1])]
    )
    centered_ranked_x = ranked_x - ranked_x.mean(axis=0)
    if np.linalg.matrix_rank(centered_ranked_x) < parameter_count:
        raise ValueError("ranked parameter design is collinear")

    coefficients = np.empty((parameter_count, y.shape[1]), dtype=np.float64)
    p_values = np.empty_like(coefficients)
    for parameter_index in range(parameter_count):
        controls = np.delete(ranked_x, parameter_index, axis=1)
        design = np.column_stack((np.ones(row_count), controls))
        parameter_residual = ranked_x[:, parameter_index] - design @ np.linalg.lstsq(
            design, ranked_x[:, parameter_index], rcond=None
        )[0]
        if np.linalg.norm(parameter_residual) <= np.finfo(float).eps:
            raise ValueError(
                f"parameter {parameter_labels[parameter_index]} has zero residual variance"
            )
        for output_index in range(y.shape[1]):
            output_residual = ranked_y[:, output_index] - design @ np.linalg.lstsq(
                design, ranked_y[:, output_index], rcond=None
            )[0]
            denominator = np.linalg.norm(parameter_residual) * np.linalg.norm(
                output_residual
            )
            if denominator <= np.finfo(float).eps:
                raise ValueError(
                    f"output {output_labels[output_index]} has zero conditional variance"
                )
            coefficient = float(np.dot(parameter_residual, output_residual) / denominator)
            coefficient = float(np.clip(coefficient, -1.0, 1.0))
            coefficients[parameter_index, output_index] = coefficient
            if abs(coefficient) >= 1.0:
                p_value = 0.0
            else:
                statistic = coefficient * np.sqrt(
                    degrees_of_freedom / max(1.0 - coefficient**2, np.finfo(float).tiny)
                )
                p_value = float(
                    2.0 * stats.t.sf(abs(statistic), df=degrees_of_freedom)
                )
            p_values[parameter_index, output_index] = p_value

    return SensitivityResult(
        parameter_names=parameter_labels,
        output_names=output_labels,
        coefficients=coefficients,
        p_values=p_values,
        sample_count=row_count,
        degrees_of_freedom=degrees_of_freedom,
    )
