"""Empirical prediction intervals and observed interval coverage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def _names(names: Sequence[str] | None, count: int) -> tuple[str, ...]:
    labels = tuple(names) if names is not None else tuple(
        f"output_{index}" for index in range(count)
    )
    if len(labels) != count:
        raise ValueError(f"output_names must contain exactly {count} names")
    if any(not isinstance(label, str) or not label.strip() for label in labels):
        raise ValueError("output_names must contain non-empty strings")
    if len(set(labels)) != len(labels):
        raise ValueError("output_names must be unique")
    return labels


@dataclass(frozen=True)
class EmpiricalIntervals:
    output_names: tuple[str, ...]
    levels: tuple[float, ...]
    lower: np.ndarray
    median: np.ndarray
    upper: np.ndarray
    mean: np.ndarray
    sample_count: int

    def __post_init__(self) -> None:
        output_count = len(self.output_names)
        level_count = len(self.levels)
        if output_count == 0 or len(set(self.output_names)) != output_count:
            raise ValueError("output_names must be non-empty and unique")
        if level_count == 0 or any(
            not np.isfinite(level) or not 0.0 < level < 1.0
            for level in self.levels
        ):
            raise ValueError("levels must be finite and in (0, 1)")
        if any(
            current >= following
            for current, following in zip(self.levels, self.levels[1:])
        ):
            raise ValueError("levels must be strictly increasing")
        for name, shape in (
            ("lower", (level_count, output_count)),
            ("upper", (level_count, output_count)),
            ("median", (output_count,)),
            ("mean", (output_count,)),
        ):
            array = np.asarray(getattr(self, name), dtype=np.float64).copy()
            if array.shape != shape or np.any(~np.isfinite(array)):
                raise ValueError(f"{name} must be finite with shape {shape}")
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        if self.sample_count < 2:
            raise ValueError("sample_count must be at least 2")
        if np.any(self.lower > self.upper):
            raise ValueError("interval lower bounds must not exceed upper bounds")

    def level_index(self, level: float) -> int:
        if not np.isfinite(level):
            raise ValueError("coverage level must be finite")
        matches = [
            index
            for index, available in enumerate(self.levels)
            if np.isclose(level, available, rtol=0.0, atol=1e-12)
        ]
        if len(matches) != 1:
            raise KeyError(f"interval level {level} was not computed")
        return matches[0]

    def records(self) -> list[dict[str, float | int | str]]:
        records: list[dict[str, float | int | str]] = []
        for level_index, level in enumerate(self.levels):
            for output_index, output_name in enumerate(self.output_names):
                records.append(
                    {
                        "output": output_name,
                        "level": level,
                        "lower": float(self.lower[level_index, output_index]),
                        "median": float(self.median[output_index]),
                        "mean": float(self.mean[output_index]),
                        "upper": float(self.upper[level_index, output_index]),
                        "sample_count": self.sample_count,
                    }
                )
        return records


def empirical_intervals(
    predictions: np.ndarray,
    *,
    levels: Sequence[float] = (0.5, 0.9, 0.95),
    output_names: Sequence[str] | None = None,
) -> EmpiricalIntervals:
    """Compute equal-tailed empirical intervals over simulation draws.

    Rows are independent parameter/population draws; columns may be Cmax, AUC,
    Tmax, concentrations at matched times, or any other finite model outputs.
    Missing and infinite predictions are rejected instead of silently dropped.
    """

    values = np.asarray(predictions, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, np.newaxis]
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] == 0:
        raise ValueError("predictions must be a 2D matrix with at least two rows")
    if np.any(~np.isfinite(values)):
        raise ValueError("predictions must contain only finite values")
    level_values = tuple(sorted(float(level) for level in levels))
    if not level_values:
        raise ValueError("at least one interval level is required")
    if len(set(level_values)) != len(level_values):
        raise ValueError("interval levels must be unique")
    if any(not np.isfinite(level) or not 0.0 < level < 1.0 for level in level_values):
        raise ValueError("interval levels must be finite and in (0, 1)")
    labels = _names(output_names, values.shape[1])
    lower = np.vstack(
        [np.quantile(values, (1.0 - level) / 2.0, axis=0) for level in level_values]
    )
    upper = np.vstack(
        [np.quantile(values, 1.0 - (1.0 - level) / 2.0, axis=0) for level in level_values]
    )
    return EmpiricalIntervals(
        output_names=labels,
        levels=level_values,
        lower=lower,
        median=np.median(values, axis=0),
        upper=upper,
        mean=np.mean(values, axis=0),
        sample_count=values.shape[0],
    )


@dataclass(frozen=True)
class CoverageResult:
    level: float
    output_names: tuple[str, ...]
    observed: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    covered: np.ndarray
    covered_count: int
    total_count: int
    coverage_fraction: float

    def __post_init__(self) -> None:
        count = len(self.output_names)
        if count == 0 or len(set(self.output_names)) != count:
            raise ValueError("output_names must be non-empty and unique")
        if not np.isfinite(self.level) or not 0.0 < self.level < 1.0:
            raise ValueError("coverage level must be finite and in (0, 1)")
        for name in ("observed", "lower", "upper"):
            array = np.asarray(getattr(self, name), dtype=np.float64).copy()
            if array.shape != (count,) or np.any(~np.isfinite(array)):
                raise ValueError(f"{name} must be a finite vector matching outputs")
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        covered = np.asarray(self.covered, dtype=bool).copy()
        if covered.shape != (count,):
            raise ValueError("covered mask must match outputs")
        covered.setflags(write=False)
        object.__setattr__(self, "covered", covered)
        if np.any(self.lower > self.upper):
            raise ValueError("coverage lower bounds must not exceed upper bounds")
        expected_mask = (self.observed >= self.lower) & (self.observed <= self.upper)
        if not np.array_equal(covered, expected_mask):
            raise ValueError("covered mask is inconsistent with observations and bounds")
        if self.total_count != count or self.covered_count != int(covered.sum()):
            raise ValueError("coverage counts are inconsistent")
        expected = self.covered_count / self.total_count
        if not np.isclose(self.coverage_fraction, expected, rtol=0.0, atol=1e-15):
            raise ValueError("coverage_fraction is inconsistent")

    def records(self) -> list[dict[str, float | bool | str]]:
        return [
            {
                "output": output_name,
                "level": self.level,
                "observed": float(self.observed[index]),
                "lower": float(self.lower[index]),
                "upper": float(self.upper[index]),
                "covered": bool(self.covered[index]),
            }
            for index, output_name in enumerate(self.output_names)
        ]


def observed_coverage(
    observed: np.ndarray,
    intervals: EmpiricalIntervals,
    *,
    level: float,
) -> CoverageResult:
    """Compare finite matched observations with one empirical interval level."""

    if not isinstance(intervals, EmpiricalIntervals):
        raise TypeError("intervals must be EmpiricalIntervals")
    values = np.asarray(observed, dtype=np.float64)
    if values.shape != (len(intervals.output_names),):
        raise ValueError("observed values must match interval outputs exactly")
    if np.any(~np.isfinite(values)):
        raise ValueError("observed values must contain only finite values")
    index = intervals.level_index(level)
    lower = intervals.lower[index]
    upper = intervals.upper[index]
    covered = (values >= lower) & (values <= upper)
    covered_count = int(covered.sum())
    return CoverageResult(
        level=float(level),
        output_names=intervals.output_names,
        observed=values,
        lower=lower,
        upper=upper,
        covered=covered,
        covered_count=covered_count,
        total_count=values.size,
        coverage_fraction=covered_count / values.size,
    )
