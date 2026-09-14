"""Deterministic parameter definitions and bounded coordinate transforms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np


_TRANSFORMS = {"linear", "log"}


@dataclass(frozen=True)
class ParameterSpec:
    """One named model parameter with an explicit finite analysis domain.

    ``transform="log"`` means that sampling is uniform in log(parameter), not
    that evaluator callables receive log values.  Evaluators always receive the
    physical values represented by ``nominal``, ``lower``, and ``upper``.
    """

    name: str
    nominal: float
    lower: float
    upper: float
    transform: str = "linear"
    unit: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("parameter name must be a non-empty string")
        if self.name != self.name.strip():
            raise ValueError("parameter name must not contain surrounding whitespace")
        if self.transform not in _TRANSFORMS:
            raise ValueError(f"transform must be one of {sorted(_TRANSFORMS)}")
        for field_name in ("nominal", "lower", "upper"):
            value = float(getattr(self, field_name))
            if not np.isfinite(value):
                raise ValueError(f"{self.name}.{field_name} must be finite")
            object.__setattr__(self, field_name, value)
        if self.lower >= self.upper:
            raise ValueError(f"{self.name} must have lower < upper")
        if not self.lower <= self.nominal <= self.upper:
            raise ValueError(f"{self.name}.nominal must lie within its bounds")
        if self.transform == "log" and self.lower <= 0.0:
            raise ValueError(f"{self.name} requires positive bounds for log sampling")
        if not isinstance(self.unit, str):
            raise TypeError("parameter unit must be a string")

    def from_unit_interval(self, values: np.ndarray) -> np.ndarray:
        """Map values in ``[0, 1]`` to physical parameter values."""

        unit_values = np.asarray(values, dtype=np.float64)
        if np.any(~np.isfinite(unit_values)) or np.any(
            (unit_values < 0.0) | (unit_values > 1.0)
        ):
            raise ValueError("unit-interval values must be finite and in [0, 1]")
        if self.transform == "linear":
            return self.lower + unit_values * (self.upper - self.lower)
        log_lower = np.log(self.lower)
        return np.exp(log_lower + unit_values * (np.log(self.upper) - log_lower))

    def to_unit_interval(self, values: np.ndarray) -> np.ndarray:
        """Map physical parameter values to ``[0, 1]`` coordinates."""

        physical = np.asarray(values, dtype=np.float64)
        if np.any(~np.isfinite(physical)) or np.any(
            (physical < self.lower) | (physical > self.upper)
        ):
            raise ValueError(f"{self.name} values must be finite and within bounds")
        if self.transform == "linear":
            return (physical - self.lower) / (self.upper - self.lower)
        return (np.log(physical) - np.log(self.lower)) / (
            np.log(self.upper) - np.log(self.lower)
        )


@dataclass(frozen=True)
class ParameterSpace:
    """Ordered, uniquely named parameter space.

    Order is part of the API: QMC columns, evaluator vectors, Jacobian columns,
    and reports all use the exact specification order supplied here.
    """

    specifications: tuple[ParameterSpec, ...]

    def __init__(self, specifications: Iterable[ParameterSpec]) -> None:
        specs = tuple(specifications)
        object.__setattr__(self, "specifications", specs)
        self.__post_init__()

    def __post_init__(self) -> None:
        if not self.specifications:
            raise ValueError("parameter space must contain at least one parameter")
        if any(not isinstance(spec, ParameterSpec) for spec in self.specifications):
            raise TypeError("all parameter-space entries must be ParameterSpec objects")
        names = self.names
        if len(set(names)) != len(names):
            raise ValueError("parameter names must be unique")

    def __len__(self) -> int:
        return len(self.specifications)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.specifications)

    @property
    def nominal(self) -> np.ndarray:
        return np.asarray(
            [spec.nominal for spec in self.specifications], dtype=np.float64
        )

    @property
    def lower_bounds(self) -> np.ndarray:
        return np.asarray(
            [spec.lower for spec in self.specifications], dtype=np.float64
        )

    @property
    def upper_bounds(self) -> np.ndarray:
        return np.asarray(
            [spec.upper for spec in self.specifications], dtype=np.float64
        )

    @property
    def widths(self) -> np.ndarray:
        return self.upper_bounds - self.lower_bounds

    def validate_values(self, values: np.ndarray, *, label: str = "values") -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (len(self),):
            raise ValueError(f"{label} must have shape ({len(self)},)")
        if np.any(~np.isfinite(array)):
            raise ValueError(f"{label} must contain only finite values")
        if np.any(array < self.lower_bounds) or np.any(array > self.upper_bounds):
            raise ValueError(f"{label} must lie within the parameter bounds")
        return array.copy()

    def unit_to_values(self, unit_samples: np.ndarray) -> np.ndarray:
        samples = np.asarray(unit_samples, dtype=np.float64)
        if samples.ndim == 1:
            if samples.shape != (len(self),):
                raise ValueError(f"unit sample must have shape ({len(self)},)")
        elif samples.ndim == 2:
            if samples.shape[1] != len(self):
                raise ValueError(
                    f"unit sample matrix must have {len(self)} columns"
                )
        else:
            raise ValueError("unit samples must be a one- or two-dimensional array")
        if np.any(~np.isfinite(samples)) or np.any(
            (samples < 0.0) | (samples > 1.0)
        ):
            raise ValueError("unit samples must be finite and in [0, 1]")
        transformed = np.empty_like(samples, dtype=np.float64)
        for index, spec in enumerate(self.specifications):
            transformed[..., index] = spec.from_unit_interval(samples[..., index])
        return transformed

    def values_to_unit(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim == 1:
            self.validate_values(array)
        elif array.ndim == 2:
            if array.shape[1] != len(self):
                raise ValueError(f"value matrix must have {len(self)} columns")
            if np.any(~np.isfinite(array)):
                raise ValueError("values must contain only finite values")
            if np.any(array < self.lower_bounds) or np.any(
                array > self.upper_bounds
            ):
                raise ValueError("values must lie within the parameter bounds")
        else:
            raise ValueError("values must be a one- or two-dimensional array")
        transformed = np.empty_like(array, dtype=np.float64)
        for index, spec in enumerate(self.specifications):
            transformed[..., index] = spec.to_unit_interval(array[..., index])
        return transformed

    def to_mapping(self, values: np.ndarray) -> dict[str, float]:
        checked = self.validate_values(values)
        return {
            name: float(value) for name, value in zip(self.names, checked, strict=True)
        }

    def from_mapping(self, values: Mapping[str, float]) -> np.ndarray:
        if not isinstance(values, Mapping):
            raise TypeError("parameter values must be a mapping")
        supplied = set(values)
        expected = set(self.names)
        if supplied != expected:
            missing = sorted(expected - supplied)
            unknown = sorted(supplied - expected)
            raise ValueError(
                f"parameter mapping keys do not match; missing={missing}, unknown={unknown}"
            )
        return self.validate_values(
            np.asarray([values[name] for name in self.names], dtype=np.float64),
            label="parameter mapping values",
        )

    def index(self, name: str) -> int:
        try:
            return self.names.index(name)
        except ValueError as exc:
            raise KeyError(f"unknown parameter: {name}") from exc
