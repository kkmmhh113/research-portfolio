"""Reproducible quasi-random sampling over bounded parameter spaces."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import qmc

from .parameters import ParameterSpace


@dataclass(frozen=True)
class SamplingDesign:
    method: str
    seed: int
    scramble: bool
    parameter_names: tuple[str, ...]
    unit_samples: np.ndarray
    values: np.ndarray

    def __post_init__(self) -> None:
        unit = np.asarray(self.unit_samples, dtype=np.float64).copy()
        values = np.asarray(self.values, dtype=np.float64).copy()
        if unit.ndim != 2 or values.shape != unit.shape:
            raise ValueError("sampling design arrays must be matching 2D matrices")
        if unit.shape[1] != len(self.parameter_names):
            raise ValueError("parameter_names do not match the sampling columns")
        if np.any(~np.isfinite(unit)) or np.any((unit < 0.0) | (unit > 1.0)):
            raise ValueError("unit samples must be finite and in [0, 1]")
        if np.any(~np.isfinite(values)):
            raise ValueError("physical samples must be finite")
        unit.setflags(write=False)
        values.setflags(write=False)
        object.__setattr__(self, "unit_samples", unit)
        object.__setattr__(self, "values", values)

    @property
    def sample_count(self) -> int:
        return int(self.values.shape[0])

    def records(self) -> list[dict[str, float]]:
        return [
            {
                name: float(value)
                for name, value in zip(self.parameter_names, row, strict=True)
            }
            for row in self.values
        ]


def sample_parameter_space(
    space: ParameterSpace,
    sample_count: int,
    *,
    method: str = "latin_hypercube",
    seed: int = 0,
    scramble: bool = True,
) -> SamplingDesign:
    """Create a deterministic LHS or balanced Sobol design.

    Sobol designs require a power-of-two sample count and use ``random_base2``.
    Failing on other counts avoids silently returning a design with degraded
    balance properties.
    """

    if not isinstance(space, ParameterSpace):
        raise TypeError("space must be a ParameterSpace")
    if isinstance(sample_count, bool) or not isinstance(sample_count, (int, np.integer)):
        raise TypeError("sample_count must be an integer")
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not isinstance(scramble, bool):
        raise TypeError("scramble must be a boolean")
    if method not in {"latin_hypercube", "sobol"}:
        raise ValueError("method must be 'latin_hypercube' or 'sobol'")

    if method == "latin_hypercube":
        engine = qmc.LatinHypercube(d=len(space), scramble=scramble, seed=int(seed))
        unit_samples = engine.random(n=int(sample_count))
    else:
        if sample_count & (sample_count - 1):
            raise ValueError(
                "Sobol sample_count must be a power of two for a balanced design"
            )
        engine = qmc.Sobol(d=len(space), scramble=scramble, seed=int(seed))
        exponent = int(np.log2(sample_count))
        unit_samples = engine.random_base2(m=exponent)

    values = space.unit_to_values(unit_samples)
    return SamplingDesign(
        method=method,
        seed=int(seed),
        scramble=scramble,
        parameter_names=space.names,
        unit_samples=unit_samples,
        values=values,
    )
