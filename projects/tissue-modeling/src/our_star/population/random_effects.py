"""Named, auditable random effects for post-v0.3 virtual populations.

The legacy :mod:`our_star.virtual_population` sampler intentionally remains
unchanged.  This module provides a stricter covariance boundary for future
population models: covariance rows are named, positive definiteness is strict,
and every latent draw retains the independent-normal draw from which it was
constructed.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Sequence

import numpy as np


def _validated_names(names: Sequence[str], *, label: str) -> tuple[str, ...]:
    result = tuple(names)
    if not result:
        raise ValueError(f"{label} must not be empty")
    if any(not isinstance(name, str) or not name.strip() for name in result):
        raise ValueError(f"{label} must contain non-empty strings")
    if any(name != name.strip() for name in result):
        raise ValueError(f"{label} must not contain surrounding whitespace")
    if len(set(result)) != len(result):
        raise ValueError(f"{label} must be unique")
    return result


def _validated_nonnegative_integer(value: int, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{label} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{label} must be non-negative")
    return result


@dataclass(frozen=True)
class RandomEffectSpec:
    """A named multivariate-normal latent-effect specification.

    ``covariance_names`` records the row/column order of a covariance loaded
    from an artifact.  When supplied it must exactly match ``names``; accepting
    a permutation silently would attach covariance entries to the wrong
    physiological mechanisms.
    """

    names: tuple[str, ...]
    covariance: np.ndarray
    mean: np.ndarray | None = None
    covariance_names: tuple[str, ...] | None = None
    max_condition_number: float = 1.0e8
    symmetry_tolerance: float = 1.0e-12

    def __post_init__(self) -> None:
        names = _validated_names(self.names, label="random-effect names")
        covariance_names = (
            names
            if self.covariance_names is None
            else _validated_names(
                self.covariance_names, label="covariance names"
            )
        )
        if covariance_names != names:
            raise ValueError(
                "covariance name order must exactly match random-effect names"
            )

        dimension = len(names)
        covariance = np.asarray(self.covariance, dtype=np.float64).copy()
        if covariance.shape != (dimension, dimension):
            raise ValueError(
                "covariance must have shape "
                f"{(dimension, dimension)}, got {covariance.shape}"
            )
        if np.any(~np.isfinite(covariance)):
            raise ValueError("covariance must contain only finite values")

        tolerance = float(self.symmetry_tolerance)
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("symmetry_tolerance must be finite and non-negative")
        if not np.allclose(
            covariance,
            covariance.T,
            rtol=0.0,
            atol=tolerance,
        ):
            raise ValueError("covariance must be symmetric")
        # Average only within the accepted numerical tolerance so eigensolvers
        # and Cholesky receive a mathematically symmetric matrix.
        covariance = 0.5 * (covariance + covariance.T)

        try:
            cholesky = np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as error:
            raise ValueError("covariance must be strictly positive definite") from error

        maximum_condition = float(self.max_condition_number)
        if not np.isfinite(maximum_condition) or maximum_condition < 1.0:
            raise ValueError("max_condition_number must be finite and at least one")
        condition_number = float(np.linalg.cond(covariance))
        if not np.isfinite(condition_number):
            raise ValueError("covariance condition number must be finite")
        if condition_number > maximum_condition:
            raise ValueError(
                "covariance is too ill-conditioned: "
                f"{condition_number:.6g} > {maximum_condition:.6g}"
            )

        if self.mean is None:
            mean = np.zeros(dimension, dtype=np.float64)
        else:
            mean = np.asarray(self.mean, dtype=np.float64).copy()
        if mean.shape != (dimension,) or np.any(~np.isfinite(mean)):
            raise ValueError(
                f"mean must be a finite vector with shape {(dimension,)}"
            )

        covariance.setflags(write=False)
        mean.setflags(write=False)
        cholesky.setflags(write=False)
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "covariance_names", covariance_names)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "max_condition_number", maximum_condition)
        object.__setattr__(self, "symmetry_tolerance", tolerance)
        object.__setattr__(self, "_condition_number", condition_number)
        object.__setattr__(self, "_cholesky", cholesky)

    @property
    def dimension(self) -> int:
        return len(self.names)

    @property
    def condition_number(self) -> float:
        return self._condition_number

    @property
    def cholesky(self) -> np.ndarray:
        """Read-only lower Cholesky factor in the declared name order."""

        return self._cholesky

    @property
    def fingerprint(self) -> str:
        metadata = json.dumps(
            {
                "names": self.names,
                "mean": self.mean.tolist(),
                "covariance": self.covariance.tolist(),
                "max_condition_number": self.max_condition_number,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return sha256(metadata).hexdigest()

    @classmethod
    def from_named_covariance(
        cls,
        *,
        names: Sequence[str],
        covariance: dict[str, dict[str, float]],
        mean: Sequence[float] | None = None,
        max_condition_number: float = 1.0e8,
    ) -> "RandomEffectSpec":
        """Construct from an insertion-ordered square named covariance.

        Both outer and inner key orders are checked.  This is deliberately
        stricter than key-set equality because order is part of the covariance
        artifact contract.
        """

        ordered_names = _validated_names(names, label="random-effect names")
        if tuple(covariance) != ordered_names:
            raise ValueError(
                "named covariance row order must exactly match random-effect names"
            )
        rows: list[list[float]] = []
        for row_name in ordered_names:
            row = covariance[row_name]
            if tuple(row) != ordered_names:
                raise ValueError(
                    "named covariance column order must exactly match "
                    "random-effect names"
                )
            rows.append([float(row[column]) for column in ordered_names])
        return cls(
            names=ordered_names,
            covariance=np.asarray(rows, dtype=np.float64),
            mean=None if mean is None else np.asarray(mean, dtype=np.float64),
            covariance_names=ordered_names,
            max_condition_number=max_condition_number,
        )


@dataclass(frozen=True)
class LatentDraws:
    """Correlated draws plus their reproducibility trace."""

    names: tuple[str, ...]
    values: np.ndarray
    standard_normal_draws: np.ndarray
    seed: int
    bit_generator: str
    random_effect_fingerprint: str

    def __post_init__(self) -> None:
        names = _validated_names(self.names, label="latent-draw names")
        seed = _validated_nonnegative_integer(self.seed, label="seed")
        values = np.asarray(self.values, dtype=np.float64).copy()
        standard = np.asarray(
            self.standard_normal_draws, dtype=np.float64
        ).copy()
        if values.ndim != 2 or values.shape[1] != len(names):
            raise ValueError(
                "latent values must be a 2D matrix matching the declared names"
            )
        if values.shape[0] == 0:
            raise ValueError("at least one latent draw is required")
        if standard.shape != values.shape:
            raise ValueError("standard-normal trace must match latent values")
        if np.any(~np.isfinite(values)) or np.any(~np.isfinite(standard)):
            raise ValueError("latent draws and traces must be finite")
        if not isinstance(self.bit_generator, str) or not self.bit_generator.strip():
            raise ValueError("bit_generator is required")
        if (
            not isinstance(self.random_effect_fingerprint, str)
            or len(self.random_effect_fingerprint) != 64
        ):
            raise ValueError("a SHA-256 random-effect fingerprint is required")
        values.setflags(write=False)
        standard.setflags(write=False)
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "standard_normal_draws", standard)
        object.__setattr__(self, "seed", seed)

    @property
    def count(self) -> int:
        return int(self.values.shape[0])

    def records(self) -> list[dict[str, object]]:
        return [
            {
                "draw_index": index,
                "latent_effects": {
                    name: float(value)
                    for name, value in zip(self.names, row, strict=True)
                },
                "standard_normals": {
                    name: float(value)
                    for name, value in zip(
                        self.names, self.standard_normal_draws[index], strict=True
                    )
                },
            }
            for index, row in enumerate(self.values)
        ]


def draw_latent_effects(
    specification: RandomEffectSpec,
    count: int,
    *,
    seed: int = 0,
) -> LatentDraws:
    """Draw reproducible correlated effects without using global RNG state."""

    if not isinstance(specification, RandomEffectSpec):
        raise TypeError("specification must be a RandomEffectSpec")
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)):
        raise TypeError("count must be an integer")
    count = int(count)
    if count <= 0:
        raise ValueError("count must be positive")
    seed = _validated_nonnegative_integer(seed, label="seed")

    rng = np.random.default_rng(seed)
    standard = rng.standard_normal((count, specification.dimension))
    values = specification.mean + standard @ specification.cholesky.T
    return LatentDraws(
        names=specification.names,
        values=values,
        standard_normal_draws=standard,
        seed=seed,
        bit_generator=rng.bit_generator.__class__.__name__,
        random_effect_fingerprint=specification.fingerprint,
    )


__all__ = ["LatentDraws", "RandomEffectSpec", "draw_latent_effects"]
