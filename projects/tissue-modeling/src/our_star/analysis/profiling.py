"""One-dimensional profile-likelihood utility with bounded nuisance fitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2

from .parameters import ParameterSpace


Objective = Callable[[np.ndarray], float]


def _objective_value(objective: Objective, values: np.ndarray) -> float:
    raw = np.asarray(objective(values.copy()), dtype=np.float64)
    if raw.ndim != 0:
        raise ValueError("negative-log-likelihood objective must return one scalar")
    value = float(raw)
    if not np.isfinite(value):
        raise FloatingPointError("negative-log-likelihood objective returned non-finite value")
    return value


@dataclass(frozen=True)
class ProfileLikelihoodResult:
    parameter_name: str
    parameter_names: tuple[str, ...]
    grid: np.ndarray
    negative_log_likelihood: np.ndarray
    delta_negative_log_likelihood: np.ndarray
    likelihood_ratio: np.ndarray
    nuisance_optima: np.ndarray
    confidence_level: float
    likelihood_ratio_threshold: float
    accepted: np.ndarray
    confidence_interval: tuple[float, float]
    interval_bounded_by_grid: bool
    accepted_region_connected: bool
    grid_mle: float

    def __post_init__(self) -> None:
        grid = np.asarray(self.grid, dtype=np.float64).copy()
        size = grid.size
        expected_vectors = {
            "negative_log_likelihood": self.negative_log_likelihood,
            "delta_negative_log_likelihood": self.delta_negative_log_likelihood,
            "likelihood_ratio": self.likelihood_ratio,
        }
        if grid.ndim != 1 or size < 3 or np.any(~np.isfinite(grid)):
            raise ValueError("profile grid must be a finite vector with at least 3 values")
        if np.any(np.diff(grid) <= 0.0):
            raise ValueError("profile grid must be strictly increasing")
        grid.setflags(write=False)
        object.__setattr__(self, "grid", grid)
        for name, raw in expected_vectors.items():
            array = np.asarray(raw, dtype=np.float64).copy()
            if array.shape != (size,) or np.any(~np.isfinite(array)):
                raise ValueError(f"{name} must be a finite vector matching the grid")
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        accepted = np.asarray(self.accepted, dtype=bool).copy()
        if accepted.shape != (size,) or not np.any(accepted):
            raise ValueError("accepted profile mask must match the grid and contain a value")
        accepted.setflags(write=False)
        object.__setattr__(self, "accepted", accepted)
        optima = np.asarray(self.nuisance_optima, dtype=np.float64).copy()
        if optima.shape != (size, len(self.parameter_names)) or np.any(~np.isfinite(optima)):
            raise ValueError("nuisance_optima must be finite and match grid x parameters")
        optima.setflags(write=False)
        object.__setattr__(self, "nuisance_optima", optima)


def profile_likelihood(
    objective: Objective,
    space: ParameterSpace,
    parameter_name: str,
    *,
    grid: Sequence[float] | None = None,
    grid_size: int = 21,
    initial: np.ndarray | None = None,
    confidence_level: float = 0.95,
    max_iterations: int = 1000,
) -> ProfileLikelihoodResult:
    """Profile a scalar negative log-likelihood over one fixed parameter.

    At every grid point, all remaining parameters are optimized independently
    from the same initial vector using bounded L-BFGS-B.  The returned confidence
    set uses ``2 * (NLL - min(NLL)) <= chi2.ppf(confidence_level, 1)``.  A result
    touching either grid boundary is explicitly marked as not grid-bounded.
    """

    if not callable(objective):
        raise TypeError("objective must be callable")
    if not isinstance(space, ParameterSpace):
        raise TypeError("space must be a ParameterSpace")
    target = space.index(parameter_name)
    if isinstance(grid_size, bool) or not isinstance(grid_size, (int, np.integer)):
        raise TypeError("grid_size must be an integer")
    if grid_size < 3:
        raise ValueError("grid_size must be at least 3")
    if not np.isfinite(confidence_level) or not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in (0, 1)")
    if isinstance(max_iterations, bool) or not isinstance(
        max_iterations, (int, np.integer)
    ) or max_iterations <= 0:
        raise ValueError("max_iterations must be a positive integer")
    start = space.nominal if initial is None else space.validate_values(initial, label="initial")
    spec = space.specifications[target]
    if grid is None:
        if spec.transform == "log":
            profile_grid = np.geomspace(spec.lower, spec.upper, grid_size)
        else:
            profile_grid = np.linspace(spec.lower, spec.upper, grid_size)
    else:
        profile_grid = np.asarray(tuple(grid), dtype=np.float64)
        if profile_grid.ndim != 1 or profile_grid.size < 3:
            raise ValueError("grid must contain at least 3 values")
        if np.any(~np.isfinite(profile_grid)) or np.any(np.diff(profile_grid) <= 0.0):
            raise ValueError("grid must be finite and strictly increasing")
        if profile_grid[0] < spec.lower or profile_grid[-1] > spec.upper:
            raise ValueError("grid must lie within the profiled parameter bounds")

    nuisance = np.asarray(
        [index for index in range(len(space)) if index != target], dtype=int
    )
    profile_values = np.empty(profile_grid.size, dtype=np.float64)
    optima = np.empty((profile_grid.size, len(space)), dtype=np.float64)
    nuisance_bounds = [
        (space.lower_bounds[index], space.upper_bounds[index]) for index in nuisance
    ]
    for grid_index, fixed_value in enumerate(profile_grid):
        if nuisance.size == 0:
            full = np.asarray([fixed_value], dtype=np.float64)
            profile_values[grid_index] = _objective_value(objective, full)
            optima[grid_index] = full
            continue

        def nuisance_objective(nuisance_values: np.ndarray) -> float:
            full = start.copy()
            full[target] = fixed_value
            full[nuisance] = nuisance_values
            return _objective_value(objective, full)

        optimization = minimize(
            nuisance_objective,
            start[nuisance],
            method="L-BFGS-B",
            bounds=nuisance_bounds,
            options={"maxiter": int(max_iterations)},
        )
        if not optimization.success or not np.isfinite(optimization.fun):
            raise RuntimeError(
                f"nuisance optimization failed at {parameter_name}={fixed_value}: "
                f"{optimization.message}"
            )
        full = start.copy()
        full[target] = fixed_value
        full[nuisance] = optimization.x
        full = space.validate_values(full, label="profile optimum")
        profile_values[grid_index] = _objective_value(objective, full)
        optima[grid_index] = full

    minimum = float(np.min(profile_values))
    delta = profile_values - minimum
    likelihood_ratio = 2.0 * delta
    threshold = float(chi2.ppf(confidence_level, df=1))
    accepted = likelihood_ratio <= threshold + 64.0 * np.finfo(float).eps
    accepted_indices = np.flatnonzero(accepted)
    interval = (
        float(profile_grid[accepted_indices[0]]),
        float(profile_grid[accepted_indices[-1]]),
    )
    bounded = bool(accepted_indices[0] > 0 and accepted_indices[-1] < profile_grid.size - 1)
    connected = bool(
        np.array_equal(
            accepted_indices,
            np.arange(accepted_indices[0], accepted_indices[-1] + 1),
        )
    )
    return ProfileLikelihoodResult(
        parameter_name=parameter_name,
        parameter_names=space.names,
        grid=profile_grid,
        negative_log_likelihood=profile_values,
        delta_negative_log_likelihood=delta,
        likelihood_ratio=likelihood_ratio,
        nuisance_optima=optima,
        confidence_level=float(confidence_level),
        likelihood_ratio_threshold=threshold,
        accepted=accepted,
        confidence_interval=interval,
        interval_bounded_by_grid=bounded,
        accepted_region_connected=connected,
        grid_mle=float(profile_grid[int(np.argmin(profile_values))]),
    )
