from __future__ import annotations

import numpy as np
import pytest

from our_star.analysis import (
    ParameterSpace,
    ParameterSpec,
    empirical_intervals,
    finite_difference_jacobian,
    identifiability_diagnostics,
    observed_coverage,
    partial_rank_correlation,
    profile_likelihood,
    sample_parameter_space,
)


def _space() -> ParameterSpace:
    return ParameterSpace(
        (
            ParameterSpec("dissolution_rate_h", 2.0, 0.2, 20.0, "log", "/h"),
            ParameterSpec("gastric_emptying_rate_h", 2.5, 0.3, 8.0, "log", "/h"),
            ParameterSpec("absorption_scale", 1.0, 0.25, 4.0, "log"),
        )
    )


def test_parameter_space_and_qmc_designs_are_bounded_and_reproducible() -> None:
    space = _space()
    first = sample_parameter_space(space, 32, method="latin_hypercube", seed=17)
    second = sample_parameter_space(space, 32, method="latin_hypercube", seed=17)
    np.testing.assert_array_equal(first.unit_samples, second.unit_samples)
    np.testing.assert_array_equal(first.values, second.values)
    assert np.all(first.values >= space.lower_bounds)
    assert np.all(first.values <= space.upper_bounds)
    np.testing.assert_allclose(
        space.unit_to_values(space.values_to_unit(first.values)), first.values
    )

    sobol = sample_parameter_space(space, 16, method="sobol", seed=2026)
    assert sobol.values.shape == (16, 3)
    with pytest.raises(ValueError, match="power of two"):
        sample_parameter_space(space, 15, method="sobol", seed=2026)


def test_parameter_space_rejects_ambiguous_or_invalid_definitions() -> None:
    with pytest.raises(ValueError, match="positive bounds"):
        ParameterSpec("bad_log", 1.0, 0.0, 2.0, "log")
    duplicate = ParameterSpec("duplicate", 1.0, 0.5, 2.0)
    with pytest.raises(ValueError, match="unique"):
        ParameterSpace((duplicate, duplicate))
    space = _space()
    with pytest.raises(ValueError, match="keys do not match"):
        space.from_mapping({"dissolution_rate_h": 1.0})
    with pytest.raises(ValueError, match="within"):
        space.validate_values(np.asarray([100.0, 1.0, 1.0]))


def test_prcc_recovers_signed_rank_effects_and_stable_ranking() -> None:
    design = sample_parameter_space(_space(), 512, method="sobol", seed=44)
    unit = design.unit_samples
    outputs = np.column_stack(
        (
            4.0 * unit[:, 0] - 2.0 * unit[:, 1] + 0.05 * unit[:, 2],
            3.0 * unit[:, 2] + 0.05 * unit[:, 0],
        )
    )
    result = partial_rank_correlation(
        design.values,
        outputs,
        parameter_names=design.parameter_names,
        output_names=("cmax_mg_l", "auc_last_mg_h_l"),
    )
    assert result.coefficients[0, 0] > 0.9
    assert result.coefficients[1, 0] < -0.9
    assert result.coefficients[2, 1] > 0.9
    assert result.ranked("auc_last_mg_h_l")[0]["parameter"] == "absorption_scale"

    with pytest.raises(ValueError, match="constant output"):
        partial_rank_correlation(design.values, np.ones(512))
    with pytest.raises(ValueError, match="finite"):
        invalid = outputs.copy()
        invalid[0, 0] = np.nan
        partial_rank_correlation(design.values, invalid)


def test_finite_difference_jacobian_and_svd_detect_identifiability() -> None:
    space = ParameterSpace(
        (
            ParameterSpec("x", 1.0, 0.0, 2.0),
            ParameterSpec("y", 2.0, 1.0, 3.0),
        )
    )
    jacobian = finite_difference_jacobian(
        lambda values: np.asarray(
            [2.0 * values[0] + 3.0 * values[1], values[0] - values[1]]
        ),
        space,
        output_names=("cmax_mg_l", "auc_last_mg_h_l"),
    )
    np.testing.assert_allclose(jacobian.jacobian, [[2.0, 3.0], [1.0, -1.0]], rtol=1e-10)
    diagnostic = identifiability_diagnostics(jacobian)
    assert diagnostic.effective_rank == 2
    assert diagnostic.locally_identifiable
    assert np.isfinite(diagnostic.condition_number)

    unidentifiable = finite_difference_jacobian(
        lambda values: np.asarray([values[0] + values[1]]),
        space,
        output_names=("auc_last_mg_h_l",),
    )
    unidentifiable_diagnostic = identifiability_diagnostics(unidentifiable)
    assert unidentifiable_diagnostic.effective_rank == 1
    assert not unidentifiable_diagnostic.locally_identifiable
    assert np.isinf(unidentifiable_diagnostic.condition_number)
    assert unidentifiable_diagnostic.null_space_vectors.shape == (1, 2)


def test_finite_difference_fails_closed_on_nonfinite_or_changing_outputs() -> None:
    space = ParameterSpace((ParameterSpec("x", 1.0, 0.0, 2.0),))
    with pytest.raises(FloatingPointError, match="non-finite"):
        finite_difference_jacobian(lambda _values: np.asarray([np.nan]), space)

    calls = 0

    def changing(values: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        return np.ones(1 if calls == 1 else 2)

    with pytest.raises(ValueError, match="size changed"):
        finite_difference_jacobian(changing, space)

    at_lower_bound = finite_difference_jacobian(
        lambda values: np.asarray([values[0] ** 2]),
        space,
        point=np.asarray([0.0]),
    )
    assert at_lower_bound.schemes == ("forward",)
    assert at_lower_bound.jacobian[0, 0] > 0.0


def test_profile_likelihood_optimizes_nuisance_and_reports_grid_boundary() -> None:
    space = ParameterSpace(
        (
            ParameterSpec("absorption", 1.0, 0.0, 3.0),
            ParameterSpec("clearance", 1.0, 0.0, 4.0),
        )
    )

    def negative_log_likelihood(values: np.ndarray) -> float:
        return float(((values[0] - 1.5) / 0.4) ** 2 / 2.0 + (values[1] - 2.0) ** 2)

    result = profile_likelihood(
        negative_log_likelihood,
        space,
        "absorption",
        grid=np.linspace(0.0, 3.0, 31),
        confidence_level=0.95,
    )
    assert np.isclose(result.grid_mle, 1.5)
    np.testing.assert_allclose(result.nuisance_optima[:, 1], 2.0, atol=1e-6)
    assert result.interval_bounded_by_grid
    assert result.accepted_region_connected
    assert result.confidence_interval[0] < 1.5 < result.confidence_interval[1]

    with pytest.raises(FloatingPointError, match="non-finite"):
        profile_likelihood(lambda _values: float("nan"), space, "absorption")


def test_empirical_intervals_and_matched_observed_coverage() -> None:
    predictions = np.column_stack(
        (
            np.arange(10.0),
            np.arange(100.0, 110.0),
            np.linspace(0.5, 1.4, 10),
        )
    )
    intervals = empirical_intervals(
        predictions,
        levels=(0.8, 0.95),
        output_names=("cmax_mg_l", "auc_last_mg_h_l", "tmax_h"),
    )
    coverage = observed_coverage(
        np.asarray([5.0, 200.0, 1.0]), intervals, level=0.8
    )
    assert coverage.covered.tolist() == [True, False, True]
    assert coverage.covered_count == 2
    assert coverage.total_count == 3
    assert np.isclose(coverage.coverage_fraction, 2.0 / 3.0)
    assert len(intervals.records()) == 6

    with pytest.raises(ValueError, match="finite"):
        empirical_intervals(np.asarray([[1.0], [np.nan]]))
    with pytest.raises(ValueError, match="finite"):
        observed_coverage(np.asarray([5.0, np.nan, 1.0]), intervals, level=0.8)
    with pytest.raises(KeyError, match="was not computed"):
        observed_coverage(np.asarray([5.0, 105.0, 1.0]), intervals, level=0.9)
