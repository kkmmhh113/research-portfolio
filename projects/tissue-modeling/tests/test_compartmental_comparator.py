from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from our_star.analysis.comparator_fit import (
    MultistartSettings,
    OralConcentrationCurve,
    equal_curve_log10_residuals,
    fit_oral_one_compartment,
)
from our_star.comparators import (
    OralOneCompartmentBounds,
    OralOneCompartmentParameters,
    oral_one_compartment_concentration,
)


def _bounds() -> OralOneCompartmentBounds:
    # Non-overlapping rate bounds remove the oral ka/ke flip-flop ambiguity for
    # this synthetic recovery test.  Production bounds must be protocol-frozen.
    return OralOneCompartmentBounds(
        ka_h=(0.5, 5.0),
        ke_h=(0.05, 0.4),
        scale_l_inv=(0.01, 0.2),
    )


def test_oral_analytic_solution_matches_ode_and_repeated_rate_limit() -> None:
    times = np.linspace(0.0, 24.0, 97)
    parameters = OralOneCompartmentParameters(
        ka_h=1.4, ke_h=0.18, scale_l_inv=0.04
    )
    analytic = oral_one_compartment_concentration(
        times, dose_mg=100.0, parameters=parameters
    )
    numerical = solve_ivp(
        lambda _time, state: np.asarray(
            [
                -parameters.ka_h * state[0],
                parameters.ka_h * state[0] - parameters.ke_h * state[1],
            ]
        ),
        (0.0, float(times[-1])),
        np.asarray([100.0, 0.0]),
        t_eval=times,
        rtol=1.0e-11,
        atol=1.0e-13,
    ).y[1] * parameters.scale_l_inv
    np.testing.assert_allclose(analytic, numerical, rtol=2.0e-9, atol=1.0e-11)
    assert analytic[0] == 0.0
    assert np.all(analytic >= 0.0)

    equal_rates = OralOneCompartmentParameters(
        ka_h=0.6, ke_h=0.6, scale_l_inv=0.03
    )
    repeated = oral_one_compartment_concentration(
        times, dose_mg=80.0, parameters=equal_rates
    )
    expected = 0.03 * 80.0 * 0.6 * times * np.exp(-0.6 * times)
    np.testing.assert_allclose(repeated, expected, rtol=2.0e-15, atol=0.0)

    reversed_rates = OralOneCompartmentParameters(
        ka_h=0.05, ke_h=20.0, scale_l_inv=0.03
    )
    long_times = np.asarray([0.0, 1.0, 100.0, 10_000.0])
    reversed_solution = oral_one_compartment_concentration(
        long_times, dose_mg=80.0, parameters=reversed_rates
    )
    assert np.all(np.isfinite(reversed_solution))
    assert np.all(reversed_solution >= 0.0)


def test_model_and_bounds_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="ka_h"):
        OralOneCompartmentParameters(ka_h=0.0, ke_h=0.2, scale_l_inv=0.1)
    with pytest.raises(ValueError, match="lower < upper"):
        OralOneCompartmentBounds(
            ka_h=(2.0, 1.0), ke_h=(0.1, 0.4), scale_l_inv=(0.01, 0.2)
        )
    parameters = OralOneCompartmentParameters(1.0, 0.2, 0.05)
    with pytest.raises(ValueError, match="nonnegative"):
        oral_one_compartment_concentration(
            np.asarray([-1.0, 1.0]), dose_mg=10.0, parameters=parameters
        )
    with pytest.raises(ValueError, match="dose_mg"):
        oral_one_compartment_concentration(
            np.asarray([0.0, 1.0]), dose_mg=0.0, parameters=parameters
        )


def test_curve_validation_and_explicit_predose_exclusion() -> None:
    curve = OralConcentrationCurve(
        curve_id="synthetic",
        times_h=np.asarray([0.0, 0.5, 1.0]),
        observed_mg_l=np.asarray([0.0, 1.0, 0.8]),
        dose_mg=50.0,
    )
    assert curve.excluded_predose_count == 1
    assert curve.fitted_observation_count == 2
    with pytest.raises(ValueError, match="time-zero predose"):
        OralConcentrationCurve(
            curve_id="invalid_zero",
            times_h=np.asarray([0.0, 1.0, 2.0]),
            observed_mg_l=np.asarray([0.0, 0.0, 1.0]),
            dose_mg=50.0,
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        OralConcentrationCurve(
            curve_id="duplicate_time",
            times_h=np.asarray([0.0, 1.0, 1.0]),
            observed_mg_l=np.asarray([0.0, 1.0, 0.8]),
            dose_mg=50.0,
        )


def test_equal_curve_weighting_is_independent_of_point_count() -> None:
    parameters = OralOneCompartmentParameters(1.0, 0.2, 0.05)
    times_sparse = np.asarray([0.5, 1.0, 2.0])
    times_dense = np.linspace(0.25, 8.0, 31)
    sparse_prediction = oral_one_compartment_concentration(
        times_sparse, dose_mg=20.0, parameters=parameters
    )
    dense_prediction = oral_one_compartment_concentration(
        times_dense, dose_mg=20.0, parameters=parameters
    )
    curves = (
        OralConcentrationCurve(
            "sparse",
            times_sparse,
            sparse_prediction / 2.0,
            20.0,
        ),
        OralConcentrationCurve(
            "dense",
            times_dense,
            dense_prediction / 4.0,
            20.0,
        ),
    )
    residuals = equal_curve_log10_residuals(curves, parameters)
    expected = (np.log10(2.0) ** 2 + np.log10(4.0) ** 2) / 2.0
    assert np.isclose(np.dot(residuals, residuals), expected)


def test_multistart_fit_is_deterministic_and_audits_all_starts() -> None:
    truth = OralOneCompartmentParameters(
        ka_h=1.3, ke_h=0.16, scale_l_inv=0.055
    )
    times = np.concatenate(([0.0], np.geomspace(0.15, 24.0, 25)))
    observations = oral_one_compartment_concentration(
        times, dose_mg=120.0, parameters=truth
    )
    curve = OralConcentrationCurve(
        "synthetic_recovery", times, observations, 120.0
    )
    settings = MultistartSettings(start_count=20, seed=77)
    first = fit_oral_one_compartment(
        (curve,), bounds=_bounds(), settings=settings
    )
    second = fit_oral_one_compartment(
        (curve,), bounds=_bounds(), settings=settings
    )

    np.testing.assert_allclose(
        first.parameters.as_array(), truth.as_array(), rtol=2.0e-6, atol=1.0e-9
    )
    np.testing.assert_allclose(
        first.parameters.as_array(), second.parameters.as_array(), rtol=0.0, atol=0.0
    )
    assert first.objective < 1.0e-20
    assert first.converged_start_count == 20
    assert len(first.start_diagnostics) == 20
    assert all(item.objective is not None for item in first.start_diagnostics)
    assert first.curve_excluded_predose_count == (("synthetic_recovery", 1),)
    report = first.to_dict()
    assert report["optimizer"]["parameterization"] == "natural_log_positive_parameters"
    assert len(report["start_diagnostics"]) == 20
    assert "F/V" in report["interpretation_boundary"]


def test_fit_rejects_underdeclared_multistart_and_duplicate_curve_ids() -> None:
    with pytest.raises(ValueError, match="at least 20"):
        MultistartSettings(start_count=19)
    times = np.asarray([0.5, 1.0, 2.0])
    observations = np.asarray([0.5, 0.7, 0.4])
    first = OralConcentrationCurve("duplicate", times, observations, 20.0)
    second = OralConcentrationCurve("duplicate", times, observations, 20.0)
    with pytest.raises(ValueError, match="unique"):
        fit_oral_one_compartment((first, second), bounds=_bounds())


def test_overlapping_rate_bounds_expose_flip_flop_equivalent_solutions() -> None:
    truth = OralOneCompartmentParameters(ka_h=1.3, ke_h=0.16, scale_l_inv=0.055)
    times = np.geomspace(0.15, 24.0, 25)
    curve = OralConcentrationCurve(
        "flip_flop_synthetic",
        times,
        oral_one_compartment_concentration(times, dose_mg=120.0, parameters=truth),
        120.0,
    )
    result = fit_oral_one_compartment(
        (curve,),
        bounds=OralOneCompartmentBounds(
            ka_h=(0.01, 10.0),
            ke_h=(0.01, 10.0),
            scale_l_inv=(0.001, 1.0),
        ),
        settings=MultistartSettings(start_count=20, seed=77),
    )
    assert result.flip_flop_ambiguity_detected
    spans = dict(result.near_best_parameter_log_span)
    assert spans["ka_h"] > 1.0
    assert spans["ke_h"] > 1.0
    assert result.to_dict()["flip_flop_ambiguity_detected"] is True
