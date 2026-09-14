from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from our_star.population import (
    FlowCompositionSpec,
    ModifierSpec,
    PopulationSpecification,
    RandomEffectSpec,
    apply_latent_transform,
    draw_latent_effects,
    sample_virtual_subjects,
)
from our_star.prediction import (
    CalibrationDataset,
    CalibrationRequest,
    PredictionTarget,
    apply_prediction_interval_calibration,
    calibrate_prediction_intervals,
    generate_predictive_replicates,
)


def _population_specification() -> PopulationSpecification:
    names = (
        "cardiac_output_effect",
        "liver_volume_effect",
        "enzyme_effect",
        "portal_log_ratio",
        "hepatic_artery_log_ratio",
        "renal_log_ratio",
    )
    return PopulationSpecification(
        population_id="synthetic_adults_v04",
        random_effects=RandomEffectSpec(
            names=names,
            covariance=np.diag([0.04, 0.02, 0.09, 0.03, 0.03, 0.03]),
            covariance_names=names,
        ),
        modifiers=(
            ModifierSpec(
                "cardiac_output_effect",
                "cardiac_output_l_h",
                "anatomy",
                "log",
                300.0,
                "L/h",
            ),
            ModifierSpec(
                "liver_volume_effect",
                "liver_volume_l",
                "anatomy",
                "positive",
                1.8,
                "L",
            ),
            ModifierSpec(
                "enzyme_effect",
                "enzyme_activity_factor",
                "mechanism",
                "log",
                1.0,
                "1",
            ),
        ),
        flow_composition=FlowCompositionSpec(
            total_target="cardiac_output_l_h",
            component_targets=(
                "portal_flow_l_h",
                "hepatic_artery_flow_l_h",
                "renal_flow_l_h",
                "rest_flow_l_h",
            ),
            reference_fractions=(0.24, 0.06, 0.24, 0.46),
            log_ratio_effect_names=(
                "portal_log_ratio",
                "hepatic_artery_log_ratio",
                "renal_log_ratio",
            ),
        ),
    )


def test_random_effect_spec_rejects_order_symmetry_pd_and_condition_failures() -> None:
    with pytest.raises(ValueError, match="name order"):
        RandomEffectSpec(
            names=("a", "b"),
            covariance=np.eye(2),
            covariance_names=("b", "a"),
        )
    with pytest.raises(ValueError, match="symmetric"):
        RandomEffectSpec(
            names=("a", "b"),
            covariance=np.asarray([[1.0, 0.2], [0.3, 1.0]]),
        )
    with pytest.raises(ValueError, match="positive definite"):
        RandomEffectSpec(
            names=("a", "b"),
            covariance=np.asarray([[1.0, 1.0], [1.0, 1.0]]),
        )
    with pytest.raises(ValueError, match="ill-conditioned"):
        RandomEffectSpec(
            names=("a", "b"),
            covariance=np.diag([1.0, 1.0e-10]),
            max_condition_number=1.0e6,
        )

    with pytest.raises(ValueError, match="row order"):
        RandomEffectSpec.from_named_covariance(
            names=("a", "b"),
            covariance={
                "b": {"a": 0.0, "b": 1.0},
                "a": {"a": 1.0, "b": 0.0},
            },
        )


def test_latent_draws_are_reproducible_named_and_immutable() -> None:
    spec = RandomEffectSpec(
        names=("anatomy", "mechanism"),
        covariance=np.asarray([[0.2, 0.05], [0.05, 0.3]]),
    )
    first = draw_latent_effects(spec, 8, seed=991)
    second = draw_latent_effects(spec, 8, seed=991)
    np.testing.assert_array_equal(first.values, second.values)
    np.testing.assert_array_equal(
        first.standard_normal_draws, second.standard_normal_draws
    )
    assert first.names == ("anatomy", "mechanism")
    assert first.records()[0]["draw_index"] == 0
    with pytest.raises(ValueError, match="read-only"):
        first.values[0, 0] = 123.0


def test_constraint_transforms_preserve_reference_and_domains() -> None:
    assert apply_latent_transform(2.5, 0.0, "positive") == pytest.approx(2.5)
    assert apply_latent_transform(2.5, 0.0, "log") == pytest.approx(2.5)
    assert apply_latent_transform(0.3, 0.0, "logit") == pytest.approx(0.3)
    assert apply_latent_transform(2.5, -100.0, "positive") > 0.0
    assert apply_latent_transform(2.5, -1000.0, "log") > 0.0
    assert 0.0 < apply_latent_transform(0.3, 3.0, "logit") < 1.0
    assert 0.0 < apply_latent_transform(0.3, 1000.0, "logit") < 1.0
    with pytest.raises(ValueError, match="reference"):
        apply_latent_transform(0.0, 0.0, "log")
    with pytest.raises(ValueError, match="reference"):
        apply_latent_transform(1.0, 0.0, "logit")


def test_virtual_subject_sampling_separates_domains_and_balances_flows() -> None:
    specification = _population_specification()
    first = sample_virtual_subjects(specification, 12, seed=301)
    second = sample_virtual_subjects(specification, 12, seed=301)
    np.testing.assert_array_equal(first.latent_draws.values, second.latent_draws.values)
    assert first.records() == second.records()
    assert first.trace.record() == second.trace.record()

    for subject in first.subjects:
        assert "liver_volume_l" in subject.anatomy_modifiers
        assert "enzyme_activity_factor" not in subject.anatomy_modifiers
        assert "enzyme_activity_factor" in subject.mechanism_modifiers
        total = subject.anatomy_modifiers["cardiac_output_l_h"]
        components = sum(
            subject.anatomy_modifiers[name]
            for name in (
                "portal_flow_l_h",
                "hepatic_artery_flow_l_h",
                "renal_flow_l_h",
                "rest_flow_l_h",
            )
        )
        assert components == pytest.approx(total, rel=0.0, abs=2.0 * np.spacing(total))
        assert all(
            subject.anatomy_modifiers[name] > 0.0
            for name in (
                "portal_flow_l_h",
                "hepatic_artery_flow_l_h",
                "renal_flow_l_h",
                "rest_flow_l_h",
            )
        )
        with pytest.raises(TypeError):
            subject.anatomy_modifiers["liver_volume_l"] = 100.0


def test_population_spec_rejects_independently_sampled_flow_components() -> None:
    base = _population_specification()
    extra = ModifierSpec(
        "portal_log_ratio",
        "portal_flow_l_h",
        "anatomy",
        "log",
        72.0,
        "L/h",
    )
    with pytest.raises(ValueError, match="flow components"):
        PopulationSpecification(
            population_id="invalid",
            random_effects=base.random_effects,
            modifiers=base.modifiers + (extra,),
            flow_composition=base.flow_composition,
        )


def test_prediction_targets_and_replicates_are_deterministic() -> None:
    sample = sample_virtual_subjects(_population_specification(), 7, seed=44)
    selected_id = sample.subjects[2].subject_id
    targets = (
        PredictionTarget.individual(
            "selected_cmax", "cmax_mg_l", "mg/L", selected_id
        ),
        PredictionTarget.cohort_mean(
            "mean_cmax", "cmax_mg_l", "mg/L", "synthetic_cohort"
        ),
        PredictionTarget.cohort_median(
            "median_auc", "auc_mg_h_l", "mg*h/L", "synthetic_cohort"
        ),
    )

    def simulator(subject, replicate_index, rng):
        liver = subject.anatomy_modifiers["liver_volume_l"]
        enzyme = subject.mechanism_modifiers["enzyme_activity_factor"]
        residual = rng.normal(0.0, 0.01)
        return {
            "cmax_mg_l": liver / enzyme + residual,
            "auc_mg_h_l": 10.0 * liver / enzyme + 0.1 * replicate_index,
        }

    first = generate_predictive_replicates(
        sample.subjects, targets, simulator, 9, seed=891
    )
    second = generate_predictive_replicates(
        sample.subjects, targets, simulator, 9, seed=891
    )
    np.testing.assert_array_equal(first.values, second.values)
    np.testing.assert_array_equal(
        first.trace.stream_seeds, second.trace.stream_seeds
    )
    changed = generate_predictive_replicates(
        sample.subjects, targets, simulator, 9, seed=892
    )
    assert not np.array_equal(first.values, changed.values)
    assert first.values.shape == (9, 3)
    assert first.target_ids == ("selected_cmax", "mean_cmax", "median_auc")


def _synthetic_calibration_data(
    *,
    unit_count: int,
    target_ids: tuple[str, ...] = ("mean_cmax",),
) -> CalibrationDataset:
    rng = np.random.default_rng(410)
    target_count = len(target_ids)
    centers = rng.normal(2.0, 0.2, size=(unit_count, target_count))
    draws = centers[np.newaxis, :, :] + rng.normal(
        0.0, 0.12, size=(80, unit_count, target_count)
    )
    observed = centers + np.linspace(-0.4, 0.4, unit_count)[:, np.newaxis]
    if target_count > 1:
        observed[:, 1] += rng.normal(0.0, 0.25, unit_count)
    return CalibrationDataset(
        unit_ids=tuple(f"development_unit_{index:03d}" for index in range(unit_count)),
        target_ids=target_ids,
        observed=observed,
        predictive_draws=draws,
    )


def test_external_calibration_is_rejected_before_callback() -> None:
    called = False

    def forbidden_callback():
        nonlocal called
        called = True
        return _synthetic_calibration_data(unit_count=20)

    request = CalibrationRequest(
        calibration_id="must_not_fit",
        study_id="synthetic_external",
        study_role="external",
        targets=(
            PredictionTarget.cohort_mean(
                "mean_cmax", "cmax_mg_l", "mg/L", "cohort"
            ),
        ),
    )
    with pytest.raises(PermissionError, match="development-only"):
        calibrate_prediction_intervals(request, forbidden_callback)
    assert not called


def test_calibration_fails_for_insufficient_units_and_group_covariance() -> None:
    group_target = PredictionTarget.cohort_mean(
        "mean_cmax", "cmax_mg_l", "mg/L", "cohort"
    )
    insufficient = CalibrationRequest(
        calibration_id="insufficient",
        study_id="synthetic_development",
        study_role="development",
        targets=(group_target,),
        interval_level=0.9,
        minimum_units=2,
    )
    with pytest.raises(ValueError, match="insufficient independent"):
        calibrate_prediction_intervals(
            insufficient, lambda: _synthetic_calibration_data(unit_count=5)
        )

    called = False

    def group_callback():
        nonlocal called
        called = True
        return _synthetic_calibration_data(unit_count=20)

    invalid_covariance = CalibrationRequest(
        calibration_id="invalid_group_covariance",
        study_id="synthetic_development",
        study_role="development",
        targets=(group_target,),
        fit_full_covariance=True,
    )
    with pytest.raises(ValueError, match="group-summary"):
        calibrate_prediction_intervals(invalid_covariance, group_callback)
    assert not called


def test_development_calibration_is_frozen_reproducible_and_applicable() -> None:
    target = PredictionTarget.cohort_mean(
        "mean_cmax", "cmax_mg_l", "mg/L", "cohort"
    )
    request = CalibrationRequest(
        calibration_id="synthetic_pi_v04",
        study_id="synthetic_development_only",
        study_role="development",
        targets=(target,),
        interval_level=0.9,
        minimum_units=10,
        source_hashes={"synthetic_generator": "fixed-seed-410"},
    )
    data = _synthetic_calibration_data(unit_count=20)
    first = calibrate_prediction_intervals(request, lambda: data)
    second = calibrate_prediction_intervals(request, lambda: data)
    assert first.to_record() == second.to_record()
    assert first.artifact_hash == second.artifact_hash
    assert first.frozen
    assert first.development_study_id == "synthetic_development_only"
    assert first.additive_corrections[0] >= 0.0
    with pytest.raises(ValueError, match="read-only"):
        first.additive_corrections[0] = 0.0
    with pytest.raises(FrozenInstanceError):
        first.interval_level = 0.5

    future_draws = np.linspace(1.0, 3.0, 60)[:, np.newaxis]
    intervals = apply_prediction_interval_calibration(
        first,
        future_draws,
        target_ids=("mean_cmax",),
    )
    assert intervals.lower.shape == (1,)
    assert intervals.lower[0] <= intervals.median[0] <= intervals.upper[0]
    assert intervals.calibration_artifact_hash == first.artifact_hash


def test_individual_units_can_fit_identified_full_residual_covariance() -> None:
    targets = (
        PredictionTarget.individual("cmax", "cmax_mg_l", "mg/L"),
        PredictionTarget.individual("auc", "auc_mg_h_l", "mg*h/L"),
    )
    request = CalibrationRequest(
        calibration_id="synthetic_full_covariance",
        study_id="synthetic_individual_development",
        study_role="development",
        targets=targets,
        interval_level=0.8,
        minimum_units=8,
        fit_full_covariance=True,
    )
    data = _synthetic_calibration_data(
        unit_count=24,
        target_ids=("cmax", "auc"),
    )
    artifact = calibrate_prediction_intervals(request, lambda: data)
    assert artifact.residual_covariance is not None
    assert artifact.residual_covariance.shape == (2, 2)
    np.linalg.cholesky(artifact.residual_covariance)
