import numpy as np

from my_star.metrics import (
    calibrate_residual_scale,
    regression_metrics,
    samplewise_pearson,
    top_k_recall,
)


def test_perfect_metrics() -> None:
    values = np.array([[1.0, -2.0, 0.5], [-1.0, 4.0, 2.0]])
    assert np.allclose(samplewise_pearson(values, values), 1.0)
    assert np.allclose(top_k_recall(values, values, k=2), 1.0)
    assert regression_metrics(values, values, top_k=2)["rmse"] == 0.0


def test_residual_calibration_recovers_shrinkage() -> None:
    prior = np.array([[1.0, -1.0], [2.0, -2.0]], dtype=np.float32)
    residual = np.array([[0.5, -0.5], [-0.25, 0.25]], dtype=np.float32)
    target = prior + residual
    prediction = prior + 2.0 * residual
    scale, metrics = calibrate_residual_scale(
        target,
        prediction,
        prior,
        metric="rmse",
        minimum=0.0,
        maximum=1.0,
        steps=21,
    )
    assert np.isclose(scale, 0.5)
    assert metrics["rmse"] < 1e-6
