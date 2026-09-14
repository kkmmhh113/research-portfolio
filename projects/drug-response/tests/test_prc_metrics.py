from __future__ import annotations

import numpy as np

from my_star.prc_metrics import masked_regression_metrics


def test_masked_metrics_ignore_unobserved_error() -> None:
    true = np.array([[1.0, 1000.0, -1.0]], dtype=np.float32)
    pred = np.array([[1.0, -1000.0, -1.0]], dtype=np.float32)
    mask = np.array([[True, False, True]])
    metrics = masked_regression_metrics(true, pred, mask, top_k=2)
    assert metrics["mean_sample_pearson"] == 1.0
    assert metrics["rmse"] == 0.0
    assert metrics["top_2_deg_recall"] == 1.0
