from __future__ import annotations

import numpy as np


def samplewise_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    true_centered = y_true - y_true.mean(axis=1, keepdims=True)
    pred_centered = y_pred - y_pred.mean(axis=1, keepdims=True)
    numerator = np.sum(true_centered * pred_centered, axis=1)
    denominator = np.sqrt(
        np.sum(true_centered**2, axis=1) * np.sum(pred_centered**2, axis=1)
    )
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-12)


def top_k_recall(y_true: np.ndarray, y_pred: np.ndarray, k: int = 50) -> np.ndarray:
    k = min(k, y_true.shape[1])
    true_top = np.argpartition(np.abs(y_true), -k, axis=1)[:, -k:]
    pred_top = np.argpartition(np.abs(y_pred), -k, axis=1)[:, -k:]
    return np.array(
        [len(set(t.tolist()).intersection(p.tolist())) / k for t, p in zip(true_top, pred_top)],
        dtype=np.float64,
    )


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, top_k: int = 50) -> dict[str, float]:
    pearson = samplewise_pearson(y_true, y_pred)
    recall = top_k_recall(y_true, y_pred, top_k)
    return {
        "mean_sample_pearson": float(np.mean(pearson)),
        "median_sample_pearson": float(np.median(pearson)),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        f"top_{top_k}_deg_recall": float(np.mean(recall)),
    }


def calibrate_residual_scale(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    context_prior: np.ndarray,
    top_k: int = 50,
    metric: str = "mean_sample_pearson",
    minimum: float = 0.0,
    maximum: float = 1.5,
    steps: int = 31,
) -> tuple[float, dict[str, float]]:
    """Select one validation-only shrinkage factor for the learned residual."""
    if y_true.shape != y_pred.shape or y_true.shape != context_prior.shape:
        raise ValueError("Target, prediction, and context prior shapes must match")
    if steps < 2 or minimum > maximum:
        raise ValueError("Invalid residual-scale search range")

    minimize = metric in {"rmse", "mae"}
    best_scale = minimum
    best_metrics: dict[str, float] | None = None
    for scale in np.linspace(minimum, maximum, steps):
        calibrated = context_prior + float(scale) * (y_pred - context_prior)
        metrics = regression_metrics(y_true, calibrated, top_k=top_k)
        if metric not in metrics:
            raise ValueError(f"Unknown calibration metric: {metric}")
        if best_metrics is None:
            improved = True
        elif minimize:
            improved = metrics[metric] < best_metrics[metric]
        else:
            improved = metrics[metric] > best_metrics[metric]
        if improved:
            best_scale = float(scale)
            best_metrics = metrics
    assert best_metrics is not None
    return best_scale, best_metrics
