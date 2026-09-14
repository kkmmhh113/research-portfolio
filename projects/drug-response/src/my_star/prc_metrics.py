from __future__ import annotations

import numpy as np


def masked_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
    top_k: int = 50,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    if y_true.shape != y_pred.shape or y_true.shape != mask.shape:
        raise ValueError("Targets, predictions, and masks must have matching shapes")
    pearsons = []
    recalls = []
    for true_row, pred_row, mask_row in zip(y_true, y_pred, mask):
        observed = np.flatnonzero(mask_row)
        if len(observed) < 2:
            continue
        true_values = true_row[observed]
        pred_values = pred_row[observed]
        true_centered = true_values - true_values.mean()
        pred_centered = pred_values - pred_values.mean()
        denominator = np.sqrt(
            np.square(true_centered).sum() * np.square(pred_centered).sum()
        )
        pearsons.append(
            float((true_centered * pred_centered).sum() / denominator)
            if denominator > 1e-12
            else 0.0
        )
        k = min(top_k, len(observed))
        true_top = observed[np.argpartition(np.abs(true_values), -k)[-k:]]
        pred_top = observed[np.argpartition(np.abs(pred_values), -k)[-k:]]
        recalls.append(len(set(true_top).intersection(pred_top)) / k)
    errors = (y_true - y_pred)[mask]
    if not pearsons or len(errors) == 0:
        raise ValueError("No evaluable targets")
    return {
        "mean_sample_pearson": float(np.mean(pearsons)),
        "median_sample_pearson": float(np.median(pearsons)),
        "rmse": float(np.sqrt(np.square(errors).mean())),
        "mae": float(np.abs(errors).mean()),
        f"top_{top_k}_deg_recall": float(np.mean(recalls)),
        "observed_target_fraction": float(mask.mean()),
    }


def calibrate_masked_residual_scale(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    context_prior: np.ndarray,
    mask: np.ndarray,
    top_k: int,
    metric: str,
    maximum: float,
    steps: int,
) -> tuple[float, dict[str, float]]:
    minimize = metric in {"rmse", "mae"}
    best_scale = 0.0
    best_metrics: dict[str, float] | None = None
    for scale in np.linspace(0.0, maximum, steps):
        calibrated = context_prior + float(scale) * (y_pred - context_prior)
        metrics = masked_regression_metrics(y_true, calibrated, mask, top_k)
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
    best_metrics["residual_scale"] = best_scale
    return best_scale, best_metrics
