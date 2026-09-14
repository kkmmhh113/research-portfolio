"""Strict shared-grid point metrics, distinct from historical validation masks.

No uncertainty interval, clinical gate, source canonicalization or model fit
is implemented here. A zero prediction at a positive observation stays in the
log score through the explicit concentration floor.
"""

from __future__ import annotations

import numpy as np


def sampled_point_metrics(
    times_h: np.ndarray,
    observed_mg_l: np.ndarray,
    predicted_mg_l: np.ndarray,
    *,
    concentration_floor_mg_l: float = 1e-8,
) -> dict[str, float | int]:
    """Score one curve on its actual sampled window, with no interpolation."""
    times = np.asarray(times_h, dtype=float)
    observed = np.asarray(observed_mg_l, dtype=float)
    predicted = np.asarray(predicted_mg_l, dtype=float)
    floor = float(concentration_floor_mg_l)
    if times.ndim != 1 or times.size < 2:
        raise ValueError("at least two time points in a vector are required")
    if observed.shape != times.shape or predicted.shape != times.shape:
        raise ValueError("time, observation and prediction shapes must match")
    if not all(np.all(np.isfinite(x)) for x in (times, observed, predicted)):
        raise ValueError("all time, observation and prediction values must be finite")
    if np.any(times < 0) or np.any(np.diff(times) <= 0):
        raise ValueError("times must be nonnegative, increasing and unique")
    if np.any(observed < 0) or np.any(predicted < 0):
        raise ValueError("observations and predictions must be nonnegative")
    if np.any((observed == 0) & (times != 0)):
        raise ValueError("postdose zero observation needs an explicit BLQ analysis")
    if not np.isfinite(floor) or floor <= 0:
        raise ValueError("concentration floor must be finite and positive")
    log_mask = ~((times == 0) & (observed == 0))
    if not np.any(log_mask):
        raise ValueError("no log-evaluable observations")
    log_ratio = np.log10(
        np.maximum(predicted[log_mask], floor)
        / np.maximum(observed[log_mask], floor)
    )
    observed_auc = float(np.trapezoid(observed, times))
    predicted_auc = float(np.trapezoid(predicted, times))
    if observed_auc <= 0:
        raise ValueError("observed sampled-window AUC must be positive")
    obs_peak = int(np.argmax(observed))
    pred_peak = int(np.argmax(predicted))
    abs_log = np.abs(log_ratio)
    return {
        "observation_count": int(times.size),
        "log_evaluable_count": int(log_mask.sum()),
        "excluded_exact_zero_predose_count": int((~log_mask).sum()),
        "floored_prediction_log_row_count": int(
            np.sum(predicted[log_mask] < floor)
        ),
        "window_start_h": float(times[0]),
        "window_end_h": float(times[-1]),
        "log10_rmse": float(np.sqrt(np.mean(log_ratio**2))),
        "log10_mae": float(np.mean(abs_log)),
        "afe": float(10 ** np.mean(log_ratio)),
        "aafe": float(10 ** np.mean(abs_log)),
        "fraction_within_1_5_fold": float(np.mean(abs_log <= np.log10(1.5))),
        "fraction_within_2_fold": float(np.mean(abs_log <= np.log10(2.0))),
        "observed_auc_sampled_window_mg_h_l": observed_auc,
        "predicted_auc_sampled_window_mg_h_l": predicted_auc,
        "auc_last_ratio": predicted_auc / observed_auc,
        "observed_sampled_cmax_mg_l": float(observed[obs_peak]),
        "predicted_sampled_cmax_mg_l": float(predicted[pred_peak]),
        "sampled_cmax_ratio": float(predicted[pred_peak] / observed[obs_peak]),
        "observed_sampled_tmax_h": float(times[obs_peak]),
        "predicted_sampled_tmax_h": float(times[pred_peak]),
        "sampled_absolute_tmax_error_h": float(abs(times[pred_peak] - times[obs_peak])),
    }
