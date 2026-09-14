import numpy as np
import pandas as pd

from my_star.data import ContextPrior, TargetScaler


def test_context_prior_uses_only_matching_context() -> None:
    frame = pd.DataFrame(
        {
            "cell_line": ["A", "A", "B"],
            "dose_nM": [10.0, 10.0, 100.0],
            "duration_hours": [24.0, 24.0, 24.0],
            "delta_y": [np.array([1.0, 3.0]), np.array([3.0, 5.0]), np.array([9.0, 7.0])],
        }
    )
    prior = ContextPrior.fit(frame)
    assert np.allclose(prior.lookup("A", 10.0, 24.0), [2.0, 4.0])
    assert np.allclose(prior.lookup("missing", 1.0, 1.0), frame.delta_y.mean())


def test_global_target_scaler_uses_one_scale() -> None:
    targets = np.array([[1.0, 10.0], [3.0, 20.0]], dtype=np.float32)
    scaler = TargetScaler.fit(targets, mode="global")
    assert np.allclose(scaler.scale, scaler.scale[0])
    assert np.allclose(scaler.inverse_transform(scaler.transform(targets)), targets)
