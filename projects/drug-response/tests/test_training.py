import numpy as np
import pandas as pd
import torch

from scripts.train_sciplex import fit_gene_loss_weights, sample_reliability_weights


def test_gene_loss_weights_use_only_training_compounds() -> None:
    frame = pd.DataFrame(
        {
            "compound_id": ["train", "train", "holdout"],
            "delta_y_replicate_std": [
                np.array([0.01, 0.20], dtype=np.float32),
                np.array([0.01, 0.20], dtype=np.float32),
                np.array([10.0, 0.0], dtype=np.float32),
            ],
        }
    )
    weights = fit_gene_loss_weights(
        frame,
        {"train"},
        mode="replicate_reliability",
        noise_floor=0.05,
        maximum=4.0,
    )
    assert weights is not None
    assert weights[0] > weights[1]


def test_gene_loss_weighting_can_be_disabled() -> None:
    frame = pd.DataFrame(
        {
            "compound_id": ["train"],
            "delta_y_replicate_std": [np.ones(2, dtype=np.float32)],
        }
    )
    assert fit_gene_loss_weights(frame, {"train"}, "none", 0.05, 4.0) is None


def test_sample_reliability_weights_downweight_noisy_conditions() -> None:
    correlations = torch.tensor([-0.2, 0.0, 0.5, 1.0])
    weights = sample_reliability_weights(
        correlations,
        mode="replicate_reliability",
        floor=0.25,
        power=1.0,
    )
    assert weights is not None
    assert torch.allclose(weights, torch.tensor([0.25, 0.25, 0.75, 1.25]))
