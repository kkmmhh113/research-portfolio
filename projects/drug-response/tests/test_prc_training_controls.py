from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from scripts.train_prc import build_training_sampler, optimizer_parameter_groups


class _Dataset:
    def __init__(self) -> None:
        self.frame = pd.DataFrame(
            {"compound_id": ["a", "a", "a", "b", "c", "c"]}
        )
        self.domain_labels = [0, 0, 1, 1, 1, 1]

    def __len__(self) -> int:
        return len(self.frame)


def test_compound_balanced_sampler_inverts_row_frequency() -> None:
    sampler = build_training_sampler(
        _Dataset(),
        balance_domains=False,
        balance_compounds=True,
        samples_per_epoch=None,
        seed=7,
    )

    assert sampler is not None
    weights = sampler.weights.numpy()
    np.testing.assert_allclose(weights[:3], weights[0])
    np.testing.assert_allclose(weights[4:], weights[4])
    assert weights[3] == pytest.approx(weights[0] * 3.0)
    assert weights[4] == pytest.approx(weights[0] * 1.5)


def test_sampler_is_absent_when_no_balancing_or_resampling_is_requested() -> None:
    sampler = build_training_sampler(
        _Dataset(),
        balance_domains=False,
        balance_compounds=False,
        samples_per_epoch=None,
        seed=7,
    )

    assert sampler is None


class _ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.chemical_encoder = nn.Linear(3, 2)
        self.decoder = nn.Linear(2, 1)


def test_optimizer_parameter_groups_apply_prefix_multiplier() -> None:
    model = _ToyModel()
    groups = optimizer_parameter_groups(
        model,
        base_learning_rate=4e-4,
        prefix_multipliers={"chemical_encoder": 0.1},
    )

    by_name = {group["group_name"]: group for group in groups}
    assert by_name["default"]["lr"] == pytest.approx(4e-4)
    assert by_name["chemical_encoder"]["lr"] == pytest.approx(4e-5)
    assert sum(len(group["params"]) for group in groups) == len(
        list(model.parameters())
    )


def test_optimizer_parameter_groups_reject_unmatched_prefix() -> None:
    with pytest.raises(ValueError, match="matched no trainable parameters"):
        optimizer_parameter_groups(
            _ToyModel(),
            base_learning_rate=4e-4,
            prefix_multipliers={"missing": 0.1},
        )
