from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from my_star.prc_data import (
    MaskedContextPrior,
    MaskedTargetScaler,
    PRCDataset,
    build_stage_datasets,
    build_target_auxiliary_supervision,
    collate_prc_batch,
    fit_masked_response_basis,
    fit_shared_response_basis,
    load_readout_features,
)


def test_target_auxiliary_vocabulary_is_fit_on_training_compounds_only() -> None:
    supervision = pd.DataFrame(
        {
            "compound_id": ["train", "validation"],
            "specific_target_gene_symbols": ["GENE1 | GENE2", "GENE3"],
            "eligible_for_target_auxiliary_training": [True, True],
            "allowed_as_inference_input": [False, False],
        }
    )

    vocabulary, labels = build_target_auxiliary_supervision(
        supervision, {"train"}
    )

    assert vocabulary == {"GENE1": 0, "GENE2": 1}
    np.testing.assert_array_equal(labels["train"], [1.0, 1.0])
    assert "validation" not in labels


def test_auxiliary_vocabulary_filters_singletons_and_supports_moa_labels() -> None:
    supervision = pd.DataFrame(
        {
            "compound_id": ["a", "b", "c"],
            "specific_target_gene_symbols": ["G1", "G1 | G2", "G3"],
            "moa_fine": ["Kinase inhibitor", "Kinase inhibitor", "unclear"],
            "eligible_for_target_auxiliary_training": [True, True, True],
            "allowed_as_inference_input": [False, False, False],
        }
    )

    targets, _ = build_target_auxiliary_supervision(
        supervision, {"a", "b", "c"}, minimum_compound_count=2
    )
    moa, labels = build_target_auxiliary_supervision(
        supervision,
        {"a", "b", "c"},
        label_column="moa_fine",
        minimum_compound_count=2,
    )

    assert targets == {"G1": 0}
    assert moa == {"Kinase inhibitor": 0}
    assert set(labels) == {"a", "b"}


def test_build_stage_datasets_allows_final_training_without_validation() -> None:
    frame = pd.DataFrame(
        {
            "sample_id": ["s1"],
            "dataset_domain": ["source"],
            "compound_id": ["c1"],
            "canonical_smiles": ["CCO"],
            "cell_line": ["A"],
            "dose_nM": [10.0],
            "log10_dose_molar": [-8.0],
            "duration_hours": [24.0],
            "delta_y": [np.asarray([1.0, 2.0], dtype=np.float32)],
            "target_mask": [np.ones(2, dtype=bool)],
            "baseline_available": [False],
        }
    )
    datasets, _, _ = build_stage_datasets(
        frame,
        {"splits": {"train": ["c1"], "validation": []}},
        {"A": 0},
        {"source": 0},
        target_scaling="global",
        descriptor_set="basic",
        fingerprint_radius=2,
        fingerprint_use_counts=False,
        fingerprint_include_chirality=False,
        include_validation=False,
    )
    assert set(datasets) == {"train"}
    assert len(datasets["train"]) == 1


def test_prc_dataset_zero_fills_unavailable_scalar_baseline() -> None:
    frame = pd.DataFrame(
        {
            "sample_id": ["s1"],
            "dataset_domain": ["source"],
            "compound_id": ["c1"],
            "canonical_smiles": ["CCO"],
            "cell_line": ["A"],
            "dose_nM": [10.0],
            "log10_dose_molar": [-8.0],
            "duration_hours": [24.0],
            "delta_y": [np.asarray([1.0, 2.0], dtype=np.float32)],
            "target_mask": [np.ones(2, dtype=bool)],
            "baseline_available": [False],
            "baseline_expression": [None],
        }
    )
    scaler = MaskedTargetScaler.fit(
        np.asarray([[1.0, 2.0]], dtype=np.float32),
        np.ones((1, 2), dtype=bool),
        mode="global",
    )
    prior = MaskedContextPrior.fit(frame, output_dim=2)
    dataset = PRCDataset(
        frame,
        {"c1"},
        {"A": 1},
        {"source": 0},
        scaler,
        prior,
        descriptor_set="basic",
        fingerprint_radius=2,
        fingerprint_use_counts=False,
        fingerprint_include_chirality=True,
    )
    np.testing.assert_allclose(dataset.baselines, 0.0)


def test_graph_prc_dataset_collates_different_molecule_sizes() -> None:
    frame = pd.DataFrame(
        {
            "sample_id": ["s1", "s2"],
            "dataset_domain": ["source", "source"],
            "compound_id": ["c1", "c2"],
            "canonical_smiles": ["CCO", "c1ccccc1O"],
            "cell_line": ["A", "A"],
            "dose_nM": [10.0, 10.0],
            "log10_dose_molar": [-8.0, -8.0],
            "duration_hours": [24.0, 24.0],
            "delta_y": [np.asarray([1.0, 2.0], dtype=np.float32)] * 2,
            "target_mask": [np.ones(2, dtype=bool)] * 2,
            "baseline_available": [False, False],
        }
    )
    scaler = MaskedTargetScaler.fit(
        np.asarray([[1.0, 2.0], [1.0, 2.0]], dtype=np.float32),
        np.ones((2, 2), dtype=bool),
    )
    prior = MaskedContextPrior.fit(frame, output_dim=2)
    dataset = PRCDataset(
        frame,
        {"c1", "c2"},
        {"A": 1},
        {"source": 0},
        scaler,
        prior,
        descriptor_set="basic",
        chemical_encoder_mode="graph",
    )
    batch = collate_prc_batch([dataset[0], dataset[1]])
    assert batch["atom_features"].shape == (2, 7, 7)
    assert batch["bond_types"].shape == (2, 7, 7)
    assert batch["atom_mask"].sum(dim=1).tolist() == [3, 7]
    original = dataset[0]["descriptors"].clone()
    dataset.set_chemical_permutation({"c1": "c2", "c2": "c1"})
    assert not torch.equal(dataset[0]["descriptors"], original)
    dataset.set_chemical_permutation(None)


def test_masked_target_scaler_ignores_missing_values_and_all_missing_gene() -> None:
    targets = np.array([[1.0, 100.0, 9.0], [3.0, -100.0, 9.0]], dtype=np.float32)
    masks = np.array([[True, False, False], [True, False, False]])
    scaler = MaskedTargetScaler.fit(targets, masks, mode="global")
    np.testing.assert_allclose(scaler.mean, [2.0, 0.0, 0.0])
    assert np.isfinite(scaler.scale).all()
    assert scaler.scale[0] == scaler.scale[1]


def test_masked_context_prior_uses_only_observed_targets() -> None:
    frame = pd.DataFrame(
        {
            "cell_line": ["A", "A"],
            "dose_nM": [10.0, 10.0],
            "duration_hours": [24.0, 24.0],
            "delta_y": [
                np.array([1.0, 50.0], dtype=np.float32),
                np.array([3.0, 4.0], dtype=np.float32),
            ],
            "target_mask": [np.array([True, False]), np.array([True, True])],
        }
    )
    prior = MaskedContextPrior.fit(frame, output_dim=2)
    np.testing.assert_allclose(prior.lookup("A", 10.0, 24.0), [2.0, 4.0])


def test_masked_context_prior_keeps_dataset_domains_separate() -> None:
    frame = pd.DataFrame(
        {
            "dataset_domain": ["source", "target"],
            "cell_line": ["A", "A"],
            "dose_nM": [10.0, 10.0],
            "duration_hours": [24.0, 24.0],
            "delta_y": [
                np.array([1.0, 2.0], dtype=np.float32),
                np.array([10.0, 20.0], dtype=np.float32),
            ],
            "target_mask": [np.ones(2, dtype=bool)] * 2,
        }
    )
    prior = MaskedContextPrior.fit(frame, output_dim=2)
    np.testing.assert_allclose(prior.lookup("source", "A", 10.0, 24.0), [1.0, 2.0])
    np.testing.assert_allclose(prior.lookup("target", "A", 10.0, 24.0), [10.0, 20.0])


def test_load_readout_features_appends_aligned_external_features(tmp_path) -> None:
    panel_path = tmp_path / "panel.csv"
    pd.DataFrame(
        {
            "common_panel_index": [0, 1],
            "gene_symbol": ["A", "B"],
            "control_mean": [1.0, 2.0],
            "control_variance": [2.0, 3.0],
            "control_detection_fraction": [0.2, 0.8],
        }
    ).to_csv(panel_path, index=False)
    feature_path = tmp_path / "features.npz"
    np.savez_compressed(
        feature_path,
        features=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        gene_symbols=np.asarray(["A", "B"]),
    )
    features, _ = load_readout_features(panel_path, feature_path)
    assert features.shape == (2, 5)
    np.testing.assert_allclose(features[:, -2:], [[1.0, 0.0], [0.0, 1.0]])


def test_masked_response_basis_uses_zero_filled_missing_targets() -> None:
    class Dataset:
        scaled_targets = np.array(
            [[1.0, 0.0, 2.0], [-1.0, 50.0, -2.0], [0.5, -50.0, 1.0]],
            dtype=np.float32,
        )
        masks = np.array(
            [[True, False, True], [True, False, True], [True, False, True]]
        )

        def __len__(self) -> int:
            return len(self.scaled_targets)

    basis = fit_masked_response_basis(Dataset(), rank=1)
    assert basis is not None
    assert basis.shape == (1, 3)
    np.testing.assert_allclose(basis[:, 1], 0.0, atol=1e-6)


def test_shared_response_basis_uses_only_designated_training_compounds() -> None:
    frame = pd.DataFrame(
        {
            "compound_id": ["train_a", "train_b", "validation"],
            "cell_line": ["A", "A", "A"],
            "dose_nM": [10.0, 10.0, 10.0],
            "duration_hours": [24.0, 24.0, 24.0],
            "delta_y": [
                np.array([1.0, -1.0, 0.0], dtype=np.float32),
                np.array([-1.0, 1.0, 0.0], dtype=np.float32),
                np.array([1e6, 0.0, -1e6], dtype=np.float32),
            ],
            "target_mask": [np.ones(3, dtype=bool)] * 3,
        }
    )
    split = {
        "splits": {
            "train": ["train_a", "train_b"],
            "validation": ["validation"],
        }
    }
    basis = fit_shared_response_basis(frame, split, target_scaling="global", rank=1)
    changed = frame.copy()
    changed.at[2, "delta_y"] = np.array([-1e9, 1e9, 1e9], dtype=np.float32)
    changed_basis = fit_shared_response_basis(
        changed, split, target_scaling="global", rank=1
    )
    np.testing.assert_allclose(np.abs(basis), np.abs(changed_basis), atol=1e-6)
