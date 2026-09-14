from __future__ import annotations

import torch
import pytest
from torch import nn

from scripts.train_prc import (
    compound_derangement,
    freeze_parameter_prefixes,
    load_transfer_state,
    restrict_trainable_parameter_prefixes,
)
from my_star.prc_model import (
    ChemicalPerturbationEncoder,
    PRCFactorizedModel,
    masked_correlation_loss,
    distribution_moment_loss,
    masked_reconstruction_loss,
    masked_response_norm_loss,
    masked_topk_reconstruction_loss,
    response_signature_loss,
    supervised_contrastive_loss,
    target_gene_auxiliary_loss,
)


def make_model() -> PRCFactorizedModel:
    return PRCFactorizedModel(
        readout_features=torch.randn(17, 3),
        num_cell_lines=4,
        num_domains=2,
        descriptor_input_dim=8,
        latent_dim=32,
        readout_dim=24,
        fingerprint_hidden_dim=48,
        baseline_hidden_dim=40,
        dropout=0.0,
    )


def test_prc_factorized_model_and_readout_query_contract() -> None:
    model = make_model()
    inputs = (
        torch.zeros(3, 2048),
        torch.zeros(3, 8),
        torch.tensor([1, 2, 3]),
        torch.tensor([0, 1, 0]),
        torch.tensor([[0.0, 0.5, 1.0], [0.5, 0.5, 0.0], [-0.5, 0.5, 1.0]]),
        torch.zeros(3, 17),
    )
    full = model(*inputs)
    indices = torch.tensor([1, 4, 9])
    queried = model(*inputs, readout_indices=indices)
    assert full.shape == (3, 17)
    assert queried.shape == (3, 3)
    assert torch.allclose(full[:, indices], queried, atol=1e-6)


def test_prc_model_accepts_train_only_response_basis() -> None:
    basis = torch.linalg.qr(torch.randn(17, 8)).Q.T
    model = PRCFactorizedModel(
        readout_features=torch.randn(17, 3),
        num_cell_lines=4,
        num_domains=2,
        response_basis=basis,
        descriptor_input_dim=8,
        latent_dim=16,
        readout_dim=8,
        fingerprint_hidden_dim=24,
        baseline_hidden_dim=20,
        dropout=0.0,
    )
    output = model(
        torch.zeros(2, 2048),
        torch.zeros(2, 8),
        torch.tensor([1, 2]),
        torch.tensor([0, 1]),
        torch.zeros(2, 3),
        torch.zeros(2, 17),
    )
    assert output.shape == (2, 17)
    assert torch.isfinite(output).all()


def test_readout_residual_gate_initialization_is_configurable() -> None:
    model = PRCFactorizedModel(
        readout_features=torch.randn(7, 3),
        num_cell_lines=2,
        num_domains=1,
        descriptor_input_dim=8,
        latent_dim=16,
        readout_dim=8,
        fingerprint_hidden_dim=24,
        baseline_hidden_dim=20,
        dropout=0.0,
        learned_readout_initial_logit=-3.0,
    )
    assert torch.allclose(model.learned_readout_logit, torch.tensor(-3.0))


def test_xpert_lite_auxiliary_heads_and_condition_encoder() -> None:
    model = PRCFactorizedModel(
        readout_features=torch.randn(11, 3),
        num_cell_lines=3,
        num_domains=2,
        descriptor_input_dim=8,
        latent_dim=16,
        readout_dim=8,
        fingerprint_hidden_dim=24,
        baseline_hidden_dim=20,
        dropout=0.0,
        enable_distribution_head=True,
        enable_response_signature_head=True,
        target_auxiliary_dim=5,
    )
    output = model(
        torch.zeros(2, 2048),
        torch.zeros(2, 8),
        torch.tensor([1, 2]),
        torch.tensor([0, 1]),
        torch.tensor([[-0.5, 0.2, 1.0], [0.7, 0.2, 0.0]]),
        torch.zeros(2, 11),
        return_aux=True,
    )
    assert isinstance(output, dict)
    assert output["mean"].shape == (2, 11)
    assert output["log_std"].shape == (2, 11)
    assert output["response_signature"].shape == (2, 11)
    assert output["chemical_embedding"].shape == (2, 16)
    assert output["target_gene_logits"].shape == (2, 5)


def test_gated_factorized_decoder_supports_pca_and_auxiliary_output() -> None:
    model = PRCFactorizedModel(
        readout_features=torch.randn(7, 5),
        num_cell_lines=2,
        num_domains=2,
        response_basis=torch.randn(4, 7),
        descriptor_input_dim=23,
        latent_dim=16,
        readout_dim=4,
        fingerprint_hidden_dim=16,
        baseline_hidden_dim=8,
        descriptor_hidden_dim=8,
        dropout=0.0,
        fingerprint_dropout=0.0,
        readout_attention_heads=2,
        enable_distribution_head=True,
        decoder_mode="gated_factorized",
    )
    output = model(
        torch.randn(2, 2048),
        torch.randn(2, 23),
        torch.tensor([1, 2]),
        torch.tensor([0, 1]),
        torch.randn(2, 3),
        torch.randn(2, 7),
        return_aux=True,
    )
    assert output["mean"].shape == (2, 7)
    assert output["log_std"].shape == (2, 7)


def test_raw_condition_and_disabled_readout_attention_match_contract() -> None:
    model = PRCFactorizedModel(
        readout_features=torch.randn(7, 3),
        num_cell_lines=2,
        num_domains=1,
        response_basis=torch.randn(4, 7),
        descriptor_input_dim=8,
        latent_dim=16,
        readout_dim=4,
        fingerprint_hidden_dim=16,
        baseline_hidden_dim=8,
        descriptor_hidden_dim=8,
        dropout=0.0,
        fingerprint_dropout=0.0,
        readout_attention_heads=2,
        readout_attention_layers=0,
        decoder_mode="gated_factorized",
        condition_mode="raw",
    )
    assert isinstance(model.condition_encoder, nn.Identity)
    assert isinstance(model.readout_contextualizer, nn.Identity)
    output = model(
        torch.randn(2, 2048),
        torch.randn(2, 8),
        torch.tensor([1, 2]),
        torch.tensor([0, 0]),
        torch.randn(2, 3),
        torch.randn(2, 7),
    )
    assert output.shape == (2, 7)


def test_graph_chemical_encoder_supports_variable_atom_padding() -> None:
    model = PRCFactorizedModel(
        readout_features=torch.randn(7, 3),
        num_cell_lines=2,
        num_domains=1,
        descriptor_input_dim=8,
        latent_dim=16,
        readout_dim=8,
        fingerprint_hidden_dim=16,
        baseline_hidden_dim=8,
        dropout=0.0,
        fingerprint_dropout=0.0,
        readout_attention_heads=2,
        chemical_encoder_mode="graph",
        graph_message_layers=2,
    )
    atom_features = torch.zeros(2, 4, 7, dtype=torch.long)
    atom_features[..., 0] = 6
    atom_mask = torch.tensor([[True, True, False, False], [True, True, True, True]])
    bond_types = torch.zeros(2, 4, 4, dtype=torch.long)
    bond_types[0, 0, 1] = bond_types[0, 1, 0] = 1
    bond_types[1, 0, 1] = bond_types[1, 1, 0] = 1
    output = model(
        torch.zeros(2, 2048),
        torch.zeros(2, 8),
        torch.tensor([1, 2]),
        torch.tensor([0, 0]),
        torch.zeros(2, 3),
        torch.zeros(2, 7),
        atom_features=atom_features,
        atom_mask=atom_mask,
        bond_types=bond_types,
    )
    assert output.shape == (2, 7)
    assert torch.isfinite(output).all()


def test_standalone_encoder_uses_prc_transfer_compatible_parameter_names() -> None:
    encoder = ChemicalPerturbationEncoder(
        descriptor_input_dim=8,
        latent_dim=16,
        hidden_dim=24,
        descriptor_hidden_dim=8,
        dropout=0.0,
    )
    names = set(encoder.state_dict())
    assert any(name.startswith("chemical_encoder.") for name in names)
    assert any(name.startswith("descriptor_encoder.") for name in names)
    assert any(name.startswith("chemical_norm.") for name in names)
    output = encoder(torch.zeros(2, 2048), torch.zeros(2, 8))
    assert output.shape == (2, 16)


def test_compound_derangement_is_deterministic_and_has_no_fixed_points() -> None:
    first = compound_derangement(["a", "b", "c", "d"], seed=7)
    second = compound_derangement(["d", "c", "b", "a"], seed=7)
    assert first == second
    assert all(source != target for source, target in first.items())


def test_context_and_readout_ablation_flags_preserve_output_contract() -> None:
    model = PRCFactorizedModel(
        readout_features=torch.randn(7, 5),
        num_cell_lines=2,
        num_domains=2,
        descriptor_input_dim=8,
        latent_dim=16,
        readout_dim=8,
        fingerprint_hidden_dim=16,
        baseline_hidden_dim=8,
        dropout=0.0,
        fingerprint_dropout=0.0,
        readout_attention_heads=2,
        use_cell_conditioning=False,
        use_domain_conditioning=False,
        use_dose_time_conditioning=False,
        use_baseline_conditioning=False,
        use_readout_metadata=False,
    )
    output = model(
        torch.randn(2, 2048),
        torch.randn(2, 8),
        torch.tensor([1, 2]),
        torch.tensor([0, 1]),
        torch.randn(2, 3),
        torch.randn(2, 7),
    )
    assert output.shape == (2, 7)
    assert torch.isfinite(output).all()


def test_masked_losses_ignore_unobserved_targets() -> None:
    prediction = torch.tensor([[1.0, 1000.0, -1.0], [0.0, -500.0, 2.0]])
    target = torch.tensor([[0.0, -1000.0, 0.0], [1.0, 500.0, 1.0]])
    mask = torch.tensor([[True, False, True], [True, False, True]])
    base = masked_reconstruction_loss(prediction, target, mask)
    changed = target.clone()
    changed[:, 1] = 1e9
    assert torch.allclose(base, masked_reconstruction_loss(prediction, changed, mask))
    assert torch.isfinite(masked_correlation_loss(prediction, target, mask))
    assert torch.isfinite(masked_topk_reconstruction_loss(prediction, target, mask, 2))
    assert torch.isfinite(masked_response_norm_loss(prediction, target, mask))


def test_auxiliary_losses_handle_sparse_supervision() -> None:
    embedding = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    labels = torch.tensor([1, 1, 0])
    assert torch.isfinite(supervised_contrastive_loss(embedding, labels))
    signature = response_signature_loss(
        torch.ones(2, 3),
        torch.ones(2, 3),
        torch.tensor([1.0, 0.0]),
    )
    assert signature < 1e-6
    moment = distribution_moment_loss(
        torch.zeros(2, 3),
        torch.ones(2, 3),
        torch.tensor([[True, False, True], [False, False, False]]),
    )
    assert torch.isfinite(moment)
    target_loss = target_gene_auxiliary_loss(
        torch.zeros(2, 3),
        torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        torch.tensor([1.0, 0.0]),
        positive_weight=torch.tensor([2.0, 2.0, 2.0]),
    )
    assert torch.isfinite(target_loss)


def test_partial_transfer_loads_only_requested_module_prefixes() -> None:
    source = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2))
    target = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2))
    target_second = target[1].weight.detach().clone()
    checkpoint = {"model_state_dict": source.state_dict()}
    loaded = load_transfer_state(target, checkpoint, prefixes=["0"])
    assert loaded == ["0.bias", "0.weight"]
    assert torch.allclose(target[0].weight, source[0].weight)
    assert torch.allclose(target[1].weight, target_second)


def test_freeze_parameter_prefixes_leaves_other_modules_trainable() -> None:
    model = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2))
    frozen = freeze_parameter_prefixes(model, prefixes=["0"])
    assert frozen == ["0.bias", "0.weight"]
    assert not model[0].weight.requires_grad
    assert model[1].weight.requires_grad


def test_freeze_parameter_prefixes_rejects_unknown_prefix() -> None:
    model = nn.Linear(3, 2)
    with pytest.raises(ValueError, match="No model parameters"):
        freeze_parameter_prefixes(model, prefixes=["missing"])


def test_restrict_trainable_parameter_prefixes_freezes_complement() -> None:
    model = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2))
    frozen, trainable = restrict_trainable_parameter_prefixes(model, prefixes=["1"])
    assert frozen == ["0.bias", "0.weight"]
    assert trainable == ["1.bias", "1.weight"]
    assert not model[0].weight.requires_grad
    assert model[1].weight.requires_grad
