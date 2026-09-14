import torch

from my_star.model import (
    StructureConditionedPerturbationModel,
    correlation_loss,
    reconstruction_loss,
)


def test_model_contract() -> None:
    model = StructureConditionedPerturbationModel(output_dim=32, num_cell_lines=3, latent_dim=64)
    output = model(
        torch.zeros(4, 2048),
        torch.zeros(4, 8),
        torch.ones(4, dtype=torch.long),
        torch.zeros(4, 2),
        torch.zeros(4, 32),
    )
    target = torch.randn(4, 32)
    assert output.shape == target.shape
    assert torch.isfinite(correlation_loss(output, target))


def test_factorized_low_rank_model_contract() -> None:
    basis = torch.linalg.qr(torch.randn(32, 8)).Q.T
    model = StructureConditionedPerturbationModel(
        output_dim=32,
        num_cell_lines=3,
        latent_dim=64,
        response_basis=basis,
        variant="factorized",
    )
    output = model(
        torch.zeros(4, 2048),
        torch.zeros(4, 8),
        torch.ones(4, dtype=torch.long),
        torch.zeros(4, 2),
        torch.zeros(4, 32),
    )
    assert output.shape == (4, 32)
    projected = output @ basis.T @ basis
    assert torch.allclose(output, projected, atol=1e-5)


def test_film_additive_model_contract() -> None:
    model = StructureConditionedPerturbationModel(
        output_dim=32,
        num_cell_lines=3,
        latent_dim=64,
        variant="film_additive",
        fingerprint_hidden_dim=64,
        fusion_depth=1,
        fingerprint_dropout=0.1,
    )
    output = model(
        torch.zeros(4, 2048),
        torch.zeros(4, 8),
        torch.ones(4, dtype=torch.long),
        torch.zeros(4, 2),
        torch.zeros(4, 32),
    )
    assert output.shape == (4, 32)
    assert torch.isfinite(output).all()


def test_reconstruction_loss_variants() -> None:
    prediction = torch.tensor([[0.0, 2.0]])
    target = torch.zeros_like(prediction)
    assert reconstruction_loss(prediction, target, "mse") > reconstruction_loss(
        prediction,
        target,
        "huber",
        huber_delta=1.0,
    )
    weights = torch.tensor([2.0, 0.5])
    assert reconstruction_loss(prediction, target, "mse", weight=weights) < reconstruction_loss(
        prediction,
        target,
        "mse",
    )
    sample_weights = torch.tensor([1.0, 0.25])
    sample_prediction = torch.tensor([[0.0, 0.0], [2.0, 2.0]])
    sample_target = torch.zeros_like(sample_prediction)
    assert reconstruction_loss(
        sample_prediction,
        sample_target,
        "mse",
        sample_weight=sample_weights,
    ) < reconstruction_loss(sample_prediction, sample_target, "mse")


def test_extended_descriptor_model_contract() -> None:
    model = StructureConditionedPerturbationModel(
        output_dim=16,
        num_cell_lines=3,
        latent_dim=32,
        descriptor_input_dim=23,
    )
    output = model(
        torch.zeros(2, 2048),
        torch.zeros(2, 23),
        torch.ones(2, dtype=torch.long),
        torch.zeros(2, 2),
        torch.zeros(2, 16),
    )
    assert output.shape == (2, 16)


def test_dose_gated_film_model_contract() -> None:
    model = StructureConditionedPerturbationModel(
        output_dim=16,
        num_cell_lines=3,
        latent_dim=32,
        variant="dose_gated_film",
    )
    output = model(
        torch.zeros(2, 2048),
        torch.zeros(2, 8),
        torch.ones(2, dtype=torch.long),
        torch.tensor([[-1.0, 0.5], [1.0, 0.5]]),
        torch.zeros(2, 16),
    )
    assert output.shape == (2, 16)
    assert torch.isfinite(output).all()


def test_adaptive_and_cell_dose_gated_model_contracts() -> None:
    for variant in (
        "dose_gated_film_adaptive",
        "dose_gated_film_cell",
        "dose_gated_film_cell_adaptive",
    ):
        model = StructureConditionedPerturbationModel(
            output_dim=16,
            num_cell_lines=3,
            latent_dim=32,
            variant=variant,
        )
        output = model(
            torch.zeros(4, 2048),
            torch.zeros(4, 8),
            torch.tensor([1, 1, 2, 2]),
            torch.tensor([[-1.0, 0.5], [1.0, 0.5], [-1.0, 0.5], [1.0, 0.5]]),
            torch.zeros(4, 16),
        )
        assert output.shape == (4, 16)
        assert torch.isfinite(output).all()
        assert model.dose_slope_raw is not None


def test_chemical_attention_without_baseline_contract() -> None:
    model = StructureConditionedPerturbationModel(
        output_dim=32,
        num_cell_lines=3,
        latent_dim=64,
        variant="chemical_attention",
        baseline_mode="none",
    )
    output = model(
        torch.zeros(2, 2048),
        torch.zeros(2, 8),
        torch.ones(2, dtype=torch.long),
        torch.zeros(2, 2),
        torch.zeros(2, 32),
    )
    assert output.shape == (2, 32)
