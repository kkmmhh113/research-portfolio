from __future__ import annotations

import pytest

from our_star.observation import ObservationSpec, predict_observation


def _identity(**changes):
    values = {
        "observation_id": "plasma_total_enalaprilat",
        "analyte_id": "enalaprilat",
        "source_matrix": "plasma",
        "modeled_matrix": "plasma",
        "quantity": "total",
        "matrix_bridge": "identity",
        "fixed_matrix_ratio": None,
        "bound_recovery_fraction": 1.0,
        "residual_model_id": None,
        "provenance": "synthetic observation fixture",
    }
    values.update(changes)
    return ObservationSpec(**values)


def test_total_observation_can_include_declared_bound_recovery() -> None:
    predicted = predict_observation(
        _identity(bound_recovery_fraction=0.5),
        mobile_total_amount_umol=8.0,
        bound_amount_umol=4.0,
        modeled_volume_l=5.0,
        molecular_weight_g_mol=350.0,
        fraction_unbound=0.6,
    )
    assert predicted.concentration_umol_l == pytest.approx(2.0)
    assert predicted.concentration_mg_l == pytest.approx(0.7)
    assert not predicted.proxy_without_conversion


def test_unbound_observation_excludes_bound_pool() -> None:
    spec = _identity(quantity="unbound", bound_recovery_fraction=0.0)
    predicted = predict_observation(
        spec,
        mobile_total_amount_umol=10.0,
        bound_amount_umol=100.0,
        modeled_volume_l=5.0,
        molecular_weight_g_mol=300.0,
        fraction_unbound=0.25,
    )
    assert predicted.concentration_umol_l == pytest.approx(0.5)


def test_matrix_mismatch_requires_an_explicit_bridge() -> None:
    with pytest.raises(ValueError, match="identical matrices"):
        _identity(source_matrix="serum")

    fixed = _identity(
        source_matrix="serum",
        matrix_bridge="fixed_ratio",
        fixed_matrix_ratio=0.8,
    )
    prediction = predict_observation(
        fixed,
        mobile_total_amount_umol=10.0,
        bound_amount_umol=0.0,
        modeled_volume_l=5.0,
        molecular_weight_g_mol=100.0,
        fraction_unbound=1.0,
    )
    assert prediction.concentration_umol_l == pytest.approx(1.6)

    proxy = _identity(
        source_matrix="serum",
        matrix_bridge="declared_proxy_no_conversion",
        bound_recovery_fraction=0.0,
    )
    proxy_prediction = predict_observation(
        proxy,
        mobile_total_amount_umol=10.0,
        bound_amount_umol=0.0,
        modeled_volume_l=5.0,
        molecular_weight_g_mol=100.0,
        fraction_unbound=1.0,
    )
    assert proxy_prediction.concentration_umol_l == pytest.approx(2.0)
    assert proxy_prediction.proxy_without_conversion


def test_invalid_observation_definitions_fail_closed() -> None:
    with pytest.raises(ValueError, match="cannot recover bound"):
        _identity(quantity="unbound", bound_recovery_fraction=0.1)
    with pytest.raises(ValueError, match="requires fixed_matrix_ratio"):
        _identity(matrix_bridge="fixed_ratio", fixed_matrix_ratio=None)
    with pytest.raises(ValueError, match=">= 0"):
        predict_observation(
            _identity(),
            mobile_total_amount_umol=-1.0,
            bound_amount_umol=0.0,
            modeled_volume_l=5.0,
            molecular_weight_g_mol=100.0,
            fraction_unbound=1.0,
        )
