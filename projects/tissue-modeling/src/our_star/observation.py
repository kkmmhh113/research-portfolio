"""Explicit mapping from model amounts to study-reported observations.

The observation layer never renames serum to plasma or total concentration to
unbound concentration.  Any bridge is represented by an immutable
``ObservationSpec`` and remains visible in result metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


Matrix = Literal["plasma", "serum", "whole_blood"]
Quantity = Literal["total", "unbound"]
MatrixBridge = Literal["identity", "fixed_ratio", "declared_proxy_no_conversion"]


@dataclass(frozen=True)
class ObservationSpec:
    observation_id: str
    analyte_id: str
    source_matrix: Matrix
    modeled_matrix: Matrix
    quantity: Quantity
    matrix_bridge: MatrixBridge
    fixed_matrix_ratio: float | None
    bound_recovery_fraction: float
    residual_model_id: str | None
    provenance: str

    def __post_init__(self) -> None:
        if not self.observation_id.strip() or not self.analyte_id.strip():
            raise ValueError("observation_id and analyte_id are required")
        if self.source_matrix not in {"plasma", "serum", "whole_blood"}:
            raise ValueError("unsupported source_matrix")
        if self.modeled_matrix not in {"plasma", "serum", "whole_blood"}:
            raise ValueError("unsupported modeled_matrix")
        if self.quantity not in {"total", "unbound"}:
            raise ValueError("quantity must be total or unbound")
        if self.matrix_bridge not in {
            "identity",
            "fixed_ratio",
            "declared_proxy_no_conversion",
        }:
            raise ValueError("unsupported matrix_bridge")
        if self.matrix_bridge == "identity":
            if self.source_matrix != self.modeled_matrix:
                raise ValueError("identity bridge requires identical matrices")
            if self.fixed_matrix_ratio is not None:
                raise ValueError("identity bridge must not define a matrix ratio")
        elif self.matrix_bridge == "fixed_ratio":
            if self.fixed_matrix_ratio is None:
                raise ValueError("fixed_ratio bridge requires fixed_matrix_ratio")
            ratio = float(self.fixed_matrix_ratio)
            if not np.isfinite(ratio) or ratio <= 0.0:
                raise ValueError("fixed_matrix_ratio must be finite and > 0")
            object.__setattr__(self, "fixed_matrix_ratio", ratio)
        elif self.fixed_matrix_ratio is not None:
            raise ValueError(
                "declared_proxy_no_conversion must not define a matrix ratio"
            )
        recovery = float(self.bound_recovery_fraction)
        if not np.isfinite(recovery) or not 0.0 <= recovery <= 1.0:
            raise ValueError("bound_recovery_fraction must be in [0, 1]")
        if self.quantity == "unbound" and recovery != 0.0:
            raise ValueError("unbound observations cannot recover bound analyte")
        object.__setattr__(self, "bound_recovery_fraction", recovery)
        if self.residual_model_id is not None and not self.residual_model_id.strip():
            raise ValueError("residual_model_id cannot be blank")
        if not self.provenance.strip():
            raise ValueError("observation provenance is required")

    @property
    def matrix_ratio(self) -> float:
        return 1.0 if self.fixed_matrix_ratio is None else self.fixed_matrix_ratio

    @property
    def uses_proxy_matrix(self) -> bool:
        return self.matrix_bridge == "declared_proxy_no_conversion"


@dataclass(frozen=True)
class PredictedObservation:
    observation_id: str
    analyte_id: str
    source_matrix: Matrix
    modeled_matrix: Matrix
    quantity: Quantity
    concentration_umol_l: float
    concentration_mg_l: float
    matrix_bridge: MatrixBridge
    proxy_without_conversion: bool


def predict_observation(
    spec: ObservationSpec,
    *,
    mobile_total_amount_umol: float,
    bound_amount_umol: float,
    modeled_volume_l: float,
    molecular_weight_g_mol: float,
    fraction_unbound: float,
) -> PredictedObservation:
    """Map finite model amounts to one explicitly declared observation."""

    if not isinstance(spec, ObservationSpec):
        raise TypeError("spec must be an ObservationSpec")
    mobile = float(mobile_total_amount_umol)
    bound = float(bound_amount_umol)
    volume = float(modeled_volume_l)
    molecular_weight = float(molecular_weight_g_mol)
    fu = float(fraction_unbound)
    for label, value in (("mobile_total_amount_umol", mobile), ("bound_amount_umol", bound)):
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{label} must be finite and >= 0")
    if not np.isfinite(volume) or volume <= 0.0:
        raise ValueError("modeled_volume_l must be finite and > 0")
    if not np.isfinite(molecular_weight) or molecular_weight <= 0.0:
        raise ValueError("molecular_weight_g_mol must be finite and > 0")
    if not np.isfinite(fu) or not 0.0 <= fu <= 1.0:
        raise ValueError("fraction_unbound must be finite and in [0, 1]")

    if spec.quantity == "unbound":
        modeled_concentration = fu * mobile / volume
    else:
        recovered = mobile + spec.bound_recovery_fraction * bound
        modeled_concentration = recovered / volume
    observed_concentration = spec.matrix_ratio * modeled_concentration
    return PredictedObservation(
        observation_id=spec.observation_id,
        analyte_id=spec.analyte_id,
        source_matrix=spec.source_matrix,
        modeled_matrix=spec.modeled_matrix,
        quantity=spec.quantity,
        concentration_umol_l=float(observed_concentration),
        concentration_mg_l=float(observed_concentration * molecular_weight / 1000.0),
        matrix_bridge=spec.matrix_bridge,
        proxy_without_conversion=spec.uses_proxy_matrix,
    )
