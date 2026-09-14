"""Mechanistic PBPK core for the first virtual Phase I milestone.

The model is intentionally amount based: every state is stored in milligrams so
that a delivered dose can be reconciled against drug remaining in the body,
feces, urine, and tracked metabolic products.  The liver is represented by
three serial, perfused zones.  Gut, kidney, central blood, and a rest-of-body
compartment provide the minimum systemic context needed to interpret an oral
dose.

This is research software, not a clinical dosing system.  Drug parameters must
come with external provenance before predictions can be treated as evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Mapping, Sequence

import numpy as np


class State(IntEnum):
    """Indices of amount states, all expressed in mg."""

    GUT_LUMEN_PARENT = 0
    GUT_WALL_PARENT = 1
    CENTRAL_PARENT = 2
    LIVER_ZONE1_PARENT = 3
    LIVER_ZONE2_PARENT = 4
    LIVER_ZONE3_PARENT = 5
    KIDNEY_PARENT = 6
    REST_PARENT = 7
    CENTRAL_METABOLITE = 8
    URINE_PARENT = 9
    URINE_METABOLITE = 10
    FECES_PARENT = 11
    OTHER_PRODUCTS = 12


STATE_COUNT = len(State)
STATE_NAMES = tuple(member.name.lower() for member in State)


@dataclass(frozen=True)
class DrugParameters:
    """Drug-specific PBPK parameters.

    Clearances are apparent clearances against the blood-equivalent
    concentration.  ``vmax`` values refer to a 70 kg reference adult with a
    1.8 L liver and are scaled by each virtual patient's physiology.
    """

    name: str
    smiles: str
    molecular_weight_g_mol: float
    fraction_unbound_plasma: float
    absorption_rate_h: float
    fraction_absorbed: float
    kp_gut: float
    kp_liver: float
    kp_kidney: float
    kp_rest: float
    liver_vmax_mg_h: float
    liver_km_mg_l: float
    liver_zone_activity: tuple[float, float, float]
    gut_vmax_mg_h: float
    gut_km_mg_l: float
    renal_clearance_l_h: float
    metabolite_renal_clearance_l_h: float
    metabolite_mass_yield: float
    parameter_provenance: str

    def __post_init__(self) -> None:
        positive = {
            "molecular_weight_g_mol": self.molecular_weight_g_mol,
            "absorption_rate_h": self.absorption_rate_h,
            "kp_gut": self.kp_gut,
            "kp_liver": self.kp_liver,
            "kp_kidney": self.kp_kidney,
            "kp_rest": self.kp_rest,
            "liver_km_mg_l": self.liver_km_mg_l,
            "gut_km_mg_l": self.gut_km_mg_l,
        }
        for label, value in positive.items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{label} must be finite and > 0, got {value}")
        nonnegative = {
            "liver_vmax_mg_h": self.liver_vmax_mg_h,
            "gut_vmax_mg_h": self.gut_vmax_mg_h,
            "renal_clearance_l_h": self.renal_clearance_l_h,
            "metabolite_renal_clearance_l_h": self.metabolite_renal_clearance_l_h,
        }
        for label, value in nonnegative.items():
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{label} must be finite and >= 0, got {value}")
        for label, value in {
            "fraction_unbound_plasma": self.fraction_unbound_plasma,
            "fraction_absorbed": self.fraction_absorbed,
            "metabolite_mass_yield": self.metabolite_mass_yield,
        }.items():
            if not 0 <= value <= 1:
                raise ValueError(f"{label} must be in [0, 1], got {value}")
        if len(self.liver_zone_activity) != 3:
            raise ValueError("liver_zone_activity must have exactly three entries")
        if any(value < 0 or not np.isfinite(value) for value in self.liver_zone_activity):
            raise ValueError("liver_zone_activity entries must be finite and nonnegative")
        if sum(self.liver_zone_activity) <= 0:
            raise ValueError("at least one liver zone must have positive activity")
        if not self.smiles.strip():
            raise ValueError("a molecular structure (SMILES) is required")
        if not self.parameter_provenance.strip():
            raise ValueError("parameter_provenance is required")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "DrugParameters":
        values = dict(raw)
        values["liver_zone_activity"] = tuple(
            float(value) for value in values["liver_zone_activity"]
        )
        return cls(**values)

    @property
    def normalized_zone_activity(self) -> tuple[float, float, float]:
        total = float(sum(self.liver_zone_activity))
        return tuple(value / total for value in self.liver_zone_activity)


@dataclass(frozen=True)
class PatientPhysiology:
    """One virtual patient's physiological parameters."""

    patient_id: str
    age_years: float
    sex: str
    body_weight_kg: float
    central_volume_l: float
    gut_volume_l: float
    liver_volume_l: float
    kidney_volume_l: float
    rest_volume_l: float
    portal_flow_l_h: float
    hepatic_artery_flow_l_h: float
    renal_flow_l_h: float
    rest_flow_l_h: float
    enzyme_activity_factor: float = 1.0
    liver_function_fraction: float = 1.0
    renal_function_fraction: float = 1.0
    genotype_label: str = "normal_metabolizer"

    def __post_init__(self) -> None:
        for label in (
            "body_weight_kg",
            "central_volume_l",
            "gut_volume_l",
            "liver_volume_l",
            "kidney_volume_l",
            "rest_volume_l",
            "portal_flow_l_h",
            "hepatic_artery_flow_l_h",
            "renal_flow_l_h",
            "rest_flow_l_h",
            "enzyme_activity_factor",
        ):
            value = float(getattr(self, label))
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{label} must be finite and > 0, got {value}")
        for label in ("liver_function_fraction", "renal_function_fraction"):
            value = float(getattr(self, label))
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{label} must be finite and in [0, 1], got {value}")
        if self.sex not in {"female", "male"}:
            raise ValueError("sex must be 'female' or 'male'")

    @property
    def cardiac_output_l_h(self) -> float:
        return (
            self.portal_flow_l_h
            + self.hepatic_artery_flow_l_h
            + self.renal_flow_l_h
            + self.rest_flow_l_h
        )

    @property
    def liver_flow_l_h(self) -> float:
        return self.portal_flow_l_h + self.hepatic_artery_flow_l_h


@dataclass(frozen=True)
class DoseEvent:
    time_h: float
    amount_mg: float
    route: str = "oral"

    def __post_init__(self) -> None:
        if self.time_h < 0 or not np.isfinite(self.time_h):
            raise ValueError("dose time must be finite and >= 0")
        if self.amount_mg <= 0 or not np.isfinite(self.amount_mg):
            raise ValueError("dose amount must be finite and > 0")
        if self.route not in {"oral", "iv_bolus"}:
            raise ValueError("route must be 'oral' or 'iv_bolus'")


def initial_state() -> np.ndarray:
    return np.zeros(STATE_COUNT, dtype=np.float64)


def apply_doses(state: np.ndarray, events: Sequence[DoseEvent]) -> np.ndarray:
    updated = np.asarray(state, dtype=np.float64).copy()
    for event in events:
        target = (
            State.GUT_LUMEN_PARENT
            if event.route == "oral"
            else State.CENTRAL_PARENT
        )
        updated[target] += event.amount_mg
    return updated


def _michaelis_menten(vmax_mg_h: float, km_mg_l: float, concentration_mg_l: float) -> float:
    concentration = max(float(concentration_mg_l), 0.0)
    if vmax_mg_h <= 0 or concentration <= 0:
        return 0.0
    return vmax_mg_h * concentration / (km_mg_l + concentration)


def legacy_pbpk_rhs(
    _time_h: float,
    state: np.ndarray,
    patient: PatientPhysiology,
    drug: DrugParameters,
) -> np.ndarray:
    """Original monolithic amount derivatives, retained as a golden reference."""

    amounts = np.maximum(np.asarray(state, dtype=np.float64), 0.0)
    derivative = np.zeros_like(amounts)

    central = amounts[State.CENTRAL_PARENT] / patient.central_volume_l
    gut_equivalent = (
        amounts[State.GUT_WALL_PARENT] / patient.gut_volume_l / drug.kp_gut
    )
    liver_equivalent = (
        amounts[
            [State.LIVER_ZONE1_PARENT, State.LIVER_ZONE2_PARENT, State.LIVER_ZONE3_PARENT]
        ]
        / (patient.liver_volume_l / 3.0)
        / drug.kp_liver
    )
    kidney_equivalent = (
        amounts[State.KIDNEY_PARENT] / patient.kidney_volume_l / drug.kp_kidney
    )
    rest_equivalent = amounts[State.REST_PARENT] / patient.rest_volume_l / drug.kp_rest

    lumen_loss = drug.absorption_rate_h * amounts[State.GUT_LUMEN_PARENT]
    absorbed = drug.fraction_absorbed * lumen_loss
    fecal = (1.0 - drug.fraction_absorbed) * lumen_loss

    reference_gut_volume_l = 1.2
    gut_vmax = (
        drug.gut_vmax_mg_h
        * patient.enzyme_activity_factor
        * patient.gut_volume_l
        / reference_gut_volume_l
    )
    gut_metabolism = _michaelis_menten(
        gut_vmax,
        drug.gut_km_mg_l,
        drug.fraction_unbound_plasma * gut_equivalent,
    )

    zone_weights = drug.normalized_zone_activity
    reference_liver_volume_l = 1.8
    liver_scale = (
        patient.enzyme_activity_factor
        * patient.liver_function_fraction
        * patient.liver_volume_l
        / reference_liver_volume_l
    )
    liver_metabolism = np.asarray(
        [
            _michaelis_menten(
                drug.liver_vmax_mg_h * liver_scale * zone_weights[zone],
                drug.liver_km_mg_l,
                drug.fraction_unbound_plasma * liver_equivalent[zone],
            )
            for zone in range(3)
        ],
        dtype=np.float64,
    )

    renal_scale = patient.renal_function_fraction * (patient.body_weight_kg / 70.0) ** 0.75
    renal_elimination = drug.renal_clearance_l_h * renal_scale * kidney_equivalent

    q_portal = patient.portal_flow_l_h
    q_hepatic_artery = patient.hepatic_artery_flow_l_h
    q_liver = patient.liver_flow_l_h
    q_renal = patient.renal_flow_l_h
    q_rest = patient.rest_flow_l_h

    derivative[State.GUT_LUMEN_PARENT] = -lumen_loss
    derivative[State.FECES_PARENT] = fecal
    derivative[State.GUT_WALL_PARENT] = (
        absorbed + q_portal * central - q_portal * gut_equivalent - gut_metabolism
    )

    derivative[State.LIVER_ZONE1_PARENT] = (
        q_portal * gut_equivalent
        + q_hepatic_artery * central
        - q_liver * liver_equivalent[0]
        - liver_metabolism[0]
    )
    derivative[State.LIVER_ZONE2_PARENT] = (
        q_liver * liver_equivalent[0]
        - q_liver * liver_equivalent[1]
        - liver_metabolism[1]
    )
    derivative[State.LIVER_ZONE3_PARENT] = (
        q_liver * liver_equivalent[1]
        - q_liver * liver_equivalent[2]
        - liver_metabolism[2]
    )

    derivative[State.KIDNEY_PARENT] = (
        q_renal * central - q_renal * kidney_equivalent - renal_elimination
    )
    derivative[State.REST_PARENT] = q_rest * central - q_rest * rest_equivalent
    derivative[State.CENTRAL_PARENT] = (
        q_liver * liver_equivalent[2]
        + q_renal * kidney_equivalent
        + q_rest * rest_equivalent
        - (q_portal + q_hepatic_artery + q_renal + q_rest) * central
    )
    derivative[State.URINE_PARENT] = renal_elimination

    total_metabolism = gut_metabolism + float(liver_metabolism.sum())
    formed_metabolite = drug.metabolite_mass_yield * total_metabolism
    other_products = (1.0 - drug.metabolite_mass_yield) * total_metabolism
    metabolite_concentration = (
        amounts[State.CENTRAL_METABOLITE] / patient.central_volume_l
    )
    metabolite_clearance = (
        drug.metabolite_renal_clearance_l_h
        * renal_scale
        * metabolite_concentration
    )
    derivative[State.CENTRAL_METABOLITE] = formed_metabolite - metabolite_clearance
    derivative[State.URINE_METABOLITE] = metabolite_clearance
    derivative[State.OTHER_PRODUCTS] = other_products

    return derivative


def pbpk_rhs(
    time_h: float,
    state: np.ndarray,
    patient: PatientPhysiology,
    drug: DrugParameters,
) -> np.ndarray:
    """Return derivatives from the modular organ-system assembly.

    ``legacy_pbpk_rhs`` remains independent so every structural refactor can be
    checked against the original equations rather than against itself.
    """

    from .models.legacy import assembled_legacy_rhs

    return assembled_legacy_rhs(time_h, state, patient, drug)


def concentration_record(
    state: np.ndarray,
    patient: PatientPhysiology,
    drug: DrugParameters,
) -> dict[str, float]:
    amounts = np.asarray(state, dtype=np.float64)
    zone_volume = patient.liver_volume_l / 3.0
    return {
        "plasma_parent_mg_l": float(amounts[State.CENTRAL_PARENT] / patient.central_volume_l),
        "plasma_metabolite_mg_l": float(
            amounts[State.CENTRAL_METABOLITE] / patient.central_volume_l
        ),
        "gut_parent_mg_l": float(amounts[State.GUT_WALL_PARENT] / patient.gut_volume_l),
        "liver_zone1_parent_mg_l": float(amounts[State.LIVER_ZONE1_PARENT] / zone_volume),
        "liver_zone2_parent_mg_l": float(amounts[State.LIVER_ZONE2_PARENT] / zone_volume),
        "liver_zone3_parent_mg_l": float(amounts[State.LIVER_ZONE3_PARENT] / zone_volume),
        "kidney_parent_mg_l": float(amounts[State.KIDNEY_PARENT] / patient.kidney_volume_l),
        "rest_parent_mg_l": float(amounts[State.REST_PARENT] / patient.rest_volume_l),
        "urine_parent_mg": float(amounts[State.URINE_PARENT]),
        "urine_metabolite_mg": float(amounts[State.URINE_METABOLITE]),
        "feces_parent_mg": float(amounts[State.FECES_PARENT]),
        "other_products_mg": float(amounts[State.OTHER_PRODUCTS]),
        "tracked_mass_mg": float(amounts.sum()),
    }


def total_tracked_mass_mg(state: np.ndarray) -> float:
    return float(np.asarray(state, dtype=np.float64).sum())
