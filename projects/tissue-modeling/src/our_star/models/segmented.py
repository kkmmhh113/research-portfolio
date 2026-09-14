"""Formulation-aware multi-segment gastrointestinal PBPK system."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from ..core import ModelContext, StateRegistry, StateSpec, SystemAssembler
from ..organs import KidneyModule, LiverModule, RestOfBodyModule, SegmentedGutModule
from ..pbpk import DoseEvent, DrugParameters, PatientPhysiology
from .legacy import build_species_registry


SEGMENTS = ("duodenum", "jejunum", "ileum")
SEGMENTED_STATE_NAMES = (
    "stomach_solid_parent",
    "stomach_dissolved_parent",
    "duodenum_lumen_parent",
    "duodenum_wall_parent",
    "jejunum_lumen_parent",
    "jejunum_wall_parent",
    "ileum_lumen_parent",
    "ileum_wall_parent",
    "central_parent",
    "liver_zone1_parent",
    "liver_zone2_parent",
    "liver_zone3_parent",
    "kidney_parent",
    "rest_parent",
    "central_metabolite",
    "urine_parent",
    "urine_metabolite",
    "feces_parent",
    "other_products",
)


@dataclass(frozen=True)
class FormulationParameters:
    """Oral-product and gastrointestinal transport parameters."""

    formulation_id: str
    dosage_form: str
    initial_dissolved_fraction: float
    dissolution_rate_h: float
    gastric_emptying_rate_h: float
    segment_transit_rate_h: tuple[float, float, float]
    segment_absorption_rate_h: tuple[float, float, float]
    wall_volume_fraction: tuple[float, float, float]
    portal_flow_fraction: tuple[float, float, float]
    parameter_provenance: str

    def __post_init__(self) -> None:
        if not self.formulation_id.strip():
            raise ValueError("formulation_id is required")
        if self.dosage_form not in {
            "solution",
            "immediate_release_tablet",
            "immediate_release_capsule",
        }:
            raise ValueError("unsupported dosage_form")
        if not 0.0 <= self.initial_dissolved_fraction <= 1.0:
            raise ValueError("initial_dissolved_fraction must be in [0, 1]")
        if not np.isfinite(self.dissolution_rate_h) or self.dissolution_rate_h < 0:
            raise ValueError("dissolution_rate_h must be finite and >= 0")
        if not np.isfinite(self.gastric_emptying_rate_h) or self.gastric_emptying_rate_h <= 0:
            raise ValueError("gastric_emptying_rate_h must be finite and > 0")
        for label in (
            "segment_transit_rate_h",
            "segment_absorption_rate_h",
            "wall_volume_fraction",
            "portal_flow_fraction",
        ):
            values = tuple(float(value) for value in getattr(self, label))
            if len(values) != len(SEGMENTS):
                raise ValueError(f"{label} must contain one value per GI segment")
            if any(not np.isfinite(value) or value <= 0 for value in values):
                raise ValueError(f"{label} values must be finite and > 0")
            object.__setattr__(self, label, values)
        for label in ("wall_volume_fraction", "portal_flow_fraction"):
            if not np.isclose(sum(getattr(self, label)), 1.0, rtol=0.0, atol=1e-9):
                raise ValueError(f"{label} must sum to 1")
        if not self.parameter_provenance.strip():
            raise ValueError("parameter_provenance is required")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "FormulationParameters":
        values = dict(raw)
        for label in (
            "segment_transit_rate_h",
            "segment_absorption_rate_h",
            "wall_volume_fraction",
            "portal_flow_fraction",
        ):
            values[label] = tuple(float(value) for value in values[label])
        return cls(**values)


def _species_for_state(state_name: str) -> str:
    if state_name in {"central_metabolite", "urine_metabolite"}:
        return "lumped_metabolite"
    if state_name == "other_products":
        return "other_products"
    return "parent"


def build_segmented_state_registry(drug: DrugParameters) -> StateRegistry:
    species = build_species_registry(drug)
    sinks = {"urine_parent", "urine_metabolite", "feces_parent", "other_products"}
    return StateRegistry(
        (
            StateSpec(
                name=name,
                species_id=_species_for_state(name),
                compartment=name.rsplit("_", 1)[0],
                sink=name in sinks,
            )
            for name in SEGMENTED_STATE_NAMES
        ),
        species,
    )


class SegmentedPBPKModel:
    """Simulation adapter for a dissolution- and transit-resolved oral model."""

    model_id = "our_star.segmented_gi_pbpk.v0.2"

    def __init__(self, drug: DrugParameters, formulation: FormulationParameters) -> None:
        self.drug = drug
        self.formulation = formulation
        self.state_registry = build_segmented_state_registry(drug)
        self.assembler = SystemAssembler(
            self.state_registry,
            (SegmentedGutModule(), LiverModule(), KidneyModule(), RestOfBodyModule()),
        )

    def _validate_drug(self, drug: DrugParameters) -> None:
        if drug != self.drug:
            raise ValueError("SegmentedPBPKModel must be rebuilt when drug parameters change")

    def initial_state(self) -> np.ndarray:
        return self.state_registry.zeros()

    def apply_doses(
        self, state: np.ndarray, events: Sequence[DoseEvent]
    ) -> np.ndarray:
        updated = np.asarray(state, dtype=np.float64).copy()
        if updated.shape != (len(self.state_registry),):
            raise ValueError("dose state has the wrong shape for the segmented model")
        for event in events:
            if event.route == "iv_bolus":
                updated[self.state_registry.index("central_parent")] += event.amount_mg
                continue
            dissolved = event.amount_mg * self.formulation.initial_dissolved_fraction
            updated[self.state_registry.index("stomach_dissolved_parent")] += dissolved
            updated[self.state_registry.index("stomach_solid_parent")] += (
                event.amount_mg - dissolved
            )
        return updated

    def rhs(
        self,
        time_h: float,
        state: np.ndarray,
        patient: PatientPhysiology,
        drug: DrugParameters,
    ) -> np.ndarray:
        self._validate_drug(drug)
        return self.assembler.rhs(
            time_h,
            state,
            ModelContext(
                patient=patient,
                drug=drug,
                formulation=self.formulation,
            ),
        )

    def concentration_record(
        self,
        state: np.ndarray,
        patient: PatientPhysiology,
        drug: DrugParameters,
    ) -> dict[str, float]:
        self._validate_drug(drug)
        amounts = np.asarray(state, dtype=np.float64)
        amount = lambda name: float(amounts[self.state_registry.index(name)])
        zone_volume = patient.liver_volume_l / 3.0
        wall_amounts = [amount(f"{segment}_wall_parent") for segment in SEGMENTS]
        record = {
            "plasma_parent_mg_l": amount("central_parent") / patient.central_volume_l,
            "plasma_metabolite_mg_l": (
                amount("central_metabolite") / patient.central_volume_l
            ),
            "gut_parent_mg_l": sum(wall_amounts) / patient.gut_volume_l,
            "liver_zone1_parent_mg_l": amount("liver_zone1_parent") / zone_volume,
            "liver_zone2_parent_mg_l": amount("liver_zone2_parent") / zone_volume,
            "liver_zone3_parent_mg_l": amount("liver_zone3_parent") / zone_volume,
            "kidney_parent_mg_l": amount("kidney_parent") / patient.kidney_volume_l,
            "rest_parent_mg_l": amount("rest_parent") / patient.rest_volume_l,
            "urine_parent_mg": amount("urine_parent"),
            "urine_metabolite_mg": amount("urine_metabolite"),
            "feces_parent_mg": amount("feces_parent"),
            "other_products_mg": amount("other_products"),
            "stomach_solid_parent_mg": amount("stomach_solid_parent"),
            "stomach_dissolved_parent_mg": amount("stomach_dissolved_parent"),
            "tracked_mass_mg": float(amounts.sum()),
        }
        for index, segment in enumerate(SEGMENTS):
            record[f"{segment}_lumen_parent_mg"] = amount(
                f"{segment}_lumen_parent"
            )
            segment_wall_volume = (
                patient.gut_volume_l * self.formulation.wall_volume_fraction[index]
            )
            record[f"{segment}_wall_parent_mg_l"] = (
                amount(f"{segment}_wall_parent") / segment_wall_volume
            )
        return record
