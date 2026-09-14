"""Virtual-patient population generation for the PBPK milestone."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping, Sequence

import numpy as np

from .pbpk import PatientPhysiology


@dataclass(frozen=True)
class PopulationSpecification:
    name: str
    count: int
    seed: int
    female_fraction: float
    age_min_years: float
    age_max_years: float
    female_weight_mean_kg: float
    female_weight_sd_kg: float
    male_weight_mean_kg: float
    male_weight_sd_kg: float
    physiology_cv: float
    genotype_labels: tuple[str, ...]
    genotype_probabilities: tuple[float, ...]
    genotype_activity_factors: tuple[float, ...]
    provenance: str

    def __post_init__(self) -> None:
        if self.count <= 0:
            raise ValueError("count must be > 0")
        if not 0 <= self.female_fraction <= 1:
            raise ValueError("female_fraction must be in [0, 1]")
        if self.age_min_years < 18 or self.age_max_years <= self.age_min_years:
            raise ValueError("population must define a valid adult age interval")
        if self.physiology_cv < 0 or self.physiology_cv >= 1:
            raise ValueError("physiology_cv must be in [0, 1)")
        if not (
            len(self.genotype_labels)
            == len(self.genotype_probabilities)
            == len(self.genotype_activity_factors)
        ):
            raise ValueError("genotype labels, probabilities, and factors must align")
        if any(probability < 0 for probability in self.genotype_probabilities):
            raise ValueError("genotype probabilities must be nonnegative")
        if not np.isclose(sum(self.genotype_probabilities), 1.0, atol=1e-8):
            raise ValueError("genotype probabilities must sum to one")
        if any(factor <= 0 for factor in self.genotype_activity_factors):
            raise ValueError("genotype activity factors must be positive")
        if not self.provenance.strip():
            raise ValueError("population provenance is required")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "PopulationSpecification":
        values = dict(raw)
        genotype = values.pop("genotype")
        values["genotype_labels"] = tuple(str(item["label"]) for item in genotype)
        values["genotype_probabilities"] = tuple(
            float(item["probability"]) for item in genotype
        )
        values["genotype_activity_factors"] = tuple(
            float(item["activity_factor"]) for item in genotype
        )
        return cls(**values)


def reference_patient(
    *,
    patient_id: str = "reference_70kg",
    liver_function_fraction: float = 1.0,
    renal_function_fraction: float = 1.0,
    enzyme_activity_factor: float = 1.0,
) -> PatientPhysiology:
    """Return the explicit 70 kg reference physiology used by drug configs."""

    return PatientPhysiology(
        patient_id=patient_id,
        age_years=40.0,
        sex="male",
        body_weight_kg=70.0,
        central_volume_l=5.0,
        gut_volume_l=1.2,
        liver_volume_l=1.8,
        kidney_volume_l=0.31,
        rest_volume_l=40.0,
        portal_flow_l_h=72.0,
        hepatic_artery_flow_l_h=18.0,
        renal_flow_l_h=72.0,
        rest_flow_l_h=138.0,
        enzyme_activity_factor=enzyme_activity_factor,
        liver_function_fraction=liver_function_fraction,
        renal_function_fraction=renal_function_fraction,
        genotype_label="reference",
    )


def _lognormal_multiplier(rng: np.random.Generator, cv: float, size: int) -> np.ndarray:
    if cv == 0:
        return np.ones(size, dtype=np.float64)
    sigma = np.sqrt(np.log1p(cv**2))
    mean = -0.5 * sigma**2
    return rng.lognormal(mean=mean, sigma=sigma, size=size)


def sample_population(
    specification: PopulationSpecification,
    *,
    count: int | None = None,
    seed: int | None = None,
    liver_function_fraction: float = 1.0,
    renal_function_fraction: float = 1.0,
) -> list[PatientPhysiology]:
    """Sample a reproducible, correlated virtual adult population.

    The sampler uses body size as a shared latent driver of organ volumes and
    flows rather than drawing every parameter independently.  This is still a
    compact demonstration distribution; a regulatory population would require
    externally estimated joint distributions and covariances.
    """

    n = specification.count if count is None else int(count)
    if n <= 0:
        raise ValueError("count must be > 0")
    rng = np.random.default_rng(specification.seed if seed is None else seed)

    female = rng.random(n) < specification.female_fraction
    sex = np.where(female, "female", "male")
    ages = rng.uniform(specification.age_min_years, specification.age_max_years, n)
    weights = np.where(
        female,
        rng.normal(specification.female_weight_mean_kg, specification.female_weight_sd_kg, n),
        rng.normal(specification.male_weight_mean_kg, specification.male_weight_sd_kg, n),
    )
    weights = np.clip(weights, 42.0, 145.0)
    size_scale = weights / 70.0
    flow_scale = size_scale**0.75

    genotype_indices = rng.choice(
        len(specification.genotype_labels),
        size=n,
        p=np.asarray(specification.genotype_probabilities, dtype=np.float64),
    )
    genotype_factors = np.asarray(specification.genotype_activity_factors)[genotype_indices]
    expression_noise = _lognormal_multiplier(rng, specification.physiology_cv, n)
    enzyme_factors = genotype_factors * expression_noise

    shared_organ_noise = _lognormal_multiplier(rng, specification.physiology_cv * 0.65, n)
    liver_noise = _lognormal_multiplier(rng, specification.physiology_cv * 0.55, n)
    kidney_noise = _lognormal_multiplier(rng, specification.physiology_cv * 0.55, n)
    flow_noise = _lognormal_multiplier(rng, specification.physiology_cv * 0.45, n)

    sex_blood_factor = np.where(female, 0.92, 1.0)
    central_volumes = 5.0 * size_scale * sex_blood_factor * shared_organ_noise
    gut_volumes = 1.2 * size_scale * shared_organ_noise
    liver_volumes = 1.8 * size_scale * liver_noise
    kidney_volumes = 0.31 * size_scale * kidney_noise
    rest_volumes = np.maximum(
        18.0,
        40.0 * size_scale * _lognormal_multiplier(rng, specification.physiology_cv * 0.35, n),
    )

    age_cardiac_factor = np.clip(1.0 - 0.0025 * np.maximum(ages - 40.0, 0.0), 0.78, 1.05)
    cardiac_outputs = 300.0 * flow_scale * age_cardiac_factor * flow_noise
    portal_fraction = np.clip(rng.normal(0.24, 0.012, n), 0.19, 0.29)
    hepatic_artery_fraction = np.clip(rng.normal(0.06, 0.006, n), 0.04, 0.08)
    renal_fraction = np.clip(rng.normal(0.24, 0.015, n), 0.18, 0.29)
    rest_fraction = 1.0 - portal_fraction - hepatic_artery_fraction - renal_fraction
    if np.any(rest_fraction <= 0):
        raise RuntimeError("sampled organ-flow fractions violate cardiac-output balance")

    age_renal_factor = np.clip(1.0 - 0.006 * np.maximum(ages - 40.0, 0.0), 0.48, 1.0)
    renal_noise = _lognormal_multiplier(rng, specification.physiology_cv * 0.35, n)
    sampled_renal_function = np.clip(
        age_renal_factor * renal_noise * renal_function_fraction,
        0.05,
        1.0,
    )

    patients: list[PatientPhysiology] = []
    for index in range(n):
        patients.append(
            PatientPhysiology(
                patient_id=f"{specification.name}_{index:05d}",
                age_years=float(ages[index]),
                sex=str(sex[index]),
                body_weight_kg=float(weights[index]),
                central_volume_l=float(central_volumes[index]),
                gut_volume_l=float(gut_volumes[index]),
                liver_volume_l=float(liver_volumes[index]),
                kidney_volume_l=float(kidney_volumes[index]),
                rest_volume_l=float(rest_volumes[index]),
                portal_flow_l_h=float(cardiac_outputs[index] * portal_fraction[index]),
                hepatic_artery_flow_l_h=float(
                    cardiac_outputs[index] * hepatic_artery_fraction[index]
                ),
                renal_flow_l_h=float(cardiac_outputs[index] * renal_fraction[index]),
                rest_flow_l_h=float(cardiac_outputs[index] * rest_fraction[index]),
                enzyme_activity_factor=float(enzyme_factors[index]),
                liver_function_fraction=float(liver_function_fraction),
                renal_function_fraction=float(sampled_renal_function[index]),
                genotype_label=specification.genotype_labels[genotype_indices[index]],
            )
        )
    return patients


def apply_function_scenario(
    patients: Sequence[PatientPhysiology],
    *,
    liver_function_fraction: float | None = None,
    renal_function_fraction: float | None = None,
) -> list[PatientPhysiology]:
    """Apply a documented functional impairment without deleting anatomy."""

    updated: list[PatientPhysiology] = []
    for patient in patients:
        changes: dict[str, float] = {}
        if liver_function_fraction is not None:
            changes["liver_function_fraction"] = float(liver_function_fraction)
        if renal_function_fraction is not None:
            changes["renal_function_fraction"] = float(renal_function_fraction)
        updated.append(replace(patient, **changes))
    return updated


def population_records(patients: Sequence[PatientPhysiology]) -> list[dict[str, Any]]:
    return [asdict(patient) for patient in patients]
