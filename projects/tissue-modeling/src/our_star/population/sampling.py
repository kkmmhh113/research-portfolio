"""Deterministic virtual-subject sampling with complete latent traces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .random_effects import LatentDraws, draw_latent_effects
from .specification import PopulationSpecification, VirtualSubject


@dataclass(frozen=True)
class PopulationSamplingTrace:
    """Minimal deterministic recipe for reproducing a population sample."""

    schema_version: str
    population_id: str
    population_fingerprint: str
    random_effect_fingerprint: str
    random_effect_names: tuple[str, ...]
    seed: int
    bit_generator: str
    subject_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version != "our_star.population_sampling.v0.4":
            raise ValueError("unsupported population sampling trace schema")
        if not self.subject_ids or len(set(self.subject_ids)) != len(self.subject_ids):
            raise ValueError("sampling trace subject_ids must be non-empty and unique")
        if len(self.population_fingerprint) != 64:
            raise ValueError("population_fingerprint must be a SHA-256 digest")
        if len(self.random_effect_fingerprint) != 64:
            raise ValueError("random_effect_fingerprint must be a SHA-256 digest")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, (int, np.integer))
            or self.seed < 0
        ):
            raise ValueError("trace seed must be a non-negative integer")

    @property
    def count(self) -> int:
        return len(self.subject_ids)

    def record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "population_id": self.population_id,
            "population_fingerprint": self.population_fingerprint,
            "random_effect_fingerprint": self.random_effect_fingerprint,
            "random_effect_names": list(self.random_effect_names),
            "seed": int(self.seed),
            "bit_generator": self.bit_generator,
            "subject_ids": list(self.subject_ids),
        }


@dataclass(frozen=True)
class PopulationSample:
    subjects: tuple[VirtualSubject, ...]
    latent_draws: LatentDraws
    trace: PopulationSamplingTrace

    def __post_init__(self) -> None:
        subjects = tuple(self.subjects)
        if not subjects or any(
            not isinstance(subject, VirtualSubject) for subject in subjects
        ):
            raise ValueError("subjects must contain at least one VirtualSubject")
        if len(subjects) != self.latent_draws.count:
            raise ValueError("subject count must match latent draw count")
        if len(subjects) != self.trace.count:
            raise ValueError("subject count must match sampling trace")
        ids = tuple(subject.subject_id for subject in subjects)
        if ids != self.trace.subject_ids:
            raise ValueError("subject order must match sampling trace")
        if tuple(subject.draw_index for subject in subjects) != tuple(range(len(subjects))):
            raise ValueError("subject draw indices must be contiguous and ordered")
        if self.latent_draws.names != self.trace.random_effect_names:
            raise ValueError("latent name order must match sampling trace")
        object.__setattr__(self, "subjects", subjects)

    def records(self) -> list[dict[str, object]]:
        return [subject.record() for subject in self.subjects]


def sample_virtual_subjects(
    specification: PopulationSpecification,
    count: int,
    *,
    seed: int = 0,
    subject_prefix: str | None = None,
) -> PopulationSample:
    """Create deterministic subjects while retaining all latent draws.

    Latent effects used for flow composition are never emitted as independent
    component-flow multipliers.  Instead, the single total flow is partitioned
    by :class:`~our_star.population.FlowCompositionSpec`, preserving cardiac
    output by construction.
    """

    if not isinstance(specification, PopulationSpecification):
        raise TypeError("specification must be a PopulationSpecification")
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)):
        raise TypeError("count must be an integer")
    count = int(count)
    if count <= 0:
        raise ValueError("count must be positive")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    prefix = specification.population_id if subject_prefix is None else subject_prefix
    if not isinstance(prefix, str) or not prefix.strip():
        raise ValueError("subject_prefix must be a non-empty string")
    if prefix != prefix.strip():
        raise ValueError("subject_prefix must not contain surrounding whitespace")

    draws = draw_latent_effects(
        specification.random_effects,
        count,
        seed=seed,
    )
    subjects: list[VirtualSubject] = []
    for draw_index, row in enumerate(draws.values):
        latent = {
            name: float(value)
            for name, value in zip(draws.names, row, strict=True)
        }
        anatomy: dict[str, float] = {}
        mechanisms: dict[str, float] = {}
        for modifier in specification.modifiers:
            physical_value = modifier.apply(latent[modifier.name])
            destination = anatomy if modifier.domain == "anatomy" else mechanisms
            destination[modifier.target] = physical_value

        flow = specification.flow_composition
        if flow is not None:
            component_flows = flow.compose(anatomy[flow.total_target], latent)
            anatomy.update(component_flows)

        subjects.append(
            VirtualSubject(
                subject_id=f"{prefix}_{draw_index:05d}",
                population_id=specification.population_id,
                draw_index=draw_index,
                sampling_seed=seed,
                latent_effects=latent,
                anatomy_modifiers=anatomy,
                mechanism_modifiers=mechanisms,
            )
        )

    subject_ids = tuple(subject.subject_id for subject in subjects)
    trace = PopulationSamplingTrace(
        schema_version="our_star.population_sampling.v0.4",
        population_id=specification.population_id,
        population_fingerprint=specification.fingerprint,
        random_effect_fingerprint=specification.random_effects.fingerprint,
        random_effect_names=specification.random_effects.names,
        seed=seed,
        bit_generator=draws.bit_generator,
        subject_ids=subject_ids,
    )
    return PopulationSample(
        subjects=tuple(subjects),
        latent_draws=draws,
        trace=trace,
    )


def virtual_subject_records(
    subjects: Sequence[VirtualSubject],
) -> list[dict[str, object]]:
    """Serialize subjects without exposing mutable internal mappings."""

    return [subject.record() for subject in subjects]


__all__ = [
    "PopulationSample",
    "PopulationSamplingTrace",
    "sample_virtual_subjects",
    "virtual_subject_records",
]
