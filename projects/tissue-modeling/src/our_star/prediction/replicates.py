"""Deterministic predictive replicates over immutable virtual subjects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np

from ..population import VirtualSubject
from .targets import PredictionTarget, validate_prediction_targets


SimulationCallback = Callable[
    [VirtualSubject, int, np.random.Generator], Mapping[str, float]
]


@dataclass(frozen=True)
class PredictiveReplicateTrace:
    schema_version: str
    seed: int
    bit_generator: str
    subject_ids: tuple[str, ...]
    target_ids: tuple[str, ...]
    stream_seeds: np.ndarray

    def __post_init__(self) -> None:
        if self.schema_version != "our_star.predictive_replicates.v0.4":
            raise ValueError("unsupported predictive-replicate trace schema")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, (int, np.integer))
            or self.seed < 0
        ):
            raise ValueError("seed must be a non-negative integer")
        if not self.subject_ids or len(set(self.subject_ids)) != len(self.subject_ids):
            raise ValueError("subject_ids must be non-empty and unique")
        if not self.target_ids or len(set(self.target_ids)) != len(self.target_ids):
            raise ValueError("target_ids must be non-empty and unique")
        seeds = np.asarray(self.stream_seeds, dtype=np.uint64).copy()
        if seeds.ndim != 2 or seeds.shape[1] != len(self.subject_ids):
            raise ValueError(
                "stream_seeds must be replicate-by-subject with matching subjects"
            )
        if seeds.shape[0] == 0:
            raise ValueError("at least one replicate stream is required")
        seeds.setflags(write=False)
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "stream_seeds", seeds)

    @property
    def replicate_count(self) -> int:
        return int(self.stream_seeds.shape[0])

    def record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "seed": self.seed,
            "bit_generator": self.bit_generator,
            "subject_ids": list(self.subject_ids),
            "target_ids": list(self.target_ids),
            "stream_seeds": [
                [int(value) for value in row] for row in self.stream_seeds
            ],
        }


@dataclass(frozen=True)
class PredictiveReplicates:
    """Replicate-by-target aggregate predictions and their RNG trace."""

    targets: tuple[PredictionTarget, ...]
    values: np.ndarray
    trace: PredictiveReplicateTrace

    def __post_init__(self) -> None:
        targets = validate_prediction_targets(self.targets)
        values = np.asarray(self.values, dtype=np.float64).copy()
        expected = (self.trace.replicate_count, len(targets))
        if values.shape != expected:
            raise ValueError(f"replicate values must have shape {expected}")
        if np.any(~np.isfinite(values)):
            raise ValueError("replicate values must be finite")
        if tuple(target.target_id for target in targets) != self.trace.target_ids:
            raise ValueError("target order must match predictive-replicate trace")
        values.setflags(write=False)
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "values", values)

    @property
    def target_ids(self) -> tuple[str, ...]:
        return tuple(target.target_id for target in self.targets)

    @property
    def replicate_count(self) -> int:
        return int(self.values.shape[0])

    def records(self) -> list[dict[str, float | int]]:
        return [
            {
                "replicate_index": replicate_index,
                **{
                    target.target_id: float(value)
                    for target, value in zip(self.targets, row, strict=True)
                },
            }
            for replicate_index, row in enumerate(self.values)
        ]


def generate_predictive_replicates(
    subjects: Sequence[VirtualSubject],
    targets: Sequence[PredictionTarget],
    simulator: SimulationCallback,
    replicate_count: int,
    *,
    seed: int = 0,
) -> PredictiveReplicates:
    """Evaluate a simulator with deterministic per-replicate/per-subject RNGs.

    The callback is invoked once per subject in each replicate as
    ``simulator(subject, replicate_index, rng)`` and returns a mapping from
    quantity names to finite scalar predictions.  Random streams are derived
    directly from ``(seed, replicate_index, subject_index)`` so extending the
    replicate count cannot change earlier streams.
    """

    subject_tuple = tuple(subjects)
    if not subject_tuple or any(
        not isinstance(subject, VirtualSubject) for subject in subject_tuple
    ):
        raise ValueError("subjects must contain at least one VirtualSubject")
    subject_ids = tuple(subject.subject_id for subject in subject_tuple)
    if len(set(subject_ids)) != len(subject_ids):
        raise ValueError("subject ids must be unique")
    target_tuple = validate_prediction_targets(targets)
    if not callable(simulator):
        raise TypeError("simulator must be callable")
    if (
        isinstance(replicate_count, bool)
        or not isinstance(replicate_count, (int, np.integer))
    ):
        raise TypeError("replicate_count must be an integer")
    replicate_count = int(replicate_count)
    if replicate_count <= 0:
        raise ValueError("replicate_count must be positive")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)

    stream_seeds = np.empty(
        (replicate_count, len(subject_tuple)), dtype=np.uint64
    )
    values = np.empty((replicate_count, len(target_tuple)), dtype=np.float64)
    required_quantities = tuple(dict.fromkeys(target.quantity for target in target_tuple))

    bit_generator_name: str | None = None
    for replicate_index in range(replicate_count):
        predictions_by_quantity: dict[str, dict[str, float]] = {
            quantity: {} for quantity in required_quantities
        }
        for subject_index, subject in enumerate(subject_tuple):
            sequence = np.random.SeedSequence(
                seed,
                spawn_key=(replicate_index, subject_index),
            )
            stream_seed = sequence.generate_state(1, dtype=np.uint64)[0]
            stream_seeds[replicate_index, subject_index] = stream_seed
            rng = np.random.default_rng(int(stream_seed))
            if bit_generator_name is None:
                bit_generator_name = rng.bit_generator.__class__.__name__
            raw = simulator(subject, replicate_index, rng)
            if not isinstance(raw, Mapping):
                raise TypeError("simulator must return a mapping")
            missing = [quantity for quantity in required_quantities if quantity not in raw]
            if missing:
                raise KeyError(f"simulator omitted required quantities: {missing}")
            for quantity in required_quantities:
                value = float(raw[quantity])
                if not np.isfinite(value):
                    raise FloatingPointError(
                        "simulator returned a non-finite prediction for "
                        f"{quantity}"
                    )
                predictions_by_quantity[quantity][subject.subject_id] = value

        for target_index, target in enumerate(target_tuple):
            values[replicate_index, target_index] = target.aggregate(
                predictions_by_quantity[target.quantity],
                ordered_individual_ids=subject_ids,
            )

    assert bit_generator_name is not None
    trace = PredictiveReplicateTrace(
        schema_version="our_star.predictive_replicates.v0.4",
        seed=seed,
        bit_generator=bit_generator_name,
        subject_ids=subject_ids,
        target_ids=tuple(target.target_id for target in target_tuple),
        stream_seeds=stream_seeds,
    )
    return PredictiveReplicates(targets=target_tuple, values=values, trace=trace)


__all__ = [
    "PredictiveReplicateTrace",
    "PredictiveReplicates",
    "SimulationCallback",
    "generate_predictive_replicates",
]
