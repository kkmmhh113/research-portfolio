"""Explicit individual and cohort prediction estimands."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence

import numpy as np


class PredictionEstimand(str, Enum):
    INDIVIDUAL = "individual"
    COHORT_MEAN = "cohort_mean"
    COHORT_MEDIAN = "cohort_median"


@dataclass(frozen=True)
class PredictionTarget:
    """One quantity and the statistical unit to which it refers."""

    target_id: str
    quantity: str
    unit: str
    estimand: PredictionEstimand | str
    individual_id: str | None = None
    cohort_id: str | None = None

    def __post_init__(self) -> None:
        for label, value in (
            ("target_id", self.target_id),
            ("quantity", self.quantity),
            ("unit", self.unit),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{label} is required")
            if value != value.strip():
                raise ValueError(f"{label} must not contain surrounding whitespace")
        try:
            estimand = PredictionEstimand(self.estimand)
        except ValueError as error:
            raise ValueError(
                "estimand must be individual, cohort_mean, or cohort_median"
            ) from error
        if estimand is PredictionEstimand.INDIVIDUAL:
            if self.individual_id is not None and (
                not isinstance(self.individual_id, str)
                or not self.individual_id.strip()
            ):
                raise ValueError(
                    "individual_id must be a non-empty string when supplied"
                )
            if self.cohort_id is not None:
                raise ValueError("individual target must not define cohort_id")
        else:
            if self.individual_id is not None:
                raise ValueError("cohort target must not define individual_id")
            if self.cohort_id is not None and (
                not isinstance(self.cohort_id, str) or not self.cohort_id.strip()
            ):
                raise ValueError("cohort_id must be a non-empty string when supplied")
        object.__setattr__(self, "estimand", estimand)

    @classmethod
    def individual(
        cls,
        target_id: str,
        quantity: str,
        unit: str,
        individual_id: str | None = None,
    ) -> "PredictionTarget":
        return cls(
            target_id=target_id,
            quantity=quantity,
            unit=unit,
            estimand=PredictionEstimand.INDIVIDUAL,
            individual_id=individual_id,
        )

    @classmethod
    def cohort_mean(
        cls,
        target_id: str,
        quantity: str,
        unit: str,
        cohort_id: str | None = None,
    ) -> "PredictionTarget":
        return cls(
            target_id=target_id,
            quantity=quantity,
            unit=unit,
            estimand=PredictionEstimand.COHORT_MEAN,
            cohort_id=cohort_id,
        )

    @classmethod
    def cohort_median(
        cls,
        target_id: str,
        quantity: str,
        unit: str,
        cohort_id: str | None = None,
    ) -> "PredictionTarget":
        return cls(
            target_id=target_id,
            quantity=quantity,
            unit=unit,
            estimand=PredictionEstimand.COHORT_MEDIAN,
            cohort_id=cohort_id,
        )

    @property
    def is_group_summary(self) -> bool:
        return self.estimand in {
            PredictionEstimand.COHORT_MEAN,
            PredictionEstimand.COHORT_MEDIAN,
        }

    def aggregate(
        self,
        values_by_individual: Mapping[str, float],
        *,
        ordered_individual_ids: Sequence[str] | None = None,
    ) -> float:
        """Reduce individual model outputs to this target's estimand."""

        if not values_by_individual:
            raise ValueError("at least one individual prediction is required")
        if ordered_individual_ids is None:
            ids = tuple(values_by_individual)
        else:
            ids = tuple(ordered_individual_ids)
            if len(ids) != len(set(ids)):
                raise ValueError("ordered individual ids must be unique")
            if set(ids) != set(values_by_individual):
                raise ValueError(
                    "ordered individual ids must exactly match prediction keys"
                )
        vector = np.asarray(
            [float(values_by_individual[individual_id]) for individual_id in ids],
            dtype=np.float64,
        )
        if np.any(~np.isfinite(vector)):
            raise ValueError("individual predictions must be finite")
        if self.estimand is PredictionEstimand.INDIVIDUAL:
            individual_id = self.individual_id
            if individual_id is None:
                if len(ids) != 1:
                    raise ValueError(
                        "an individual target without individual_id requires "
                        "exactly one prediction unit"
                    )
                individual_id = ids[0]
            if individual_id not in values_by_individual:
                raise KeyError(
                    f"individual target references unknown id: {individual_id}"
                )
            return float(values_by_individual[individual_id])
        if self.estimand is PredictionEstimand.COHORT_MEAN:
            return float(np.mean(vector))
        return float(np.median(vector))


def validate_prediction_targets(
    targets: Sequence[PredictionTarget],
) -> tuple[PredictionTarget, ...]:
    result = tuple(targets)
    if not result:
        raise ValueError("at least one PredictionTarget is required")
    if any(not isinstance(target, PredictionTarget) for target in result):
        raise TypeError("targets must contain only PredictionTarget values")
    ids = tuple(target.target_id for target in result)
    if len(set(ids)) != len(ids):
        raise ValueError("prediction target ids must be unique")
    return result


__all__ = [
    "PredictionEstimand",
    "PredictionTarget",
    "validate_prediction_targets",
]
