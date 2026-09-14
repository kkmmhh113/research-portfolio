"""Development-only prediction-interval calibration with frozen artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from types import MappingProxyType
from typing import Callable, Mapping, Sequence

import numpy as np

from .targets import PredictionTarget, validate_prediction_targets


@dataclass(frozen=True)
class CalibrationDataset:
    """Matched independent calibration units and predictive draws.

    ``predictive_draws`` has shape ``(draw, unit, target)``.  A unit is an
    individual for individual targets and a genuinely independent cohort/group
    for cohort-summary targets; repeated time points from one person are not
    independent calibration units.
    """

    unit_ids: tuple[str, ...]
    target_ids: tuple[str, ...]
    observed: np.ndarray
    predictive_draws: np.ndarray

    def __post_init__(self) -> None:
        unit_ids = tuple(self.unit_ids)
        target_ids = tuple(self.target_ids)
        if not unit_ids or any(
            not isinstance(unit, str) or not unit.strip() for unit in unit_ids
        ):
            raise ValueError("unit_ids must contain non-empty strings")
        if len(set(unit_ids)) != len(unit_ids):
            raise ValueError("calibration units must be independent and uniquely named")
        if not target_ids or any(
            not isinstance(target, str) or not target.strip()
            for target in target_ids
        ):
            raise ValueError("target_ids must contain non-empty strings")
        if len(set(target_ids)) != len(target_ids):
            raise ValueError("calibration target_ids must be unique")

        observed = np.asarray(self.observed, dtype=np.float64).copy()
        predictions = np.asarray(self.predictive_draws, dtype=np.float64).copy()
        expected_observed = (len(unit_ids), len(target_ids))
        if observed.shape != expected_observed:
            raise ValueError(f"observed must have shape {expected_observed}")
        expected_prediction_tail = expected_observed
        if (
            predictions.ndim != 3
            or predictions.shape[0] < 2
            or predictions.shape[1:] != expected_prediction_tail
        ):
            raise ValueError(
                "predictive_draws must have shape "
                f"(at least 2, {len(unit_ids)}, {len(target_ids)})"
            )
        if np.any(~np.isfinite(observed)) or np.any(~np.isfinite(predictions)):
            raise ValueError("calibration observations and predictions must be finite")
        observed.setflags(write=False)
        predictions.setflags(write=False)
        object.__setattr__(self, "unit_ids", unit_ids)
        object.__setattr__(self, "target_ids", target_ids)
        object.__setattr__(self, "observed", observed)
        object.__setattr__(self, "predictive_draws", predictions)

    @property
    def unit_count(self) -> int:
        return len(self.unit_ids)

    @property
    def draw_count(self) -> int:
        return int(self.predictive_draws.shape[0])


def _freeze_hashes(source_hashes: Mapping[str, str]) -> Mapping[str, str]:
    copied: dict[str, str] = {}
    for name, digest in source_hashes.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("source hash names must be non-empty strings")
        if not isinstance(digest, str) or not digest.strip():
            raise ValueError("source hash values must be non-empty strings")
        copied[name] = digest
    return MappingProxyType(dict(sorted(copied.items())))


@dataclass(frozen=True)
class CalibrationRequest:
    """Predeclared permission and statistical contract for PI calibration."""

    calibration_id: str
    study_id: str
    study_role: str
    targets: tuple[PredictionTarget, ...]
    interval_level: float = 0.9
    minimum_units: int = 2
    fit_full_covariance: bool = False
    source_hashes: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for label, value in (
            ("calibration_id", self.calibration_id),
            ("study_id", self.study_id),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{label} is required")
        if self.study_role not in {"development", "external"}:
            raise ValueError("study_role must be 'development' or 'external'")
        targets = validate_prediction_targets(self.targets)
        level = float(self.interval_level)
        if not np.isfinite(level) or not 0.0 < level < 1.0:
            raise ValueError("interval_level must be finite and in (0, 1)")
        if (
            isinstance(self.minimum_units, bool)
            or not isinstance(self.minimum_units, (int, np.integer))
            or self.minimum_units < 2
        ):
            raise ValueError("minimum_units must be an integer of at least two")
        if not isinstance(self.fit_full_covariance, bool):
            raise TypeError("fit_full_covariance must be a boolean")
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "interval_level", level)
        object.__setattr__(self, "minimum_units", int(self.minimum_units))
        object.__setattr__(self, "source_hashes", _freeze_hashes(self.source_hashes))

    @property
    def effective_minimum_units(self) -> int:
        # At least one independent calibration unit must be available in the
        # nominal tail.  This also makes interval-resolution limitations
        # explicit rather than silently claiming a 95% calibration from 3 units.
        tail_resolution_minimum = int(
            np.ceil(1.0 / (1.0 - self.interval_level) - 1.0e-12)
        )
        return max(self.minimum_units, tail_resolution_minimum)


@dataclass(frozen=True)
class FrozenCalibrationArtifact:
    """Immutable development-derived interval correction."""

    schema_version: str
    calibration_id: str
    development_study_id: str
    target_ids: tuple[str, ...]
    target_units: tuple[str, ...]
    target_estimands: tuple[str, ...]
    interval_level: float
    unit_count: int
    unit_ids: tuple[str, ...]
    additive_corrections: np.ndarray
    raw_coverage: np.ndarray
    fit_full_covariance: bool
    residual_covariance: np.ndarray | None
    source_hashes: Mapping[str, str]
    input_digest: str
    frozen: bool = True

    def __post_init__(self) -> None:
        if self.schema_version != "our_star.pi_calibration.v0.4":
            raise ValueError("unsupported PI calibration artifact schema")
        if not self.frozen:
            raise ValueError("calibration artifact must be frozen")
        count = len(self.target_ids)
        if count == 0 or len(set(self.target_ids)) != count:
            raise ValueError("artifact target ids must be non-empty and unique")
        if len(self.target_units) != count or len(self.target_estimands) != count:
            raise ValueError("artifact target metadata must align")
        if self.unit_count != len(self.unit_ids) or self.unit_count < 2:
            raise ValueError("artifact calibration-unit count is inconsistent")
        corrections = np.asarray(
            self.additive_corrections, dtype=np.float64
        ).copy()
        coverage = np.asarray(self.raw_coverage, dtype=np.float64).copy()
        if corrections.shape != (count,) or np.any(~np.isfinite(corrections)):
            raise ValueError("additive corrections must be a finite target vector")
        if np.any(corrections < 0.0):
            raise ValueError("additive corrections must be non-negative")
        if (
            coverage.shape != (count,)
            or np.any(~np.isfinite(coverage))
            or np.any((coverage < 0.0) | (coverage > 1.0))
        ):
            raise ValueError("raw coverage must be a target vector in [0, 1]")
        covariance: np.ndarray | None
        if self.fit_full_covariance:
            if self.residual_covariance is None:
                raise ValueError("full covariance fit requires a covariance matrix")
            covariance = np.asarray(
                self.residual_covariance, dtype=np.float64
            ).copy()
            if covariance.shape != (count, count) or np.any(~np.isfinite(covariance)):
                raise ValueError("residual covariance has invalid shape or values")
            if not np.allclose(covariance, covariance.T, rtol=0.0, atol=1e-12):
                raise ValueError("residual covariance must be symmetric")
            try:
                np.linalg.cholesky(covariance)
            except np.linalg.LinAlgError as error:
                raise ValueError(
                    "residual covariance must be strictly positive definite"
                ) from error
            covariance.setflags(write=False)
        else:
            if self.residual_covariance is not None:
                raise ValueError(
                    "residual covariance must be absent when full fitting is disabled"
                )
            covariance = None
        if not isinstance(self.input_digest, str) or len(self.input_digest) != 64:
            raise ValueError("input_digest must be a SHA-256 digest")
        corrections.setflags(write=False)
        coverage.setflags(write=False)
        object.__setattr__(self, "additive_corrections", corrections)
        object.__setattr__(self, "raw_coverage", coverage)
        object.__setattr__(self, "residual_covariance", covariance)
        object.__setattr__(self, "source_hashes", _freeze_hashes(self.source_hashes))

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "calibration_id": self.calibration_id,
            "development_study_id": self.development_study_id,
            "target_ids": list(self.target_ids),
            "target_units": list(self.target_units),
            "target_estimands": list(self.target_estimands),
            "interval_level": self.interval_level,
            "unit_count": self.unit_count,
            "unit_ids": list(self.unit_ids),
            "additive_corrections": self.additive_corrections.tolist(),
            "raw_coverage": self.raw_coverage.tolist(),
            "fit_full_covariance": self.fit_full_covariance,
            "residual_covariance": (
                None
                if self.residual_covariance is None
                else self.residual_covariance.tolist()
            ),
            "source_hashes": dict(self.source_hashes),
            "input_digest": self.input_digest,
            "frozen": True,
        }

    @property
    def artifact_hash(self) -> str:
        return sha256(
            json.dumps(
                self.to_record(), sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True)
class CalibratedIntervals:
    target_ids: tuple[str, ...]
    interval_level: float
    lower: np.ndarray
    median: np.ndarray
    upper: np.ndarray
    calibration_artifact_hash: str

    def __post_init__(self) -> None:
        lower = np.asarray(self.lower, dtype=np.float64).copy()
        median = np.asarray(self.median, dtype=np.float64).copy()
        upper = np.asarray(self.upper, dtype=np.float64).copy()
        if lower.shape != median.shape or upper.shape != median.shape:
            raise ValueError("calibrated interval arrays must have matching shapes")
        if median.ndim < 1 or median.shape[-1] != len(self.target_ids):
            raise ValueError("last interval dimension must match target ids")
        if any(np.any(~np.isfinite(array)) for array in (lower, median, upper)):
            raise ValueError("calibrated intervals must be finite")
        if np.any(lower > median) or np.any(median > upper):
            raise ValueError("calibrated interval ordering is invalid")
        for array in (lower, median, upper):
            array.setflags(write=False)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "median", median)
        object.__setattr__(self, "upper", upper)


CalibrationDataCallback = Callable[[], CalibrationDataset]


def _input_digest(request: CalibrationRequest, data: CalibrationDataset) -> str:
    digest = sha256()
    metadata = {
        "calibration_id": request.calibration_id,
        "study_id": request.study_id,
        "study_role": request.study_role,
        "targets": [
            {
                "target_id": target.target_id,
                "quantity": target.quantity,
                "unit": target.unit,
                "estimand": target.estimand.value,
            }
            for target in request.targets
        ],
        "interval_level": request.interval_level,
        "minimum_units": request.minimum_units,
        "fit_full_covariance": request.fit_full_covariance,
        "source_hashes": dict(request.source_hashes),
        "unit_ids": data.unit_ids,
        "target_ids": data.target_ids,
    }
    digest.update(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    digest.update(data.observed.tobytes(order="C"))
    digest.update(data.predictive_draws.tobytes(order="C"))
    return digest.hexdigest()


def calibrate_prediction_intervals(
    request: CalibrationRequest,
    data_callback: CalibrationDataCallback,
) -> FrozenCalibrationArtifact:
    """Fit additive split-conformal corrections from development units only.

    Permission and target-level checks intentionally run before ``data_callback``
    so an external/locked outcome source is never opened as part of calibration.
    """

    if not isinstance(request, CalibrationRequest):
        raise TypeError("request must be a CalibrationRequest")
    if request.study_role != "development":
        raise PermissionError(
            "prediction-interval calibration is development-only; external "
            "outcomes were rejected before invoking the data callback"
        )
    if request.fit_full_covariance and any(
        target.is_group_summary for target in request.targets
    ):
        raise ValueError(
            "group-summary targets cannot be used to fit a full covariance"
        )
    if not callable(data_callback):
        raise TypeError("data_callback must be callable")

    data = data_callback()
    if not isinstance(data, CalibrationDataset):
        raise TypeError("data_callback must return CalibrationDataset")
    expected_target_ids = tuple(target.target_id for target in request.targets)
    if data.target_ids != expected_target_ids:
        raise ValueError(
            "calibration dataset target order must exactly match the request"
        )
    if data.unit_count < request.effective_minimum_units:
        raise ValueError(
            "insufficient independent calibration units: "
            f"{data.unit_count} < {request.effective_minimum_units}"
        )

    alpha = 1.0 - request.interval_level
    raw_lower = np.quantile(
        data.predictive_draws, alpha / 2.0, axis=0
    )
    raw_upper = np.quantile(
        data.predictive_draws, 1.0 - alpha / 2.0, axis=0
    )
    raw_median = np.median(data.predictive_draws, axis=0)
    scores = np.maximum(
        np.maximum(raw_lower - data.observed, data.observed - raw_upper),
        0.0,
    )
    conformal_quantile = min(
        1.0,
        np.ceil((data.unit_count + 1) * request.interval_level)
        / data.unit_count,
    )
    corrections = np.quantile(
        scores,
        conformal_quantile,
        axis=0,
        method="higher",
    )
    raw_coverage = np.mean(
        (data.observed >= raw_lower) & (data.observed <= raw_upper),
        axis=0,
    )

    residual_covariance: np.ndarray | None = None
    if request.fit_full_covariance:
        target_count = len(request.targets)
        if data.unit_count <= target_count:
            raise ValueError(
                "insufficient independent units to fit full covariance: "
                f"need > {target_count}, got {data.unit_count}"
            )
        residuals = data.observed - raw_median
        covariance = np.atleast_2d(np.cov(residuals, rowvar=False, ddof=1))
        covariance = 0.5 * (covariance + covariance.T)
        try:
            np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as error:
            raise ValueError(
                "development residuals do not identify a positive-definite "
                "full covariance"
            ) from error
        residual_covariance = covariance

    return FrozenCalibrationArtifact(
        schema_version="our_star.pi_calibration.v0.4",
        calibration_id=request.calibration_id,
        development_study_id=request.study_id,
        target_ids=expected_target_ids,
        target_units=tuple(target.unit for target in request.targets),
        target_estimands=tuple(target.estimand.value for target in request.targets),
        interval_level=request.interval_level,
        unit_count=data.unit_count,
        unit_ids=data.unit_ids,
        additive_corrections=corrections,
        raw_coverage=raw_coverage,
        fit_full_covariance=request.fit_full_covariance,
        residual_covariance=residual_covariance,
        source_hashes=request.source_hashes,
        input_digest=_input_digest(request, data),
    )


def apply_prediction_interval_calibration(
    artifact: FrozenCalibrationArtifact,
    predictive_draws: np.ndarray,
    *,
    target_ids: Sequence[str] | None = None,
) -> CalibratedIntervals:
    """Apply a frozen correction without refitting or reading outcomes."""

    if not isinstance(artifact, FrozenCalibrationArtifact):
        raise TypeError("artifact must be a FrozenCalibrationArtifact")
    draws = np.asarray(predictive_draws, dtype=np.float64)
    if draws.ndim < 2 or draws.shape[0] < 2:
        raise ValueError("predictive_draws need at least two draws")
    if draws.shape[-1] != len(artifact.target_ids):
        raise ValueError("predictive draw target dimension does not match artifact")
    if np.any(~np.isfinite(draws)):
        raise ValueError("predictive_draws must be finite")
    supplied_ids = artifact.target_ids if target_ids is None else tuple(target_ids)
    if supplied_ids != artifact.target_ids:
        raise ValueError("target order must exactly match the frozen artifact")
    alpha = 1.0 - artifact.interval_level
    lower = np.quantile(draws, alpha / 2.0, axis=0)
    median = np.median(draws, axis=0)
    upper = np.quantile(draws, 1.0 - alpha / 2.0, axis=0)
    correction_shape = (1,) * (lower.ndim - 1) + (
        len(artifact.target_ids),
    )
    correction = artifact.additive_corrections.reshape(correction_shape)
    return CalibratedIntervals(
        target_ids=artifact.target_ids,
        interval_level=artifact.interval_level,
        lower=lower - correction,
        median=median,
        upper=upper + correction,
        calibration_artifact_hash=artifact.artifact_hash,
    )


__all__ = [
    "CalibratedIntervals",
    "CalibrationDataCallback",
    "CalibrationDataset",
    "CalibrationRequest",
    "FrozenCalibrationArtifact",
    "apply_prediction_interval_calibration",
    "calibrate_prediction_intervals",
]
