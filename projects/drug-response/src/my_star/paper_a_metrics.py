"""Prospective paired compound-macro evaluation, with no data access or fitting.

This is a numerical component, not a frozen evaluator or an access approval.
The caller must establish the authorized population, gene axis, target mask,
fixed ensemble/calibration and prediction provenance before using real data.
Historical metric implementations and result artifacts are left unchanged.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import hashlib
import json
import math

import numpy as np
import pandas as pd


META = ("sample_id", "compound_id", "fold_id", "cell_line", "dose_nM", "duration_hours")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class Reference:
    metadata: pd.DataFrame
    gene_ids: tuple[str, ...]
    targets: np.ndarray
    mask: np.ndarray


@dataclass(frozen=True)
class Predictions:
    metadata: pd.DataFrame
    gene_ids: tuple[str, ...]
    values: np.ndarray


@dataclass(frozen=True)
class PearsonPolicy:
    # These choices are explicit: no hidden exclusion or undefined-to-zero rule.
    degenerate: str
    denominator_floor: float

    def validate(self) -> None:
        require(self.degenerate in {"error", "zero_with_flag"}, "Unknown degenerate Pearson policy")
        require(type(self.denominator_floor) in {int, float}
                and np.isfinite(self.denominator_floor) and self.denominator_floor >= 0,
                "Invalid Pearson denominator floor")


def _metadata(frame: pd.DataFrame) -> pd.DataFrame:
    require(isinstance(frame, pd.DataFrame) and set(META) <= set(frame), "Missing condition metadata")
    require(not frame.empty and frame.columns.is_unique, "Empty or repeated metadata columns")
    result = frame[list(META)].copy()
    for name in ("sample_id", "compound_id", "cell_line"):
        require(result[name].map(lambda v: isinstance(v, str) and bool(v.strip()) and v == v.strip()).all(),
                f"Invalid {name}")
    require(not result.sample_id.duplicated().any(), "Duplicate sample IDs")
    require(result.fold_id.map(lambda v: isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_))
                               and v >= 0).all(), "Fold IDs must be non-negative integers")
    for name in ("dose_nM", "duration_hours"):
        require(result[name].dtype.kind in "fiu", f"Invalid {name}")
        result[name] = result[name].astype(np.float64)
        require(np.isfinite(result[name]).all() and (result[name] >= 0).all(), f"Invalid {name}")
    result["fold_id"] = result.fold_id.astype(np.int64)
    require((result.groupby("compound_id").fold_id.nunique() == 1).all(),
            "A compound occurs in multiple evaluation folds")
    require(not result.duplicated(["compound_id", "cell_line", "dose_nM", "duration_hours"]).any(),
            "Duplicate biological conditions")
    return result


def _numeric(value, shape: tuple[int, int], name: str) -> np.ndarray:
    array = np.asarray(value)
    require(array.shape == shape and array.dtype.kind in "fiu", f"Invalid {name} shape/type")
    return array.astype(np.float64, copy=True)


def _reference(reference: Reference) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    metadata = _metadata(reference.metadata)
    genes = reference.gene_ids
    require(isinstance(genes, tuple) and len(genes) >= 2
            and all(isinstance(g, str) and g and g == g.strip() for g in genes)
            and len(set(genes)) == len(genes), "Invalid or duplicate gene axis")
    shape = (len(metadata), len(genes))
    target = _numeric(reference.targets, shape, "target")
    mask = np.asarray(reference.mask)
    require(mask.shape == shape and mask.dtype == np.dtype(bool), "Mask must be an exact boolean matrix")
    require((mask.sum(axis=1) >= 2).all(), "Every condition needs at least two observed genes; no silent row drop")
    require(np.isfinite(target[mask]).all(), "Non-finite observed targets")
    order = np.argsort(metadata.sample_id.to_numpy(), kind="stable")
    return metadata.iloc[order].reset_index(drop=True), target[order], mask[order].copy()


def _aligned_values(reference: Reference, metadata: pd.DataFrame, mask: np.ndarray,
                    prediction: Predictions) -> np.ndarray:
    require(isinstance(prediction.gene_ids, tuple) and prediction.gene_ids == reference.gene_ids,
            "Prediction gene identity/order differs")
    own = _metadata(prediction.metadata)
    require(set(own.sample_id) == set(metadata.sample_id), "Prediction population differs")
    values = _numeric(prediction.values, (len(own), len(reference.gene_ids)), "prediction")
    order = np.argsort(own.sample_id.to_numpy(), kind="stable")
    require(own.iloc[order].reset_index(drop=True).equals(metadata), "Prediction condition metadata differs")
    values = values[order]
    require(np.isfinite(values[mask]).all(), "Non-finite observed predictions")
    return values


def equal_seed_ensemble(reference: Reference, members: Mapping[int, Predictions], *,
                        required_seeds: tuple[int, ...]) -> Predictions:
    """Average aligned predictions first; never average scores or fit weights."""
    require(isinstance(required_seeds, tuple) and len(required_seeds) >= 1
            and all(type(s) is int and s >= 0 for s in required_seeds)
            and len(set(required_seeds)) == len(required_seeds), "Invalid required seeds")
    require(all(type(s) is int for s in members) and set(members) == set(required_seeds),
            "Missing/extra seed member")
    metadata, _, mask = _reference(reference)
    arrays = [_aligned_values(reference, metadata, mask, members[s]) for s in sorted(required_seeds)]
    values = np.full(mask.shape, np.nan, dtype=np.float64)
    with np.errstate(over="raise", invalid="raise"):
        values[mask] = np.mean(np.stack([array[mask] for array in arrays]), axis=0)
    require(np.isfinite(values[mask]).all(), "Non-finite ensemble")
    return Predictions(metadata, reference.gene_ids, values)


def _pearson(target: np.ndarray, prediction: np.ndarray, policy: PearsonPolicy) -> tuple[float, bool]:
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        a, b = target - target.mean(), prediction - prediction.mean()
        denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
        require(np.isfinite(denominator), "Non-finite Pearson denominator")
        degenerate = denominator <= policy.denominator_floor
        if degenerate:
            require(policy.degenerate == "zero_with_flag", "Undefined/degenerate Pearson; explicit policy required")
            return 0.0, True
        score = float(np.dot(a / np.linalg.norm(a), b / np.linalg.norm(b)))
    require(np.isfinite(score), "Non-finite Pearson")
    return float(np.clip(score, -1.0, 1.0)), False


def compound_macro_bootstrap(effects: Mapping[str, float], *, iterations: int = 10_000,
                             seed: int = 2026) -> dict:
    """Resample paired compound means, not row sums divided by row counts."""
    require(len(effects) >= 2 and all(isinstance(k, str) and k.strip() for k in effects),
            "Uncertainty requires at least two identified compound effects")
    require(all(isinstance(v, (int, float, np.integer, np.floating))
                and not isinstance(v, (bool, np.bool_)) for v in effects.values()), "Non-numeric compound effect")
    require(type(iterations) is int and iterations >= 2 and type(seed) is int and seed >= 0,
            "Invalid bootstrap settings")
    values = np.asarray([effects[k] for k in sorted(effects)], dtype=np.float64)
    require(np.isfinite(values).all(), "Non-finite compound effect")
    generator = np.random.Generator(np.random.PCG64(seed))
    estimates = np.empty(iterations, dtype=np.float64)
    for start in range(0, iterations, 500):
        size = min(500, iterations - start)
        indices = generator.integers(0, len(values), size=(size, len(values)))
        estimates[start:start + size] = values[indices].mean(axis=1)
    require(np.isfinite(estimates).all(), "Non-finite bootstrap estimate")
    lower, upper = np.quantile(estimates, [0.025, 0.975], method="linear")
    return {"estimand": "unweighted_mean_of_paired_compound_mean_sample_Pearson_gains",
            "observed_mean_difference": float(math.fsum(values) / len(values)),
            "bootstrap_mean_difference": float(estimates.mean()),
            "ci95_lower": float(lower), "ci95_upper": float(upper),
            "bootstrap_fraction_above_zero": float((estimates > 0).mean()),
            "bootstrap_fraction_equal_zero": float((estimates == 0).mean()),
            "compound_blocks": len(values), "iterations": iterations, "seed": seed,
            "rng": "PCG64", "interval_method": "percentile_linear_quantiles",
            "degenerate_bootstrap_distribution": bool(np.ptp(estimates) == 0),
            "conditional_on_fixed_predictions": True,
            "resamples_training_or_seeds_or_scaffolds": False,
            "positive_fraction_is_p_value_or_posterior_probability": False}


def evaluate_paired(reference: Reference, candidate: Predictions, baseline: Predictions, *,
                    allowed_folds: tuple[int, ...], pearson_policy: PearsonPolicy,
                    iterations: int = 10_000, bootstrap_seed: int = 2026) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Return auditable row/compound tables and the specified primary estimate.

    No calibration, model selection, promotion decision, masking policy search,
    dataset I/O, or split/exposure authorization occurs in this function.
    """
    pearson_policy.validate()
    require(isinstance(allowed_folds, tuple) and allowed_folds
            and all(type(f) is int and f >= 0 for f in allowed_folds)
            and len(set(allowed_folds)) == len(allowed_folds), "Invalid declared evaluation folds")
    metadata, targets, mask = _reference(reference)
    require(set(metadata.fold_id) == set(allowed_folds), "Declared fold population is missing or unexpected")
    predictions = {"candidate": _aligned_values(reference, metadata, mask, candidate),
                   "baseline": _aligned_values(reference, metadata, mask, baseline)}
    rows = metadata.copy()
    rows["observed_gene_count"] = mask.sum(axis=1)
    for name, values in predictions.items():
        scores = [_pearson(t[m], p[m], pearson_policy) for t, p, m in zip(targets, values, mask)]
        rows[f"{name}_pearson_score"] = [value for value, _ in scores]
        rows[f"{name}_degenerate"] = [flag for _, flag in scores]
    rows["paired_gain"] = rows.candidate_pearson_score - rows.baseline_pearson_score
    compound_rows = []
    for compound, group in rows.groupby("compound_id", sort=True):
        compound_rows.append({"compound_id": compound, "fold_id": int(group.fold_id.iloc[0]),
                              "conditions": len(group),
                              **{name: math.fsum(group[name]) / len(group) for name in
                                 ("candidate_pearson_score", "baseline_pearson_score", "paired_gain")}})
    compounds = pd.DataFrame(compound_rows)
    primary = compound_macro_bootstrap(dict(zip(compounds.compound_id, compounds.paired_gain)),
                                       iterations=iterations, seed=bootstrap_seed)
    gene_axis_sha = hashlib.sha256(json.dumps(reference.gene_ids, ensure_ascii=False,
                                             separators=(",", ":")).encode()).hexdigest()
    return {"schema_version": 1, "status": "computed_from_supplied_predictions_not_a_frozen_evaluator",
            "primary": primary,
            "secondary_row_micro_paired_gain": float(math.fsum(rows.paired_gain) / len(rows)),
            "compound_macro_candidate_score": float(compounds.candidate_pearson_score.mean()),
            "compound_macro_baseline_score": float(compounds.baseline_pearson_score.mean()),
            "per_fold": {str(f): {"compounds": len(g), "conditions": int(g.conditions.sum()),
                                  "compound_macro_paired_gain": float(g.paired_gain.mean())}
                         for f, g in compounds.groupby("fold_id", sort=True)},
            "conditions": len(rows), "compounds": len(compounds),
            "gene_axis_sha256": gene_axis_sha, "gene_count": len(reference.gene_ids),
            "pearson_policy": asdict(pearson_policy),
            "degenerate_rows": {name: int(rows[f"{name}_degenerate"].sum()) for name in predictions},
            "zero_fallback_is_mathematical_Pearson": False,
            "calibration_fit_performed": False, "promotion_or_access_approval_conferred": False,
            "split_and_population_provenance_verified_by_this_kernel": False}, rows, compounds
