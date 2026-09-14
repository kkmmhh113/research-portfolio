"""Expression-free, exact-cell provenance for training-only control statistics.

This module does not fit statistics. It specifies and verifies the *only* row
positions a later gene-panel fitter may consume under a given membership.
Shared controls are allowed when genuinely associated with training conditions;
held-out-only controls and all treated cells are forbidden for this fit scope.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from .splits import membership_semantic_digest


class FitScopeError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FitScopeError(message)


def index_digest(values: list[int] | np.ndarray) -> str:
    array = np.asarray(values)
    require(array.ndim == 1 and (not array.size or array.dtype.kind in "iu"), "Row indices must be integers")
    require(not array.size or (array >= 0).all(), "Negative row indices are forbidden")
    return hashlib.sha256(array.astype("<i8").tobytes()).hexdigest()


def exact_membership(conditions: pd.DataFrame, manifest: dict) -> dict[str, str]:
    require(set(manifest["splits"]) == {"train", "validation", "test"}, "Expected three partition roles")
    assigned = {}
    for role, keys in manifest["splits"].items():
        require(isinstance(keys, list) and all(isinstance(key, str) for key in keys), "Membership IDs must be strings")
        for key in keys:
            require(key not in assigned, "Duplicate or overlapping membership")
            assigned[key] = role
    require(set(assigned) == set(conditions.compound_id), "Membership must cover eligible compounds exactly")
    require(bool(manifest["splits"]["train"]), "Empty training membership")
    require(not conditions.sample_id.duplicated().any(), "Duplicate condition identity")
    return assigned


def control_fit_scope(obs: pd.DataFrame, group_ids: np.ndarray, groups: pd.DataFrame,
                      conditions: pd.DataFrame, manifest: dict) -> dict:
    """Return exact train-associated group/cell unions, without expression input."""
    require(group_ids.ndim == 1 and len(group_ids) == len(obs) and group_ids.dtype.kind in "iu",
            "Group IDs must be an integer vector aligned to raw observation order")
    require((group_ids >= -1).all() and (group_ids < len(groups)).all(), "Unknown metadata group ID")
    require(groups.index.tolist() == list(range(len(groups))), "Group index must be contiguous and positional")
    counts = np.bincount(group_ids[group_ids >= 0], minlength=len(groups))
    require(np.array_equal(counts, groups.cell_count.to_numpy()) and (counts > 0).all(),
            "Declared group counts differ from actual source-cell membership")
    membership = exact_membership(conditions, manifest)
    used = {role: set() for role in ("train", "validation", "test")}
    condition_counts = {role: 0 for role in used}
    links = []
    for condition in conditions.sort_values("sample_id").itertuples():
        controls = condition.control_group_indices
        require(isinstance(controls, list) and controls and all(type(i) is int for i in controls)
                and controls == sorted(set(controls)), "Control group references must be sorted unique integer lists")
        role = membership[condition.compound_id]
        condition_counts[role] += 1
        for index in controls:
            require(0 <= index < len(groups), "Condition points to an unknown control group")
            control = groups.loc[index]
            require(control["kind"] == "control" and control["perturbation"] == "control",
                    "Condition references a treated group as a control")
            require(control["cell_line"] == condition.cell_line
                    and float(control["duration_hours"]) == float(condition.duration_hours),
                    "Matched control has a different cell/time context")
        used[role].update(controls)
        links.append([condition.sample_id, condition.compound_id, role, controls])

    training = used["train"]
    heldout = used["validation"] | used["test"]
    require(bool(training), "No training-associated control groups")
    train_rows = np.flatnonzero(np.isin(group_ids, sorted(training)))
    heldout_only = heldout - training
    heldout_only_rows = np.flatnonzero(np.isin(group_ids, sorted(heldout_only)))
    all_used = training | heldout
    controls_all = set(groups.index[groups["kind"] == "control"].tolist())
    require(all_used <= controls_all, "Non-control group entered a fitting scope")
    # Independently inspect the selected obs metadata; group labels alone are
    # not sufficient proof that a selected row is actually a control cell.
    for index in sorted(all_used):
        positions = np.flatnonzero(group_ids == index)
        control = groups.loc[index]
        actual = obs.iloc[positions]
        require(actual.perturbation.eq("control").all() and (actual.library_size > 0).all(),
                "Selected group contains treated or invalid-library cells")
        for field in ("cell_line", "duration_hours", "plate", "replicate"):
            require(actual[field].eq(control[field]).all(), "Control cell metadata differs from its group")
    require(not np.intersect1d(train_rows, heldout_only_rows).size, "Held-out-only cells entered panel fit")

    def rows_for(role: str) -> np.ndarray:
        return np.flatnonzero(np.isin(group_ids, sorted(used[role])))

    return {"schema_version": 1, "membership_semantic_sha256": membership_semantic_digest(manifest),
        "condition_control_link_sha256": hashlib.sha256(json.dumps(links, separators=(",", ":")).encode()).hexdigest(),
        "source_observation_count": len(obs),
        "role_counts": {role: {"compounds": len(manifest["splits"][role]),
                               "eligible_conditions": condition_counts[role],
                               "matched_control_groups": len(used[role]), "matched_control_cells": len(rows_for(role))}
                        for role in used},
        "fit_control_group_indices": sorted(training), "fit_control_row_positions": train_rows.tolist(),
        "fit_control_rows_sha256": index_digest(train_rows),
        "heldout_only_control_group_indices": sorted(heldout_only),
        "heldout_only_control_row_positions": heldout_only_rows.tolist(),
        "heldout_only_control_rows_sha256": index_digest(heldout_only_rows),
        "shared_train_heldout_control_groups": sorted(training & heldout),
        "control_groups_not_associated_with_any_retained_condition": sorted(controls_all - all_used),
        "selected_treated_cells": 0, "selected_heldout_only_control_cells": 0,
        "fit_performed": False, "training_protocol_approved": False,
        "scope": "gene-panel control-only variance/detection statistics; not response bases, scalers, priors or calibration"}


def require_exact_fit_rows(scope: dict, proposed_rows: np.ndarray) -> None:
    """A later fitter must supply this exact, ordered train-only row vector."""
    require(index_digest(scope["fit_control_row_positions"]) == scope["fit_control_rows_sha256"],
            "Stored fit scope does not reproduce")
    index_digest(proposed_rows)
    require(np.array_equal(proposed_rows, np.asarray(scope["fit_control_row_positions"], dtype=np.int64)),
            "Proposed panel rows differ from the exact training-control union")


def compare_fit_scopes(outer: dict, inner: list[dict]) -> dict:
    require(bool(inner), "No inner-fold scope to compare")
    # Cardinalities or even hashes alone do not substitute for exact equality.
    vectors = [item["fit_control_row_positions"] for item in inner]
    for scope in [outer, *inner]:
        require_exact_fit_rows(scope, np.asarray(scope["fit_control_row_positions"], dtype=np.int64))
    same_inner = all(vector == vectors[0] for vector in vectors)
    same_outer = all(vector == outer["fit_control_row_positions"] for vector in vectors)
    return {"all_inner_training_control_unions_equal": same_inner,
            "all_inner_unions_equal_outer_train_union": same_outer,
            "single_shared_panel_fit_scope_possible_for_this_candidate": same_inner and same_outer,
            "required_action": "bind the exact common row vector before any panel fit" if same_inner and same_outer
                else "fit and bind separate train-only panels for differing folds; do not reuse the outer-only panel",
            "scope_approval": "metadata feasibility only; no gene statistics, frozen split or protocol approval"}
