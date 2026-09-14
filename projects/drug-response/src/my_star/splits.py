from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import pandas as pd

from .chemistry import scaffold_key, tautomer_invariant_scaffold_key


SPLIT_NAMES = ("train", "validation", "test")


def _precomputed_group_column(strategy: str) -> str | None:
    if strategy == "scaffold":
        return "scaffold"
    if strategy == "tautomer_scaffold":
        return "tautomer_invariant_scaffold"
    return None


def _stable_digest(values: list[str]) -> str:
    payload = "\n".join(sorted(values)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def membership_semantic_digest(manifest: dict) -> str:
    """Bind each compound ID to its partition role, independent of JSON bytes."""
    rows = [
        f"{split}\t{compound_id}"
        for split in SPLIT_NAMES
        for compound_id in sorted(str(value) for value in manifest["splits"][split])
    ]
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def _assign_groups(
    group_to_compounds: dict[str, list[str]],
    compound_weights: dict[str, int],
    fractions: tuple[float, float, float],
    seed: int,
) -> dict[str, list[str]]:
    rng = random.Random(seed)
    groups = list(group_to_compounds.items())
    rng.shuffle(groups)
    groups.sort(
        key=lambda item: sum(compound_weights[c] for c in item[1]),
        reverse=True,
    )
    total = sum(compound_weights.values())
    targets = dict(zip(SPLIT_NAMES, [fraction * total for fraction in fractions]))
    assigned = {name: [] for name in SPLIT_NAMES}
    counts = {name: 0 for name in SPLIT_NAMES}

    for _, compounds in groups:
        weight = sum(compound_weights[c] for c in compounds)
        split = min(
            SPLIT_NAMES,
            key=lambda name: (counts[name] + weight) / max(targets[name], 1.0),
        )
        assigned[split].extend(compounds)
        counts[split] += weight
    return assigned


def create_split_manifest(
    observations: pd.DataFrame,
    strategy: str,
    fractions: tuple[float, float, float],
    seed: int,
) -> dict:
    if abs(sum(fractions) - 1.0) > 1e-8:
        raise ValueError("Split fractions must sum to 1")
    group_column = _precomputed_group_column(strategy)
    use_precomputed = group_column is not None and group_column in observations.columns
    compound_columns = ["compound_id", "canonical_smiles"]
    if use_precomputed:
        compound_columns.append(group_column)
    compounds = observations[compound_columns].drop_duplicates()
    if compounds["compound_id"].duplicated().any():
        raise ValueError("A compound_id maps to multiple structures")
    if use_precomputed and (
        compounds[group_column].isna().any()
        or compounds[group_column].astype(str).str.strip().eq("").any()
    ):
        raise ValueError(f"Precomputed {group_column} contains missing values")
    weights = observations.groupby("compound_id").size().astype(int).to_dict()

    if strategy == "compound":
        group_to_compounds = {row.compound_id: [row.compound_id] for row in compounds.itertuples()}
    elif strategy in {"scaffold", "tautomer_scaffold"}:
        key_function = (
            scaffold_key
            if strategy == "scaffold"
            else tautomer_invariant_scaffold_key
        )
        group_to_compounds: dict[str, list[str]] = {}
        for row in compounds.itertuples():
            group_key = (
                str(getattr(row, group_column))
                if use_precomputed
                else key_function(row.canonical_smiles)
            )
            group_to_compounds.setdefault(group_key, []).append(row.compound_id)
    else:
        raise ValueError(f"Unknown split strategy: {strategy}")

    assigned = _assign_groups(group_to_compounds, weights, fractions, seed)
    manifest = {
        "version": 2,
        "strategy": strategy,
        "seed": seed,
        "fractions": dict(zip(SPLIT_NAMES, fractions)),
        "dataset_compound_digest": _stable_digest(compounds["compound_id"].tolist()),
        "group_key_source": group_column if use_precomputed else "computed",
        "splits": {name: sorted(values) for name, values in assigned.items()},
    }
    manifest["membership_semantic_digest"] = membership_semantic_digest(manifest)
    validate_split_manifest(observations, manifest, verify_digests=True)
    return manifest


def validate_split_manifest(
    observations: pd.DataFrame,
    manifest: dict,
    *,
    verify_digests: bool = False,
) -> None:
    for name in SPLIT_NAMES:
        values = manifest["splits"][name]
        if len(values) != len(set(str(value) for value in values)):
            raise ValueError(f"Duplicate compound IDs within {name} split")
    split_sets = {name: set(manifest["splits"][name]) for name in SPLIT_NAMES}
    for left_index, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[left_index + 1 :]:
            overlap = split_sets[left].intersection(split_sets[right])
            if overlap:
                raise ValueError(f"Compound leakage between {left} and {right}: {sorted(overlap)[:5]}")
    observed = set(observations["compound_id"].unique())
    assigned = set.union(*split_sets.values())
    if observed != assigned:
        raise ValueError("Split manifest does not cover the dataset exactly")

    expected_membership_digest = manifest.get("membership_semantic_digest")
    if expected_membership_digest is not None:
        actual_membership_digest = membership_semantic_digest(manifest)
        if expected_membership_digest != actual_membership_digest:
            raise ValueError("Split membership semantic digest does not reproduce")
    if verify_digests:
        expected_compound_digest = manifest.get("dataset_compound_digest")
        if expected_compound_digest is None:
            raise ValueError("Split manifest lacks dataset_compound_digest")
        actual_compound_digest = _stable_digest([str(value) for value in observed])
        if expected_compound_digest != actual_compound_digest:
            raise ValueError("Split dataset compound digest does not reproduce")
        if expected_membership_digest is None:
            raise ValueError("Split manifest lacks membership_semantic_digest")

    if manifest["strategy"] in {"scaffold", "tautomer_scaffold"}:
        group_column = _precomputed_group_column(manifest["strategy"])
        use_precomputed = (
            group_column is not None and group_column in observations.columns
        )
        key_function = (
            scaffold_key
            if manifest["strategy"] == "scaffold"
            else tautomer_invariant_scaffold_key
        )
        compound_columns = ["compound_id", "canonical_smiles"]
        if use_precomputed:
            compound_columns.append(group_column)
        smiles = observations[compound_columns].drop_duplicates()
        scaffold_sets = {
            name: {
                (
                    str(getattr(row, group_column))
                    if use_precomputed
                    else key_function(row.canonical_smiles)
                )
                for row in smiles[smiles.compound_id.isin(compounds)].itertuples()
            }
            for name, compounds in split_sets.items()
        }
        for left_index, left in enumerate(SPLIT_NAMES):
            for right in SPLIT_NAMES[left_index + 1 :]:
                if scaffold_sets[left].intersection(scaffold_sets[right]):
                    raise ValueError(f"Scaffold leakage between {left} and {right}")


def save_manifest(manifest: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
