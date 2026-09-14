#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from my_star.chemistry import scaffold_key, tautomer_invariant_scaffold_key
from my_star.data import load_manifest, load_observations
from my_star.splits import membership_semantic_digest, save_manifest, validate_split_manifest


def stable_digest(values: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode("utf-8")).hexdigest()


def manifest_digest(manifest: dict) -> str:
    """Return a stable digest of the complete outer-split assignment."""
    payload = json.dumps(
        manifest,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def resolve_inner_strategy(outer_manifest: dict, requested: str | None) -> str:
    """Use the requested scaffold definition, or inherit a scaffold outer split."""
    if requested is not None:
        strategy = requested
    else:
        outer_strategy = str(outer_manifest.get("strategy", ""))
        strategy = (
            outer_strategy
            if outer_strategy in {"scaffold", "tautomer_scaffold"}
            else "scaffold"
        )
    if strategy not in {"scaffold", "tautomer_scaffold"}:
        raise ValueError(f"Unsupported inner-fold strategy: {strategy}")
    return strategy


def build_inner_folds(
    observations: pd.DataFrame,
    outer_manifest: dict,
    folds: int,
    seed: int,
    strategy: str | None = None,
) -> list[dict]:
    if folds < 2:
        raise ValueError("folds must be at least two")
    outer_train = set(outer_manifest["splits"]["train"])
    outer_holdout = set(outer_manifest["splits"]["validation"]) | set(
        outer_manifest["splits"]["test"]
    )
    strategy = resolve_inner_strategy(outer_manifest, strategy)
    group_column = (
        "tautomer_invariant_scaffold"
        if strategy == "tautomer_scaffold"
        else "scaffold"
    )
    use_precomputed = group_column in observations.columns
    structure_columns = ["compound_id", "canonical_smiles"]
    if use_precomputed:
        structure_columns.append(group_column)
    structures = observations.loc[
        observations["compound_id"].isin(outer_train),
        structure_columns,
    ].drop_duplicates()
    if structures["compound_id"].duplicated().any():
        raise ValueError("A compound_id maps to multiple structures or scaffold keys")
    if use_precomputed and (
        structures[group_column].isna().any()
        or structures[group_column].astype(str).str.strip().eq("").any()
    ):
        raise ValueError(f"Precomputed {group_column} contains missing values")
    weights = observations.groupby("compound_id").size().astype(int).to_dict()
    scaffold_groups: dict[str, list[str]] = {}
    for row in structures.itertuples():
        if use_precomputed:
            group_key = str(getattr(row, group_column))
        elif strategy == "tautomer_scaffold":
            group_key = tautomer_invariant_scaffold_key(row.canonical_smiles)
        else:
            group_key = scaffold_key(row.canonical_smiles)
        scaffold_groups.setdefault(group_key, []).append(str(row.compound_id))

    rng = random.Random(seed)
    groups = list(scaffold_groups.values())
    rng.shuffle(groups)
    groups.sort(key=lambda group: sum(weights[item] for item in group), reverse=True)
    assigned = [[] for _ in range(folds)]
    fold_weights = [0 for _ in range(folds)]
    for group in groups:
        target = min(range(folds), key=fold_weights.__getitem__)
        assigned[target].extend(group)
        fold_weights[target] += sum(weights[item] for item in group)

    manifests = []
    for fold_index, validation_ids in enumerate(assigned):
        validation = set(validation_ids)
        train = outer_train - validation
        manifest = {
            "version": 2,
            "strategy": strategy,
            "seed": seed,
            "inner_fold": fold_index,
            "outer_manifest_sha256": manifest_digest(outer_manifest),
            "group_key_source": group_column if use_precomputed else "computed",
            "dataset_compound_digest": stable_digest(
                observations["compound_id"].astype(str).unique().tolist()
            ),
            "splits": {
                "train": sorted(train),
                "validation": sorted(validation),
                "test": sorted(outer_holdout),
            },
        }
        manifest["membership_semantic_digest"] = membership_semantic_digest(manifest)
        validate_split_manifest(observations, manifest, verify_digests=True)
        manifests.append(manifest)
    return manifests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "data/processed/sciplex3_matched_pseudobulk.parquet",
    )
    parser.add_argument(
        "--outer-split",
        type=Path,
        default=ROOT / "data/splits/sciplex3_scaffold_split.json",
    )
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--prefix", default="sciplex3_inner_cv")
    parser.add_argument(
        "--strategy",
        choices=("scaffold", "tautomer_scaffold"),
        default=None,
        help="Inner grouping rule; by default inherit a scaffold-based outer manifest.",
    )
    args = parser.parse_args()

    observations = load_observations(args.dataset)
    outer_manifest = load_manifest(args.outer_split)
    manifests = build_inner_folds(
        observations,
        outer_manifest,
        args.folds,
        args.seed,
        strategy=args.strategy,
    )
    for index, manifest in enumerate(manifests):
        output = ROOT / "data/splits" / f"{args.prefix}_fold{index}.json"
        save_manifest(manifest, output)
        sizes = {
            name: int(observations.compound_id.isin(ids).sum())
            for name, ids in manifest["splits"].items()
        }
        print(f"{output}: {sizes}")


if __name__ == "__main__":
    main()
