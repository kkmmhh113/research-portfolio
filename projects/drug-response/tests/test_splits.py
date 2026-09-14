import pandas as pd
import pytest

from my_star.chemistry import tautomer_invariant_scaffold_key
from my_star.splits import (
    create_split_manifest,
    membership_semantic_digest,
    validate_split_manifest,
)
import my_star.splits as splits


def test_scaffold_split_has_no_compound_or_scaffold_leakage() -> None:
    compounds = {
        "a": "c1ccccc1O",
        "b": "c1ccccc1N",
        "c": "C1CCCCC1",
        "d": "CCCC",
        "e": "CCO",
        "f": "CCN",
    }
    rows = []
    for compound, smiles in compounds.items():
        for replicate in range(2):
            rows.append({"compound_id": compound, "canonical_smiles": smiles, "replicate": replicate})
    frame = pd.DataFrame(rows)
    manifest = create_split_manifest(frame, "scaffold", (0.5, 0.25, 0.25), seed=7)
    validate_split_manifest(frame, manifest)
    split_sets = [set(manifest["splits"][name]) for name in ("train", "validation", "test")]
    assert not split_sets[0].intersection(split_sets[1])
    assert not split_sets[0].intersection(split_sets[2])


def test_tautomer_scaffold_split_keeps_equivalent_scaffolds_together() -> None:
    tautomer_a = "CCCCOc1c(C(=O)c2c(F)cc(C)cc2F)cnc2[nH]ncc12"
    tautomer_b = "CCCCOc1c(C(=O)c2c(F)cc(C)cc2F)cnc2n[nH]cc12"
    assert tautomer_invariant_scaffold_key(tautomer_a) == (
        tautomer_invariant_scaffold_key(tautomer_b)
    )
    frame = pd.DataFrame(
        {
            "compound_id": ["a", "b", "c", "d", "e", "f"],
            "canonical_smiles": [tautomer_a, tautomer_b, "c1ccccc1", "C1CCCCC1", "CCCC", "CCO"],
        }
    )
    manifest = create_split_manifest(
        frame, "tautomer_scaffold", (0.5, 0.25, 0.25), seed=19
    )
    validate_split_manifest(frame, manifest)
    membership = {
        compound_id: split
        for split, compound_ids in manifest["splits"].items()
        for compound_id in compound_ids
    }
    assert membership["a"] == membership["b"]


def test_tautomer_split_reuses_audited_precomputed_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame(
        {
            "compound_id": ["a", "b", "c", "d"],
            "canonical_smiles": ["c1ccccc1O", "c1ccccc1N", "C1CCCCC1", "CCO"],
            "tautomer_invariant_scaffold": ["aromatic", "aromatic", "ring", "acyclic"],
        }
    )

    def fail_if_recomputed(_: str) -> str:
        raise AssertionError("precomputed scaffold key was recomputed")

    monkeypatch.setattr(splits, "tautomer_invariant_scaffold_key", fail_if_recomputed)
    manifest = create_split_manifest(
        frame, "tautomer_scaffold", (0.5, 0.25, 0.25), seed=23
    )

    assert manifest["group_key_source"] == "tautomer_invariant_scaffold"
    validate_split_manifest(frame, manifest)


def test_precomputed_scaffold_keys_must_be_complete() -> None:
    frame = pd.DataFrame(
        {
            "compound_id": ["a", "b", "c"],
            "canonical_smiles": ["c1ccccc1", "C1CCCCC1", "CCO"],
            "tautomer_invariant_scaffold": ["aromatic", None, "acyclic"],
        }
    )

    with pytest.raises(ValueError, match="contains missing values"):
        create_split_manifest(
            frame, "tautomer_scaffold", (0.5, 0.25, 0.25), seed=29
        )


def test_split_validator_rejects_duplicate_ids_within_a_role() -> None:
    frame = pd.DataFrame(
        {
            "compound_id": ["a", "b", "c"],
            "canonical_smiles": ["CCO", "CCN", "CCC"],
        }
    )
    manifest = {
        "strategy": "compound",
        "splits": {"train": ["a", "a"], "validation": ["b"], "test": ["c"]},
    }

    with pytest.raises(ValueError, match="Duplicate compound IDs"):
        validate_split_manifest(frame, manifest)


def test_membership_digest_binds_partition_roles() -> None:
    first = {
        "splits": {"train": ["a"], "validation": ["b"], "test": ["c"]}
    }
    second = {
        "splits": {"train": ["b"], "validation": ["a"], "test": ["c"]}
    }

    assert membership_semantic_digest(first) != membership_semantic_digest(second)


def test_strict_digest_validation_rejects_tampered_compound_digest() -> None:
    frame = pd.DataFrame(
        {
            "compound_id": ["a", "b", "c"],
            "canonical_smiles": ["CCO", "CCN", "CCC"],
        }
    )
    manifest = create_split_manifest(frame, "compound", (0.5, 0.25, 0.25), seed=31)
    manifest["dataset_compound_digest"] = "0" * 64

    with pytest.raises(ValueError, match="dataset compound digest"):
        validate_split_manifest(frame, manifest, verify_digests=True)
