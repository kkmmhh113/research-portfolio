import numpy as np

from my_star.chemistry import (
    molecular_descriptors,
    molecular_graph,
    morgan_fingerprint,
    scaffold_key,
    standardize_smiles,
    tautomer_invariant_scaffold_key,
)


def test_structure_features_are_deterministic() -> None:
    structure = standardize_smiles("CC(=O)OC1=CC=CC=C1C(=O)O")
    assert structure is not None
    first = morgan_fingerprint(structure.canonical_smiles)
    second = morgan_fingerprint(structure.canonical_smiles)
    assert first.shape == (2048,)
    assert np.array_equal(first, second)
    assert molecular_descriptors(structure.canonical_smiles).shape == (8,)
    extended = molecular_descriptors(structure.canonical_smiles, "extended")
    assert extended.shape == (23,)
    assert np.isfinite(extended).all()


def test_acyclic_scaffolds_remain_compound_specific() -> None:
    assert scaffold_key("CCCC") != scaffold_key("CCO")


def test_count_fingerprint_is_deterministic_and_nonbinary() -> None:
    first = morgan_fingerprint("CCCCCC", use_counts=True)
    second = morgan_fingerprint("CCCCCC", use_counts=True)
    assert np.array_equal(first, second)
    assert first.max() > np.log(2.0)


def test_chiral_fingerprint_distinguishes_enantiomers() -> None:
    left = morgan_fingerprint("N[C@@H](C)C(=O)O", include_chirality=True)
    right = morgan_fingerprint("N[C@H](C)C(=O)O", include_chirality=True)
    assert not np.array_equal(left, right)


def test_tautomer_invariant_scaffold_key_collapses_equivalent_scaffolds() -> None:
    first = "CCCCOc1c(C(=O)c2c(F)cc(C)cc2F)cnc2[nH]ncc12"
    second = "CCCCOc1c(C(=O)c2c(F)cc(C)cc2F)cnc2n[nH]cc12"
    assert tautomer_invariant_scaffold_key(first) == tautomer_invariant_scaffold_key(
        second
    )


def test_molecular_graph_is_symmetric_and_preserves_bond_types() -> None:
    atom_features, bond_types = molecular_graph("CC(=O)c1ccccc1")
    assert atom_features.shape == (9, 7)
    assert bond_types.shape == (9, 9)
    assert np.array_equal(bond_types, bond_types.T)
    assert 2 in bond_types
    assert 4 in bond_types
