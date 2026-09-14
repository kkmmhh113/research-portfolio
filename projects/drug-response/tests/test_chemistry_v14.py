import pytest

from my_star import chemistry_v14 as chemistry
from my_star.chemistry import standardize_smiles as historical_standardize


@pytest.mark.parametrize("value", [None, "", "NA", "not a smiles", "*CC", "[Na+].*"])
def test_invalid_parent_rejected(value):
    assert chemistry.standardize_parent_smiles(value) is None


@pytest.mark.parametrize(
    "covalent,ionic,neutral",
    [
        ("[Na]OC(=O)CCCc1ccccc1", "[Na+].[O-]C(=O)CCCc1ccccc1", "O=C(O)CCCc1ccccc1"),
        ("O.O.O.NCCCC(O)(P(O)(O)=O)P(O)(=O)O[Na]",
         "[Na+].[O-]P(=O)(O)C(O)(CCCN)P(=O)(O)O.O.O.O",
         "NCCCC(O)(P(=O)(O)O)P(=O)(O)O"),
    ],
)
def test_salt_parent_is_representation_invariant(covalent, ionic, neutral):
    expected = chemistry.standardize_parent_smiles(neutral)
    assert expected is not None
    assert chemistry.standardize_parent_smiles(covalent) == expected
    assert chemistry.standardize_parent_smiles(ionic) == expected
    assert chemistry.standardize_parent_smiles(expected.canonical_smiles) == expected


def test_historical_behavior_is_not_silently_rewritten():
    value = "[Na]OC(=O)CCCc1ccccc1"
    assert historical_standardize(value) != chemistry.standardize_parent_smiles(value)


def test_parent_retains_stereochemical_and_tautomer_source_distinctions():
    left = chemistry.standardize_parent_smiles("N[C@H](C)C(=O)O")
    right = chemistry.standardize_parent_smiles("N[C@@H](C)C(=O)O")
    unspecified = chemistry.standardize_parent_smiles("NC(C)C(=O)O")
    assert len({left.inchi_key, right.inchi_key, unspecified.inchi_key}) == 3
    keto = chemistry.standardize_parent_smiles("CC(=O)CC(C)=O")
    enol = chemistry.standardize_parent_smiles("CC(O)=CC(C)=O")
    assert keto.canonical_smiles != enol.canonical_smiles
    assert chemistry.tautomer_invariant_scaffold_key(keto.canonical_smiles) == (
        chemistry.tautomer_invariant_scaffold_key(enol.canonical_smiles)
    )


def test_tautomer_limit_fails_closed(monkeypatch):
    chemistry.tautomer_scaffold_details.cache_clear()
    monkeypatch.setattr(chemistry, "MAX_TAUTOMERS", 1)
    try:
        with pytest.raises(ValueError, match="Incomplete tautomer enumeration"):
            chemistry.tautomer_scaffold_details("CC(=O)CC(C)=O")
    finally:
        chemistry.tautomer_scaffold_details.cache_clear()


def test_tautomer_details_deterministic_and_complete():
    first = chemistry.tautomer_scaffold_details("CC(=O)CC(C)=O")
    chemistry.tautomer_scaffold_details.cache_clear()
    second = chemistry.tautomer_scaffold_details("CC(=O)CC(C)=O")
    assert first == second
    assert first[0].startswith("ACYCLIC::")
    assert first[1] > 1
