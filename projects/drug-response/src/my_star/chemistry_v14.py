"""Explicit v1.4 chemistry contract; historical chemistry remains unchanged."""

from __future__ import annotations

from functools import lru_cache

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

from .chemistry import INVALID_STRUCTURE_VALUES, StandardizedStructure


PARENT_POLICY = "rdkit_cleanup_fragment_parent_uncharge_v1"
TAUTOMER_POLICY = "complete_enumeration_10000_tautomers_100000_transforms_v1"
MAX_TAUTOMERS = 10_000
MAX_TRANSFORMS = 100_000


def standardize_parent_smiles(smiles: object) -> StandardizedStructure | None:
    """Disconnect salt metals and normalize representations before parent selection.

    Tautomers and unspecified stereochemistry are not filled in from drug names.
    The original author SMILES stays in the input/provenance table.
    """
    if smiles is None:
        return None
    value = str(smiles).strip()
    if value.lower() in INVALID_STRUCTURE_VALUES:
        return None
    try:
        mol = Chem.MolFromSmiles(value)
        if mol is None or mol.GetNumAtoms() == 0:
            return None
        if any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms()):
            return None
        cleaned = rdMolStandardize.Cleanup(mol)
        parent = rdMolStandardize.FragmentParent(cleaned, skipStandardize=True)
        parent = rdMolStandardize.Uncharger().uncharge(parent)
        if any(atom.GetAtomicNum() == 0 for atom in parent.GetAtoms()):
            return None
        canonical = Chem.MolToSmiles(parent, canonical=True, isomericSmiles=True)
        inchi_key = Chem.MolToInchiKey(parent)
    except (RuntimeError, ValueError):
        return None
    if not canonical or not inchi_key:
        return None
    return StandardizedStructure(canonical, inchi_key)


@lru_cache(maxsize=4096)
def tautomer_scaffold_details(smiles: str) -> tuple[str, int]:
    """Return an audited key only after enumeration completes within fixed limits."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid canonical SMILES: {smiles}")
    Chem.RemoveStereochemistry(mol)
    enumerator = rdMolStandardize.TautomerEnumerator()
    enumerator.SetMaxTautomers(MAX_TAUTOMERS)
    enumerator.SetMaxTransforms(MAX_TRANSFORMS)
    result = enumerator.Enumerate(mol)
    if result.status != rdMolStandardize.TautomerEnumeratorStatus.Completed:
        raise ValueError(
            f"Incomplete tautomer enumeration ({result.status}, {len(result)} states): {smiles}"
        )
    canonical = enumerator.PickCanonical(result)
    Chem.RemoveStereochemistry(canonical)
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=canonical, includeChirality=False)
    if not scaffold:
        scaffold = "ACYCLIC::" + Chem.MolToSmiles(
            canonical, canonical=True, isomericSmiles=False
        )
    return scaffold, len(result)


def tautomer_invariant_scaffold_key(smiles: str) -> str:
    return tautomer_scaffold_details(smiles)[0]
