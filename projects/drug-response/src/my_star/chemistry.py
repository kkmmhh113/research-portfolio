from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, QED
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold


INVALID_STRUCTURE_VALUES = {"", "-666", "-666.0", "nan", "none", "unknown", "restricted"}
_TAUTOMER_ENUMERATOR = rdMolStandardize.TautomerEnumerator()


@dataclass(frozen=True)
class StandardizedStructure:
    canonical_smiles: str
    inchi_key: str


def clean_compound_name(value: str) -> str:
    value = re.sub(r"\([^)]*\)", " ", str(value).strip().lower())
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value).split())


def standardize_smiles(smiles: object) -> StandardizedStructure | None:
    if smiles is None:
        return None
    text = str(smiles).strip()
    if text.lower() in INVALID_STRUCTURE_VALUES:
        return None
    try:
        mol = Chem.MolFromSmiles(text)
        if mol is None:
            return None
        fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
        if fragments:
            mol = max(fragments, key=lambda fragment: fragment.GetNumHeavyAtoms())
        mol = rdMolStandardize.Uncharger().uncharge(mol)
        canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        key = Chem.MolToInchiKey(mol)
    except (RuntimeError, ValueError):
        return None
    if not canonical or not key:
        return None
    return StandardizedStructure(canonical, key)


def morgan_fingerprint(
    smiles: str,
    radius: int = 2,
    n_bits: int = 2048,
    use_counts: bool = False,
    include_chirality: bool = False,
) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid canonical SMILES: {smiles}")
    generator = AllChem.GetMorganGenerator(
        radius=radius,
        fpSize=n_bits,
        includeChirality=include_chirality,
    )
    array = np.zeros(n_bits, dtype=np.float32)
    if use_counts:
        fingerprint = generator.GetCountFingerprint(mol)
        for index, count in fingerprint.GetNonzeroElements().items():
            array[index] = np.log1p(count)
    else:
        fingerprint = generator.GetFingerprint(mol)
        DataStructs.ConvertToNumpyArray(fingerprint, array)
    return array


def molecular_descriptors(smiles: str, descriptor_set: str = "basic") -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid canonical SMILES: {smiles}")
    basic = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumRotatableBonds(mol),
        Chem.GetFormalCharge(mol),
        Descriptors.RingCount(mol),
    ]
    basic_scale = [500.0, 5.0, 150.0, 10.0, 10.0, 10.0, 5.0, 10.0]
    if descriptor_set == "basic":
        values = np.asarray(basic, dtype=np.float32)
        scale = np.asarray(basic_scale, dtype=np.float32)
    elif descriptor_set == "extended":
        values = np.asarray(
            basic
            + [
                Descriptors.HeavyAtomCount(mol),
                Descriptors.NumHeteroatoms(mol),
                Descriptors.NumAromaticRings(mol),
                Descriptors.NumAliphaticRings(mol),
                Descriptors.NumSaturatedRings(mol),
                Descriptors.FractionCSP3(mol),
                Descriptors.NHOHCount(mol),
                Descriptors.NOCount(mol),
                Descriptors.MolMR(mol),
                Descriptors.LabuteASA(mol),
                QED.qed(mol),
                Descriptors.BertzCT(mol),
                Descriptors.Chi0v(mol),
                Descriptors.Chi1v(mol),
                Descriptors.HallKierAlpha(mol),
            ],
            dtype=np.float32,
        )
        scale = np.asarray(
            basic_scale
            + [50.0, 20.0, 10.0, 10.0, 10.0, 1.0, 10.0, 20.0, 150.0, 200.0, 1.0, 1500.0, 30.0, 20.0, 5.0],
            dtype=np.float32,
        )
    else:
        raise ValueError(f"Unknown descriptor set: {descriptor_set}")
    values = np.nan_to_num(values, nan=0.0, posinf=5.0, neginf=-5.0)
    return np.clip(values / scale, -5.0, 5.0)


def descriptor_dimension(descriptor_set: str) -> int:
    if descriptor_set == "basic":
        return 8
    if descriptor_set == "extended":
        return 23
    raise ValueError(f"Unknown descriptor set: {descriptor_set}")


def molecular_graph(smiles: str) -> tuple[np.ndarray, np.ndarray]:
    """Encode a molecule as categorical atom features and a typed adjacency matrix."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid canonical SMILES: {smiles}")
    atom_features = np.asarray(
        [
            [
                min(atom.GetAtomicNum(), 118),
                min(atom.GetTotalDegree(), 6),
                min(max(atom.GetFormalCharge(), -5), 5) + 5,
                min(int(atom.GetHybridization()), 8),
                min(atom.GetTotalNumHs(), 5),
                min(int(atom.GetChiralTag()), 4),
                int(atom.GetIsAromatic()),
            ]
            for atom in mol.GetAtoms()
        ],
        dtype=np.int64,
    )
    bond_types = np.zeros((mol.GetNumAtoms(), mol.GetNumAtoms()), dtype=np.int64)
    bond_type_map = {
        Chem.BondType.SINGLE: 1,
        Chem.BondType.DOUBLE: 2,
        Chem.BondType.TRIPLE: 3,
        Chem.BondType.AROMATIC: 4,
    }
    for bond in mol.GetBonds():
        bond_type = bond_type_map.get(bond.GetBondType(), 1)
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_types[begin, end] = bond_type
        bond_types[end, begin] = bond_type
    return atom_features, bond_types


def scaffold_key(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid canonical SMILES: {smiles}")
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    return scaffold if scaffold else f"ACYCLIC::{smiles}"


def tautomer_invariant_scaffold_key(smiles: str) -> str:
    """Return a Murcko key after tautomer canonicalization for leakage audits."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid canonical SMILES: {smiles}")
    try:
        # The key deliberately ignores stereochemistry. Removing it before and
        # after tautomer enumeration also avoids invalid bond-stereo states that
        # some otherwise parseable ChEMBL structures trigger in RDKit.
        Chem.RemoveStereochemistry(mol)
        canonical = _TAUTOMER_ENUMERATOR.Canonicalize(mol)
        Chem.RemoveStereochemistry(canonical)
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(
            mol=canonical, includeChirality=False
        )
        if scaffold:
            return scaffold
        acyclic = Chem.MolToSmiles(canonical, canonical=True, isomericSmiles=False)
        return f"ACYCLIC::{acyclic}"
    except (RuntimeError, ValueError) as error:
        raise ValueError(
            f"Could not derive a tautomer-invariant scaffold: {smiles}"
        ) from error
