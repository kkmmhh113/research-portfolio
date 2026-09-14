"""Qualified chemical metadata that must survive SciPlex alias aggregation.

Graph identity is not experimental identity. These checks preserve representation
and provenance, but never approve a research protocol or an external data join.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import pandas as pd
from rdkit import Chem, rdBase

from .chemistry import scaffold_key
from .chemistry_v14 import PARENT_POLICY, TAUTOMER_POLICY, standardize_parent_smiles, tautomer_scaffold_details

CONTRACT_COLUMN = "representation_contract_json"
CONTRACT_HASH_COLUMN = "representation_contract_sha256"
CONTRACT_COLUMNS = (CONTRACT_COLUMN, CONTRACT_HASH_COLUMN)
OBSERVATION_CONTRACT_COLUMNS = ("structure_contracts_json", "structure_contracts_sha256")
QUALIFIERS = {"source_encoded_parent", "trans_racemate_intended_metadata_not_lot_verified",
              "trans_relative_geometry_absolute_composition_uncertain"}
COARSE_QUALIFIERS = QUALIFIERS - {"source_encoded_parent"}


class StructureContractError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise StructureContractError(message)


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def text_sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stereo_completeness(smiles: str) -> dict:
    parent = standardize_parent_smiles(smiles)
    require(parent is not None, "Invalid chemical graph")
    mol = Chem.MolFromSmiles(parent.canonical_smiles)
    centres = Chem.FindMolChiralCenters(Chem.Mol(mol), force=True, includeUnassigned=True,
                                       useLegacyImplementation=False)
    unspecified = sum(cip == "?" for _, cip in centres)
    assigned = len(centres) - unspecified
    state = ("none_detected" if not centres else "partially_specified" if unspecified and assigned
             else "unspecified" if unspecified else "fully_specified_detected_centres")
    return {"detected_centres": len(centres), "assigned": assigned, "unspecified": unspecified,
            "state": state, "cip_descriptors": [cip for _, cip in centres],
            "specified_bond_stereo": sorted(str(b.GetStereo()) for b in mol.GetBonds()
                if b.GetStereo() != Chem.BondStereo.STEREONONE),
            "does_not_certify_achirality_purity_or_mixture": True}


@dataclass(frozen=True)
class QualifiedStructure:
    """Validated canonical JSON is immutable; returned dictionaries are copies."""
    encoded: str

    @property
    def payload(self) -> dict:
        return json.loads(self.encoded)

    @classmethod
    def from_row(cls, row: dict) -> "QualifiedStructure":
        encoded = row.get(CONTRACT_COLUMN)
        require(isinstance(encoded, str), "Missing representation contract")
        require(text_sha(encoded) == row.get(CONTRACT_HASH_COLUMN), "Representation contract hash mismatch")
        value = json.loads(encoded)
        require(canonical_json(value) == encoded, "Representation contract must use canonical JSON")
        require(set(value) == {"schema_version", "source", "graph", "stereochemistry", "caveats", "review_evidence"}
                and type(value["schema_version"]) is int and value["schema_version"] == 1,
                "Unsupported representation contract schema")
        source, graph, stereo = value["source"], value["graph"], value["stereochemistry"]
        require(source["perturbation"] == row["perturbation"]
                and source["catalog_number"] == row["structure_identifier"]
                and source["input_smiles_sha256"] == text_sha(row["input_smiles"]),
                "Representation contract is bound to another source row")
        require(source.get("assay_lot_identity_certified") is False,
                "This metadata contract cannot certify an experimental lot")
        require(type(source.get("nonabsolute_source_annotation")) is bool
                and type(source.get("active_coarsening_proposed")) is bool,
                "Source qualifier flags must be booleans")
        require(isinstance(source.get("review_kind"), str) and bool(source["review_kind"]), "Missing source review kind")
        bindings = source.get("review_bindings")
        require(isinstance(bindings, dict) and set(bindings) == {"input_receipt", "stereo_review", "mapping_review"}
                and all(isinstance(v, str) and len(v) == 64 and all(c in "0123456789abcdef" for c in v)
                        for v in bindings.values()), "Missing typed source-review bindings")
        parent = standardize_parent_smiles(row["canonical_smiles"])
        require(parent is not None and parent.canonical_smiles == row["canonical_smiles"]
                and parent.inchi_key == row["compound_id"] == row["inchi_key"], "Parent graph identity mismatch")
        scaffold, states = tautomer_scaffold_details(parent.canonical_smiles)
        expected_graph = {"canonical_smiles": parent.canonical_smiles, "parent_graph_key": parent.inchi_key,
                          "scaffold": scaffold_key(parent.canonical_smiles), "tautomer_scaffold": scaffold,
                          "completed_tautomer_states": states, "parent_policy": PARENT_POLICY,
                          "tautomer_policy": TAUTOMER_POLICY, "rdkit_version": rdBase.rdkitVersion}
        require(canonical_json(graph) == canonical_json(expected_graph)
                and row["scaffold"] == graph["scaffold"]
                and row["tautomer_invariant_scaffold"] == scaffold, "Graph/scaffold contract did not replay")
        require(stereo["qualifier"] in QUALIFIERS, "Unknown composition qualifier")
        require(canonical_json(stereo["completeness"]) == canonical_json(stereo_completeness(parent.canonical_smiles)),
                "Full-table stereo completeness did not replay")
        refs = stereo.get("reference_enantiomer_smiles")
        require(isinstance(refs, list) and refs == sorted(set(refs)), "Invalid enantiomer-reference set")
        coarse = stereo["qualifier"] in COARSE_QUALIFIERS
        require(source["active_coarsening_proposed"] is coarse, "Coarsening/source-origin mismatch")
        require(type(stereo.get("coarse_graph_does_not_encode_cis_trans")) is bool
                and stereo["coarse_graph_does_not_encode_cis_trans"] is coarse,
                "Coarse graph must retain its geometry limitation")
        if coarse:
            require(len(refs) == 2 and stereo["relative_geometry"] == "trans"
                    and stereo["completeness"]["unspecified"] > 0,
                    "Coarse graph lost its relative-geometry/enantiomer references")
            molecules = []
            for reference in refs:
                ref_parent = standardize_parent_smiles(reference)
                require(ref_parent is not None and ref_parent.canonical_smiles == reference, "Invalid reference graph")
                mol = Chem.MolFromSmiles(reference)
                molecules.append(Chem.Mol(mol))
                for atom in mol.GetAtoms():
                    require(atom.GetChiralTag() in {Chem.ChiralType.CHI_UNSPECIFIED,
                            Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW},
                            "Unsupported reference chirality")
                    atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
                require(Chem.MolToSmiles(mol, isomericSmiles=True) == parent.canonical_smiles,
                        "Reference graph differs beyond tetrahedral labels")
            for atom in molecules[0].GetAtoms():
                atom.InvertChirality()
            require(Chem.MolToSmiles(molecules[0], isomericSmiles=True) == refs[1],
                    "References are not a mirror pair")
        else:
            require(refs == [] and stereo.get("relative_geometry") is None,
                    "Unexpected coarse reference on source-encoded graph")
        caveats = value["caveats"]
        require(isinstance(caveats, list) and all(isinstance(v, str) and v for v in caveats)
                and caveats == sorted(set(caveats)) and "original_assay_lot_not_certified" in caveats,
                "Missing or invalid chemical caveats")
        require(isinstance(value["review_evidence"], dict) and bool(value["review_evidence"]), "Source evidence was dropped")
        return cls(encoded)

    def pooling_signature(self) -> str:
        value = self.payload
        stereo = value["stereochemistry"]
        # Provenance is retained for every source separately; the key contains
        # only chemical meaning. Counterion-only review must not break ENMD aliases.
        return canonical_json({"graph": value["graph"], "stereochemistry": stereo,
                               "specificity_caveats": [c for c in value["caveats"]
                                  if c.startswith("specificity_") or c == "do_not_merge_with_glesatinib"]})


def has_qualified_metadata(compounds: pd.DataFrame) -> bool:
    present = set(CONTRACT_COLUMNS) & set(compounds.columns)
    require(not present or present == set(CONTRACT_COLUMNS), "Partial representation metadata cannot be ignored")
    return bool(present)


def validate_qualified_table(compounds: pd.DataFrame) -> dict[str, QualifiedStructure]:
    require(has_qualified_metadata(compounds) and not compounds.empty, "Qualified structures required")
    require(not compounds["perturbation"].isna().any() and not compounds["perturbation"].duplicated().any(),
            "Missing or duplicate source perturbation")
    require(not compounds["structure_identifier"].isna().any()
            and not compounds["structure_identifier"].duplicated().any(), "Missing or duplicate source catalog")
    records = {row["perturbation"]: QualifiedStructure.from_row(row) for row in compounds.to_dict("records")}
    by_parent: dict[str, set[str]] = {}
    for record in records.values():
        value = record.payload
        by_parent.setdefault(value["graph"]["parent_graph_key"], set()).add(record.pooling_signature())
    require(all(len(signatures) == 1 for signatures in by_parent.values()),
            "Equal graphs have incompatible stereo/composition qualifiers; pooling is forbidden")
    return records


def attach_qualified_metadata(observations: pd.DataFrame, compounds: pd.DataFrame) -> pd.DataFrame:
    """Keep the exact contributing source contracts on every pooled observation."""
    records = validate_qualified_table(compounds)
    require(not set(OBSERVATION_CONTRACT_COLUMNS) & set(observations.columns), "Refusing to overwrite output provenance")
    require("perturbation" in observations, "Pooled rows must retain contributing perturbation names")
    payloads = []
    for row in observations.to_dict("records"):
        names = str(row["perturbation"]).split(" || ")
        require(names == sorted(set(names)) and all(name in records for name in names),
                "Invalid or missing contributing source aliases")
        if "perturbation_count" in row:
            require(row["perturbation_count"] == len(names), "Pooled alias count differs from source provenance")
        values = [records[name].payload for name in names]
        require(all(value["graph"]["parent_graph_key"] == row["compound_id"] for value in values),
                "A pooled observation points to another parent graph")
        if "canonical_smiles" in row:
            require(all(value["graph"]["canonical_smiles"] == row["canonical_smiles"] for value in values),
                    "Pooled observation SMILES changed")
        payloads.append(canonical_json({"schema_version": 1, "measured_source_contracts": values,
            "training_protocol_approval": "not_conferred_by_chemical_metadata"}))
    result = observations.copy()
    result[OBSERVATION_CONTRACT_COLUMNS[0]] = payloads
    result[OBSERVATION_CONTRACT_COLUMNS[1]] = [text_sha(value) for value in payloads]
    return result


def external_join_disposition(left: QualifiedStructure, right: QualifiedStructure) -> dict:
    """A representation screen, never sufficient permission to join responses."""
    a, b = left.payload, right.payload
    if a["graph"]["parent_graph_key"] != b["graph"]["parent_graph_key"]:
        status = "reject_different_parent_graph"
    elif left.pooling_signature() != right.pooling_signature():
        status = "review_required_incompatible_qualifier_or_specificity"
    else:
        status = "representation_compatible_external_provenance_and_split_audit_still_required"
    return {"status": status, "external_response_join_approved": False,
            "name_only_join_allowed": False, "assay_lot_identity_certified": False}
