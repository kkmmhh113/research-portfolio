"""Inspect structure features without loading expression data or model weights."""

import argparse
import json

from my_star.chemistry import molecular_descriptors, morgan_fingerprint, standardize_smiles


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smiles", default="CCO")
    args = parser.parse_args()
    structure = standardize_smiles(args.smiles)
    if structure is None:
        parser.error("SMILES could not be standardized")
    fingerprint = morgan_fingerprint(structure.canonical_smiles)
    descriptors = molecular_descriptors(structure.canonical_smiles)
    print(json.dumps({
        "canonical_smiles": structure.canonical_smiles,
        "fingerprint_dimensions": int(fingerprint.size),
        "active_fingerprint_bits": int((fingerprint != 0).sum()),
        "descriptor_dimensions": int(descriptors.size),
    }, indent=2))


if __name__ == "__main__":
    main()
