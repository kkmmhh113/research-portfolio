# Code verification

Reviewed on 14 September 2026. This source export keeps numerical implementation separate from historical performance evidence.

| Check | Result |
| --- | --- |
| Drug-response package tests | 117 passed |
| Tissue-modeling package tests | 133 passed |
| Wheel builds and separate-directory installation | Both packages passed |
| Installed submodule imports | 70 passed |
| Molecular feature example | 2,048 fingerprint dimensions and 8 descriptor dimensions for ethanol |
| Synthetic PBPK example | 49 time points; maximum mass error approximately 2.84 × 10⁻⁹ mg; minimum state amount 0 mg |

Tests and examples were run on macOS with Python 3.13, PyTorch 2.11.0, RDKit 2026.3.4, NumPy 2.4.4, pandas 2.3.3, SciPy 1.17.1, PyYAML 6.0.3, PyArrow 25.0.0, and pytest 9.1.1. The workflow in this repository also runs the included tests and examples on Linux with Python 3.11; its current result is available in GitHub Actions.

These are the tests included in this curated source export, not the entire original research-workspace suites. No benchmark training or locked-test evaluation was rerun. The DermaMNIST notebooks retain the separate verification limits described in their [project page](../projects/dermamnist/README.md).

## Export decisions

- Preserve original module names, numerical logic, validation checks, and useful comments explaining assumptions or numerical details.
- Keep scientific provenance in parameter configurations. Remove a redundant inline comment; source-to-export Python syntax-tree comparisons confirm the copied numerical code is unchanged.
- Include models, reusable analysis functions, representative training/inference utilities, relevant configurations, and self-contained tests.
- Exclude raw datasets, trained weights, personal and remote-host settings, private orchestration, cached files, manuscript-generation tools, and internal working notes.

The current source snapshot includes later development code. It is not asserted to be the exact implementation revision used for every historical result. The reported historical metrics and their limitations remain in the project evaluation records.

[Back to portfolio](../README.md)
