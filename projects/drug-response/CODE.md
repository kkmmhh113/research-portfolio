# Code guide

This is a curated source export from the ongoing `my_star` project. The original module names are retained so the model, training code, and saved experiment terminology can be followed together.

## Where to start

| Path | Purpose |
| --- | --- |
| [chemistry.py](src/my_star/chemistry.py) | Structure standardization, molecular fingerprints, descriptors, and scaffold keys |
| [model.py](src/my_star/model.py) | Historical structure/context/baseline model and loss functions |
| [data.py](src/my_star/data.py) and [splits.py](src/my_star/splits.py) | Dataset preparation, scaling, context priors, and compound/scaffold separation |
| [prc_model.py](src/my_star/prc_model.py) | Perturbation/context/readout model variants |
| [paper_a_metrics.py](src/my_star/paper_a_metrics.py) | Compound-level metric and uncertainty utilities, distinct from historical row-weighted metrics |
| [scripts](scripts/) | Historical training, prediction, candidate freezing, and evaluation utilities |
| [tests](tests/) | Small-fixture checks for features, models, splits, loss functions, and metrics |

The `chemistry_v14`, `structure_contract_v14`, and `control_fit_scope_v14` modules are development helpers. Their presence does not mean the corrected v1.4 benchmark is complete. Phase numbers in filenames refer to internal experiment stages, not stages of a clinical trial. The Phase 8 analysis helper is included as a historical metric comparator; no Phase 8 performance claim is made.

## Local setup

From this project directory, using Python 3.11 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q
python examples/molecular_features.py --smiles 'CCO'
```

On Windows, activate with `.venv\Scripts\Activate.ps1` in PowerShell. The example only extracts molecular features; it does not train a model or predict biological responses.

## Training and inference

[train_sciplex.py](scripts/train_sciplex.py) contains the historical training loop, and [train_prc.py](scripts/train_prc.py) contains the subsequent factorized-model training implementation. [predict_sciplex.py](scripts/predict_sciplex.py) accepts checkpoints, SMILES, treatment context, and a baseline-expression vector. Argument help can be inspected without loading datasets:

```bash
python scripts/train_sciplex.py --help
python scripts/train_prc.py --help
python scripts/predict_sciplex.py --help
```

The [Phase 4 configuration](configs/sciplex_phase4_v2_final_adaptive.yaml) records historical settings. Its `data/` paths are intentionally unresolved in this source export: expression tables, split artifacts, and trained weights are not bundled. Training and benchmark reproduction therefore require separately obtained, version-matched inputs. The current source is a later development snapshot and is not asserted to be the exact code revision that produced every historical score.

Before new experiments, resolve the corrected data contract and use validation for model selection. Do not reuse an exposed outer set as a new independent test. See [evaluation details](docs/evaluation.md) for the annotation, structure, and exposure limitations.

The included tests exercise implementation behavior with small fixtures. They do not train a benchmark model or verify the published historical scores end to end.

[Back to project](README.md)
