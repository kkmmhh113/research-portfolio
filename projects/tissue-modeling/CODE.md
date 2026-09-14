# Code guide

This is a curated source export from the ongoing `our_star` project. It includes numerical model and analysis code, with original module names preserved. Clinical-data ingestion, study-access orchestration, manuscript generation, and private result archives are outside this package.

## Where to start

| Path | Purpose |
| --- | --- |
| [models/segmented.py](src/our_star/models/segmented.py) | Segmented gastrointestinal PBPK model |
| [models/parent_metabolite.py](src/our_star/models/parent_metabolite.py) | Parent–metabolite model used in the historical v0.3 family |
| [virtual_trial.py](src/our_star/virtual_trial.py) | ODE integration, dosing events, and trajectory summaries |
| [models](src/our_star/models/) and [chemistry](src/our_star/chemistry/) | Additional model variants, reaction networks, and amount accounting |
| [population](src/our_star/population/) and [prediction](src/our_star/prediction/) | Virtual-population sampling and prediction utilities |
| [analysis](src/our_star/analysis/) and [comparators](src/our_star/comparators/) | Sensitivity, identifiability, profiling, uncertainty, and simpler compartmental models |
| [configs](configs/) | Synthetic examples and historical parameter configurations needed by the included tests |
| [tests](tests/) | Numerical conservation, model behavior, configuration validation, and metric checks |

Model variants include later exploratory work as well as the historical models discussed in the README. A model's inclusion here does not imply clinical validation. Source parameters retain their original provenance and interpretation notes.

## Local setup

From this project directory, using Python 3.11 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q
python examples/synthetic_pbpk.py
```

On Windows, activate with `.venv\Scripts\Activate.ps1` in PowerShell. The example runs on CPU with a synthetic compound configuration. It reports mass accounting and state positivity, with no network calls or clinical observation inputs.

## Relationship to the reported results

The [evaluation record](docs/evaluation.md) and [saved aggregate results](results/reviewed_results.json) describe historical comparisons. This export contains a later source snapshot, not an independently reproduced release of every experiment.

The example and tests can run without the original clinical datasets. They verify numerical and software behavior; they do not recreate the human-data metrics. In particular, the synthetic probe's parameters are demonstration values, not estimates for a named medicine. Raw clinical observation tables and the complete historical evaluation pipeline are not distributed here.

All models remain reduced research models. Their outputs do not establish patient-level dosing, efficacy, or safety.

[Back to project](README.md)
