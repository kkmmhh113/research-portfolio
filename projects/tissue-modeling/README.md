# Tissue-Based Modeling toward a Human Digital Twin

**2026–present · Independent project · Ongoing**

[Model code](src/our_star/models/) · [Tests](tests/) · [Code guide](CODE.md) · [Detailed evaluation](docs/evaluation.md)

## Research topic

I started this project with an interest in connecting tissue-specific models and different types of biological data into a human digital twin. The current work focuses on drug movement and elimination, comparing a simplified model of several organs with human pharmacokinetic observations.

The central topic is the relationship between a model's numerical consistency and its ability to predict measured drug concentrations.

## Current work and my role

The project uses physiologically based pharmacokinetic (PBPK) models to represent drug movement through the gastrointestinal tract, circulation, liver, kidneys, and other tissues. It compares concentration–time predictions with published human data and simpler compartmental models.

I proposed the tissue-integration direction. I am studying the model mechanisms and assumptions to understand the evaluation results and their limitations.

## Selected results

The historical v0.3 evaluation compared predictions with four analytes measured across three studies.

| Analyte | Absolute average fold error, AAFE ↓ |
| --- | ---: |
| Caffeine | 1.348 |
| Paracetamol | 1.389 |
| Enalapril | 3.137 |
| Enalaprilat | 12.229 |

AAFE measures absolute fold error; **values closer to 1 indicate better agreement**. Errors were relatively small for caffeine and paracetamol, but larger for enalapril and its metabolite enalaprilat.

Model configurations differed by compound, so these are not four independent validations of one identical model. Caffeine and paracetamol also failed to meet all of the historical evaluation criteria.

## Findings and limitations

- **Numerical consistency and predictive accuracy required separate checks.** Mass-balance errors stayed within the numerical tolerance even when concentration predictions differed substantially from observations.
- **Some parameters could not be uniquely determined from the available curves.** A parameter value that fits a curve is not necessarily a reliable estimate of the underlying physiology.
- **Adding a mechanism did not always improve prediction.** In a separate midazolam development experiment, adding intestinal CYP3A increased parent-drug AAFE from 1.421 to 5.980. This reflects the settings of that experiment, not a general conclusion that the mechanism is unnecessary.

The historical coverage calculation compared observed cohort means with simulated individual distributions, so it does not establish calibrated uncertainty. Some comparisons also used serum observations against modeled plasma without conversion. These are historical research results, not independently preregistered prospective validation or evidence of clinical readiness.

## Next steps

I aim to clarify the main parameters and assumptions, investigate prediction failures, and compare extensions with simpler models. Evaluation also needs to match predictions and observations by analyte, measurement conditions, and statistical target.

Human digital twins and using preclinical data to predict clinical outcomes remain long-term interests. The current focus is understanding and evaluating the reduced pharmacokinetic model.

<details>
<summary>Detailed results and code coverage</summary>

The [evaluation record](docs/evaluation.md) preserves the full metric table, failed criteria, identifiability results, and evaluation-history qualifications. [Saved aggregate results](results/reviewed_results.json) retain full-precision values and source hashes.

The source package includes numerical models, population sampling, prediction and analysis utilities, example configurations, and tests. It excludes private orchestration, unpublished manuscript tooling, and raw clinical datasets. The included synthetic example demonstrates the solver; it does not reproduce the human-data results above. See the [code guide](CODE.md).

</details>

[Back to portfolio](../../README.md)
