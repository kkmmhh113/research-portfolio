# Tissue-Based Modeling toward a Human Digital Twin

**2026–present · Independent project · Project concept**

My long-term interest is to connect tissue-specific models and multiple types of biological data into a human digital twin. The current project starts with a narrower question: how can a compartment-based model describe drug movement and elimination across tissues?

## Current approach

The project uses physiologically based pharmacokinetic (PBPK) modeling to represent drug amounts and concentration–time profiles. This is a starting point for tissue modeling, rather than a completed human digital twin.

The current work examines a reduced multi-compartment model against published human pharmacokinetic data and simpler compartmental models. A central evaluation question is whether numerical consistency also leads to accurate concentration predictions.

## My role

I proposed the tissue-integration direction. I am continuing to study the model mechanisms and clarify which assumptions and comparisons are appropriate for its current implementation.

## Selected historical evaluation results

The saved v0.3 evaluation compared concentration–time predictions with published human pharmacokinetic observations from three studies. Caffeine and paracetamol used the segmented gastrointestinal model lineage; enalapril and its metabolite enalaprilat used a parent–metabolite model. These are four measured analytes across three studies, rather than four independent studies of one identical model.

| Analyte | Study | AAFE ↓ | Predictions within twofold of observations ↑ | Historical p05–p95 envelope coverage |
| --- | --- | ---: | ---: | ---: |
| Caffeine | PKDB00019 | **1.348** | 90.9% | 54.5% |
| Paracetamol | PKDB00021 | **1.389** | 88.9% | 66.7% |
| Enalapril | PKDB00805 | 3.137 | 33.3% | 5.9% |
| Enalaprilat | PKDB00805 | 12.229 | 14.3% | 6.3% |

AAFE summarizes absolute fold error, with 1 indicating exact agreement. Twofold agreement uses positive observations eligible for the logarithmic comparison; coverage uses the available observation points, so the denominators can differ. Enalaprilat was measured after enalapril administration.

Caffeine and paracetamol met the historical point-prediction criteria, including AAFE ≤ 2 and at least 80% of eligible predictions within twofold. Both failed the historical coverage threshold of 70%. Enalapril and enalaprilat failed multiple point-prediction criteria as well as coverage.

**The coverage column is a historical diagnostic, not a calibrated confidence-interval result:** it compares published cohort means with the p05–p95 range of simulated individuals. Those quantities do not represent the same statistical target. Enalapril and enalaprilat also involved a serum-to-modeled-plasma comparison without conversion.

## What the evaluations revealed

**Numerical consistency and prediction accuracy need separate checks.** The maximum recorded mass-balance error across these evaluations was **4.35 × 10⁻⁸ mg**, within the project's numerical tolerance, even though some concentration predictions were poor. Mass conservation verifies the model's accounting; it does not establish biological accuracy.

**The available curves did not uniquely determine all parameters.** The caffeine sensitivity analysis examined 19 parameter directions against 11 output dimensions. Its effective rank was 10, below the output-limited maximum of 11, and three parameter profiles reached their search boundaries. These results limit how confidently fitted parameters can be interpreted.

**Adding a mechanism did not automatically improve the fit.** In a separate midazolam development comparison, adding intestinal CYP3A increased parent-drug AAFE from **1.421 to 5.980**. This was a development-stage comparison with its existing fitting choices, not an independent test or a general conclusion about that mechanism.

## Current interpretation and next questions

The historical evaluation was internally frozen, with a disclosed schema/design amendment after the data files were materialized and before scoring. It was not an independently registered prospective validation. The tables preserve those results and failures; they do not establish clinical readiness.

Before extending the tissue model, I want to understand which parameters the data support, compare added mechanisms with simpler models, and match uncertainty estimates to the observed quantity. Human pharmacokinetic data are the current evaluation source. Using preclinical data to predict clinical outcomes remains a broader research interest.

## Result record

The [reviewed result extract](../results/reviewed_results.json) includes full-precision metrics, historical failed criteria, the identifiability and midazolam results, and source-file hashes. This review used saved reports without new fitting, simulation, or access to clinical observation tables. The source-report hashes matched those recorded in the existing figure data. The source package is described in [CODE.md](../CODE.md); raw clinical data are not distributed here.

[Back to portfolio](../../../README.md)
