# Molecular Structure-Based Drug Response Prediction

**2026–present · Independent project · Research planning**

[Model code](src/my_star/model.py) · [Training code](scripts/train_sciplex.py) · [Tests](tests/) · [Code guide](CODE.md)

## Research topic

This project explores whether molecular structure can help predict the response to compounds not observed during training. My initial interest was in broader biological effects; after reviewing early performance and available data, I narrowed the work to drug-induced gene-expression changes in A549, K562, and MCF7 cells using public SciPlex3 data.

## Approach and my role

I proposed molecular structure as an input, chose the gene-expression focus, and selected the three cell lines supported by the data.

The historical model combines molecular fingerprints and descriptors with cell, dose, time, and baseline-expression information. It predicts changes relative to controls. All conditions for one compound stay in the same split, with chemical scaffold groups separated across training, validation, and evaluation.

## Selected historical results

The Phase 4 benchmark used 186 compounds and 2,396 conditions. Its outer evaluation covered 28 compounds and 359 conditions on 978 historical response coordinates.

**These results use scPerturb v1.3, with subsequently identified gene-annotation and chemical structure-mapping problems. They are not final results for a corrected dataset.**

| Model | Mean Pearson ↑ | RMSE ↓ | MAE ↓ |
| --- | ---: | ---: | ---: |
| Training-mean baseline | 0.1708 | 0.1478 | 0.0869 |
| Context-mean baseline | 0.2992 | 0.1435 | 0.0848 |
| Molecular-structure ensemble | **0.4326** | **0.1341** | **0.0796** |

Mean Pearson measures response-pattern agreement averaged over treatment conditions, not classification accuracy. The saved gain over the context baseline was **0.1335**, with a historical compound-block bootstrap 95% interval of **0.0946–0.1720**. This interval does not account for the later data-quality findings.

## Findings and limitations

- **The aggregate improvement was not uniform.** In 29 observations at 72 hours, Pearson was 0.3891 for the model versus 0.4004 for the context baseline.
- **Additional supervision did not consistently help.** Target-label and repeated-target/mechanism experiments failed their development selection gates; neither proceeded to outer evaluation.
- **Later scores are not independent confirmation.** Phase 6 recorded Pearson 0.4674, but reused the evaluated outer set and changed the training setup and output panel.
- **The data need reconstruction.** The annotation and structure-mapping findings prevent named-gene biological interpretation of historical scores. A separately reviewed v1.4 dataset and evaluation protocol remain necessary.

These results concern three cancer cell lines. They do not establish patient-level drug effects, clinical outcomes, or superiority over published methods.

## Next steps

The priority is to resolve the response definition, structure mapping, preprocessing, and evaluation contracts before interpreting further model changes. Additional biological information and model complexity need controlled comparisons.

<details>
<summary>Detailed experiments and code coverage</summary>

The [evaluation record](docs/evaluation.md) contains the complete narrative, auxiliary comparisons, and interpretation limits. The [saved result extract](results/reviewed_results.json) contains full-precision metrics and source hashes.

The source package includes the historical structure-conditioned and P/C/R models, training and inference code, split utilities, metric functions, and tests. The v1.4 helper modules are development utilities, not a completed corrected dataset pipeline. Raw expression data, checkpoints, remote experiment launchers, and manuscript tooling are excluded. The published tests use small fixtures and do not reproduce the benchmark scores. See the [code guide](CODE.md).

</details>

[Back to portfolio](../../README.md)
