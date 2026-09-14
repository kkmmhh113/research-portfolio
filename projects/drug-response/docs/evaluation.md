# Molecular Structure-Based Drug Response Prediction

**2026–present · Independent project · Research planning**

Can molecular structure help predict the response to a chemical compound that was not observed during training?

My initial goal was to predict broader biological effects of compounds. After reviewing early model performance and available data, I narrowed the project to drug-induced gene-expression changes. The current project uses public SciPlex3 data from A549, K562, and MCF7 cells.

## My decisions

I proposed using molecular structure as an input, chose to focus on gene-expression responses, and selected the three cell lines supported by the available data. The aim is to learn a relationship between chemical structure and biological response, rather than rely on drug names alone.

## Approach

The project combines molecular structure with cell-line and treatment information to predict expression changes relative to controls. Gene expression is the current measurable output; predicting broader drug effects remains a longer-term goal.

The historical model used molecular fingerprints and descriptors together with cell, dose, time, and baseline-expression information. Conditions belonging to the same compound were kept together, and chemical scaffold groups were separated between training, validation, and evaluation.

## Selected historical results

The Phase 4 experiment used 186 compounds and 2,396 treatment conditions across three cell lines. Its outer evaluation contained 28 compounds and 359 conditions. The saved three-seed ensemble was compared with simple baselines on 978 historical response coordinates.

**These are results from the earlier scPerturb v1.3 dataset, with unresolved biological annotation and structure-mapping limitations. They are not final results for the rebuilt dataset.**

| Saved Phase 4 result | Mean Pearson ↑ | RMSE ↓ | MAE ↓ |
| --- | ---: | ---: | ---: |
| Training-mean baseline | 0.1708 | 0.1478 | 0.0869 |
| Context-mean baseline | 0.2992 | 0.1435 | 0.0848 |
| Molecular-structure ensemble | **0.4326** | **0.1341** | **0.0796** |

Mean Pearson measures agreement in the response pattern, averaged over treatment conditions; it is not classification accuracy. RMSE and MAE measure error on the processed response scale. The context baseline provides a reference based on biological context without distinguishing compounds by their molecular structure.

The saved mean Pearson gain over the context baseline was **0.1335**. A historical compound-block bootstrap gave a 95% interval of **0.0946–0.1720** using 2,000 resamples. This interval describes uncertainty within the old benchmark; it does not account for the subsequently discovered data issues.

The aggregate improvement was not uniform: in the 29 observations at 72 hours, the model's Pearson was **0.3891**, compared with **0.4004** for the context baseline. This small, retrospectively examined subgroup motivates further time-course evaluation.

## Further experiments and limitations

- **Transfer and model refinement:** a later Phase 6 ensemble recorded Pearson **0.4674** on 977 historical coordinates. It reused the previously evaluated outer set and changed the training setup and output panel, so it is retained as exploratory history rather than independent confirmation of improvement over Phase 4.
- **Auxiliary supervision:** adding molecular-target labels did not improve the selection-fold result: Pearson **0.3852 → 0.3844**. A separate repeated-target/mechanism experiment also declined, **0.3848 → 0.3802**. Both stopped at the first selection gate without outer-test evaluation. These are development comparisons, not evidence that auxiliary supervision cannot work in general.
- **Data review:** a later audit found an off-by-one gene-annotation issue in the v1.3 source and incorrect historical structure mappings for some compounds. The saved numbers can describe the old numerical experiment, but they cannot support named-gene biological conclusions. A separately reviewed v1.4 dataset, including the response definition and preprocessing, is still needed.
- **Generalization:** the data come from three cancer cell lines. These results do not establish patient-level drug effects or clinical outcomes, and no superiority over published methods has been established.

The next priority is to resolve the data and evaluation contracts before interpreting further model changes. The negative results also make it useful to test additional biological information and model complexity through controlled comparisons.

## Result record

The [reviewed result extract](../results/reviewed_results.json) contains full-precision metrics, the saved subgroup and auxiliary results, and source-file hashes. This portfolio review used existing reports without retraining or rerunning the outer evaluation. The Phase 4 frozen manifest and three checkpoint hashes matched the stored audit. The source package is described in [CODE.md](../CODE.md); raw datasets and trained weights are not distributed here.

[Back to portfolio](../../../README.md)
