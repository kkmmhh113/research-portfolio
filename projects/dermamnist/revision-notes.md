# Notebook revision notes

The notebooks are a 2026 revision of a 2025 training project. The original files are retained privately; the notebook code in this repository matches the previously reviewed revision.

## Changes

- Save checkpoints under unique epoch filenames and track the selected files explicitly. The original fixed-parameter notebook could overwrite a ranked file without preserving the previous ranking.
- Name the ensemble aggregation as mean logits. A checkpoint selected by both accuracy and loss retains two votes, preserving the original weighting rule.
- Use a separate directory for each run to avoid mixing checkpoints from different experiments.
- Replace hard-coded CUDA calls with device-based tensor placement.
- Select ImageNet V1 weights explicitly, matching the original pretrained API.
- Calculate class weights from training labels and save configuration, selection, and metric records.
- Separate test evaluation from training and validation.
- Split long notebook cells by stage and clear old outputs from the revised notebooks.

## Remaining limits

The revised notebooks have not been retrained or verified end to end. The source of the original normalization constants is still unconfirmed. Original checkpoints and Optuna study results are unavailable. Historical test accuracy cannot be verified as the intended ensemble's performance from the retained materials alone.

The test split has already been examined. A new run on that split should not be described as evaluation on previously unseen test data.

The fixed-parameter and Optuna notebooks use different schedulers. They do not isolate the effect of hyperparameter search.

[Back to project](README.md)
