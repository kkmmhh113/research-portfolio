# DermaMNIST Classification Model

**2025 · Independent project · Model training experience**

I chose DermaMNIST as my first deep-learning project because I was interested in medical image analysis and could access a public dataset. I used an ImageNet-pretrained MobileNetV2, replaced its classifier for seven classes, and fine-tuned the model.

[한국어 프로젝트 기록](project-notes.ko.md) · [Fixed-parameter notebook](notebooks/dermamnist_baseline.ipynb) · [Optuna notebook](notebooks/dermamnist_optuna.ipynb)

## What I tried

I selected MobileNetV2 because its lightweight architecture seemed suitable for trying training settings with limited computing resources. I included a learning-rate scheduler from the start and used AdamW as the optimizer.

While checking validation loss and accuracy across epochs, I noticed that the lowest-loss checkpoint and highest-accuracy checkpoint were not always the same. I tried combining saved checkpoints and explored Optuna to compare hyperparameter settings.

| Component | Configuration in the notebooks |
| --- | --- |
| Model | ImageNet-pretrained MobileNetV2; seven-class classifier; full-model fine-tuning |
| Optimizer | AdamW |
| Scheduler | CosineAnnealingLR in the fixed-parameter notebook; CosineAnnealingWarmRestarts in the Optuna notebook |
| Ensemble | Mean logits from selected epoch checkpoints of the same architecture |
| Optuna parameters | Feature-extractor learning rate, classifier learning rate, weight decay |

Other settings retained in the notebooks include augmentation, class-weighted focal loss, and Mixup. The table highlights the choices discussed in my project account; it is not a claim that I designed a new network architecture.

## Results and what I learned

My recollection is that ensembling and hyperparameter tuning did not produce a large improvement. The retained files do not provide a complete comparison table or Optuna study results, so I do not report a measured gain from either method.

The original notebook contains a test-ensemble accuracy of **86.93%**. This is a historical saved output, not a result from the revised notebooks. During code review, a checkpoint-ranking problem was found, and the original checkpoint files are unavailable. The score therefore cannot currently be verified against the intended ensemble selection. See [the result record](results/historical_results.json) and [revision notes](revision-notes.md).

This project helped me understand the training loop: making predictions, calculating loss, updating weights, and checking validation results. It also taught me that adding a method does not automatically improve accuracy, and that checkpoint handling is part of a reliable evaluation.

## Code status

The notebooks in this repository are the 2026 revision of the 2025 project. They separate training stages and correct checkpoint tracking. They have **not been retrained**. Normalization provenance and end-to-end execution still need verification. The two notebooks also use different schedules, so their difference is not a controlled estimate of Optuna's effect.

## References

- [MedMNIST dataset](https://medmnist.com/)
- [Torchvision MobileNetV2](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v2.html)

[Back to portfolio](../../README.md)
