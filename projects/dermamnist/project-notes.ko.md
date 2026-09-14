# 첫 딥러닝 프로젝트 MobileNetV2로 DermaMNIST 분류하기

## 시작한 이유

의료영상 분석에 관심이 있어서 첫 프로젝트로 피부 병변 분류를 선택했다. DermaMNIST는 공개 데이터셋이라 구하기 쉬웠고, 처음 이미지 분류를 해보기에 괜찮을 것 같았다.

## 모델과 학습 설정

모델은 MobileNetV2를 골랐다. 가볍고 빠른 모델이라 여러 설정을 바꿔가며 학습하기에 적합하다고 생각했다. ImageNet pretrained model을 사용했다. 마지막 분류층은 피부 병변 7종을 분류하도록 바꿨고, 전체 모델을 fine-tuning했다.

Optimizer는 AdamW를 사용했다. Learning rate를 조절하기 위해 scheduler도 처음부터 넣었다. 특정 문제가 생겨서 추가한 것은 아니었다. 기본 학습 코드에는 CosineAnnealingLR, Optuna 코드에는 CosineAnnealingWarmRestarts를 사용했다.

## Ensemble을 시도한 이유

학습하면서 epoch별 validation loss와 accuracy를 확인했다. Loss가 가장 낮은 시점과 accuracy가 가장 높은 시점이 달라서, 각 시점에 저장한 모델을 합쳐보면 어떨까 생각했다. 성능을 더 높일 방법이 잘 떠오르지 않던 때라 ensemble을 시도했다.

합친 것은 같은 MobileNetV2를 서로 다른 epoch에 저장한 checkpoint들이다. 각 checkpoint에서 나온 로짓을 평균해 최종 클래스를 선택했다. 당시 단일 모델과 비교했을 때는 accuracy가 거의 비슷했고, 큰 성능 향상을 느끼지 못했다. 다만 지금 남아 있는 파일만으로는 당시의 단일 모델과 ensemble 차이를 정확한 수치로 확인하기 어렵다.

## Optuna

Hyperparameter에 따른 결과를 그래프로 비교할 수 있다고 해서 Optuna도 사용해봤다. 직접 값을 바꾸는 것보다 튜닝하기 편할 것 같았다. 코드에는 특징 추출부와 분류층의 learning rate, weight decay를 탐색하도록 설정했다.

당시에는 큰 성능 향상을 보지 못했던 것으로 기억한다. 현재 보관한 노트북에는 Optuna 실행 결과가 남아 있지 않아 최적 설정이나 개선 폭은 제시하지 않았다. 기본 코드와 Optuna 코드의 scheduler도 다르므로, 두 결과의 차이를 전부 Optuna의 효과로 해석할 수는 없다.

## 배운 점

첫 프로젝트라 딥러닝 모델이 어떻게 학습하고 예측하는지 알아가는 과정이었다. Loss를 계산하고 가중치를 업데이트하면서 정답에 가까워지는 방식을 배웠다. 직접 학습시킨 모델이 이미지를 분류하는 걸 보면서 ‘실제로 되는구나’라는 느낌을 받았다.

## 포트폴리오를 정리하며 확인한 것

기존 코드를 다시 검토하는 과정에서 정확도 상위 checkpoint를 저장하는 방식에 문제가 있다는 것을 확인했다. 새 최고 점수가 나오면 이전 순위의 파일을 제대로 유지하지 못하는 구조였다. 정리한 코드에서는 epoch별로 파일을 저장하고, 선택 목록이 해당 파일을 가리키도록 바꿨다.

기존 노트북에는 test ensemble accuracy 86.93%가 저장돼 있다. 하지만 원래 checkpoint 파일이 없고 저장 방식에도 문제가 있었기 때문에, 의도한 ensemble 구성에서 나온 점수인지 현재 자료만으로 검증할 수 없다. 수정본은 아직 다시 학습하지 않았으므로 이 점수를 수정된 코드의 결과로 제시하지 않는다.

지금 공개하는 자료에는 2025년 실험을 정리한 2026년 수정 노트북이 들어 있다. 당시의 시도와 이후의 코드 수정을 구분해서 기록했다.

## 자료

- [프로젝트 요약과 노트북](README.md)
- [코드 변경 기록](revision-notes.md)
- [MedMNIST](https://medmnist.com/)
- [Torchvision MobileNetV2](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v2.html)
