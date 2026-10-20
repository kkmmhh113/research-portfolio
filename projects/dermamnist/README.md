[포트폴리오](../../README.md) / DermaMNIST

# DermaMNIST

### 피부 병변 이미지 분류

2025 · 개인 프로젝트 / 2026 · 코드 정리

**프로젝트 소개** &nbsp; · &nbsp; [학습 과정](project-notes.ko.md) &nbsp; · &nbsp; [기본 학습 코드](notebooks/dermamnist_baseline.ipynb) &nbsp; · &nbsp; [Optuna 코드](notebooks/dermamnist_optuna.ipynb)

---

| 항목 | 내용 |
| --- | --- |
| 학습 주제 | 사전 학습 모델을 미세 조정하고 학습 설정에 따른 결과 비교 |
| 데이터 | DermaMNIST 피부 병변 이미지 · 7개 범주 |
| 모델 | ImageNet 사전 학습 MobileNetV2 |
| 현재 상태 | 학습 경험 정리 · 수정한 코드의 재학습은 미진행 |

## 시작한 이유

딥러닝을 처음 공부할 때 이미지 분석에 관심이 생겨, 공개 데이터로 직접 분류 모델을 학습시켜 보고 싶었습니다. DermaMNIST를 선택하고, 제한된 연산 자원에서 여러 설정을 시도하기 위해 비교적 가벼운 MobileNetV2를 사용했습니다.

## 시도한 내용

사전 학습 모델의 마지막 분류층을 7개 범주에 맞게 바꾸고 전체 모델을 미세 조정했습니다. AdamW로 가중치를 갱신하고 스케줄러로 학습률을 조절했으며, 데이터 증강·클래스 가중치를 적용한 Focal Loss·Mixup도 사용했습니다.

학습 중 검증 손실이 가장 낮은 시점과 정확도가 가장 높은 시점이 항상 같지는 않았습니다. 서로 다른 에포크에 저장한 체크포인트의 출력을 합치는 앙상블을 시도했고, 학습률과 가중치 감쇠를 탐색하기 위해 Optuna도 사용했습니다.

| 구성 | 적용한 방법 |
| --- | --- |
| 기본 학습 | MobileNetV2 미세 조정 + AdamW + CosineAnnealingLR |
| 체크포인트 앙상블 | 같은 모델 구조의 서로 다른 체크포인트에서 나온 로짓 평균 |
| Optuna 탐색 | 특징 추출부 학습률·분류층 학습률·가중치 감쇠 탐색 |

## 확인한 점

당시에는 앙상블과 하이퍼파라미터 조정으로 큰 성능 향상을 얻지 못했던 것으로 기억합니다. 비교 기록과 Optuna 탐색 결과가 충분히 남아 있지 않아 정확한 개선 폭을 다시 확인하기는 어렵습니다.

코드를 다시 정리하며 체크포인트 순위를 관리하는 문제를 발견해 저장 방식을 수정했습니다. 당시 체크포인트 파일은 남아 있지 않고 수정한 코드로 다시 학습하지도 않았으므로, 과거의 점수를 현재 코드의 성능으로 제시하지 않습니다.

**모델을 학습시키는 것뿐 아니라, 무엇을 비교하고 어떤 가중치로 평가했는지를 남기는 일도 중요하다는 점을 배웠습니다.** 예측·손실 계산·가중치 갱신·검증으로 이어지는 학습 과정을 접한 첫 프로젝트였습니다.

## 학습 코드

- [기본 학습 코드](notebooks/dermamnist_baseline.ipynb) — 데이터 준비, 미세 조정, 체크포인트 선택, 최종 평가
- [Optuna 탐색 코드](notebooks/dermamnist_optuna.ipynb) — 하이퍼파라미터 탐색과 선택한 설정의 평가
- [학습 과정](project-notes.ko.md) — 앙상블을 시도한 이유와 비교 결과를 해석할 때 남은 한계

실행은 각 코드의 플래그로 제어합니다. 정규화 값의 출처 등 재학습 전에 확인할 내용도 코드에 표시했습니다.

## 참고 자료

[MedMNIST](https://medmnist.com/) &nbsp; · &nbsp; [Torchvision MobileNetV2](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v2.html)

---

[← 포트폴리오](../../README.md) &nbsp; · &nbsp; [학습 과정 →](project-notes.ko.md)
