# Minhyoung Kim

Research Portfolio

[Intro](#intro) · [Projects](#projects) · [Contact](#contact)

---

## Intro

현재 건국대학교 융합생명공학과에 재학 중이며, 2027년 2월 졸업 예정입니다.

생명공학을 공부하면서 유전자 발현과 약물 반응을 데이터로 살펴보는 데 관심을 갖게 되었습니다. 「파이썬을 활용한 데이터 분석」과 「인공지능의 이해」 수업을 들었고, 이번 학기에는 「딥러닝의 이해」를 수강하며 역전파 계산을 배우고 있습니다. Stanford CS229 머신러닝 강의와 MIT 18.06 선형대수학 강의도 병행하고 있습니다.

논문을 읽다가 생긴 궁금증을 작은 개인 프로젝트로 시도해 보고 있습니다. 지금은 분자 구조를 이용한 유전자 발현 변화 예측과 조직별 약물 이동 모델링을 공부하고 있습니다. 장기적으로는 개인의 생물학적 차이를 반영한 약물 반응 예측과, 전임상 데이터를 사람의 치료 반응 예측에 활용하는 연구를 하고 싶습니다.

## Projects

### 01. Mol2Expr — 분자 구조 기반 발현 변화 예측

> 분자 구조와 약물 처리 조건으로 유전자 발현 변화를 예측하는 방법을 살펴봅니다.

**2026–현재 · 진행 중**

LPM 논문에서 학습에 없는 새로운 화합물을 표현하는 문제에 관심이 생겨, 분자 구조를 입력하는 작은 모델을 시도했습니다. 시작한 이유와 예시 코드, 아직 확인하지 못한 부분을 정리했습니다.

`Python` · `PyTorch` · `RDKit`

[프로젝트 소개 →](projects/drug-response/README.md) · [모델 설명](projects/drug-response/MODEL.md) · [핵심 코드](projects/drug-response/model_sketch.py)

---

### 02. TissuePK — 조직별 약물 이동 모델링

> 여러 조직을 단순화한 구획 모델로 약물의 이동과 제거 과정을 표현합니다.

**2026–현재 · 진행 중**

약물의 대사와 체내 이동을 수학적으로 설명한 논문을 읽고, 단순화한 PBPK 모델의 농도 계산을 공개 관측값과 비교해 보았습니다. 모델의 가정과 결과를 이해하며 공부하는 과정을 정리했습니다.

`Python` · `NumPy` · `SciPy`

[프로젝트 소개 →](projects/tissue-modeling/README.md) · [모델 설명](projects/tissue-modeling/MODEL.md) · [핵심 코드](projects/tissue-modeling/model_sketch.py)

---

### 03. DermaMNIST 이미지 분류 모델

> MobileNetV2를 미세 조정해 피부 병변 이미지를 7개 범주로 분류합니다.

**2025 · 모델 학습 경험 · 2026년 코드 정리**

첫 딥러닝 프로젝트에서 학습 설정을 바꾸고, 체크포인트 앙상블과 Optuna를 사용해 보았습니다. 모델을 선택한 이유와 학습 과정, 배운 점, 코드를 다시 정리하며 발견한 문제를 기록했습니다.

`Python` · `PyTorch` · `Torchvision` · `Optuna`

[프로젝트 소개 →](projects/dermamnist/README.md) · [소스 코드](projects/dermamnist/notebooks/)

---

## Contact

**이메일** · [vlsi71@konkuk.ac.kr](mailto:vlsi71@konkuk.ac.kr)<br>
**전화번호** · [010-2202-8414](tel:+821022028414)<br>
