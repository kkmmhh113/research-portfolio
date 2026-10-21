<div align="center">

# 김민형 · 연구 포트폴리오

**유전자 발현 · 약물 반응 · 생리학 기반 모델링**

건국대학교 융합생명공학과 · 2027년 2월 졸업 예정

[소개](#소개) &nbsp; · &nbsp; [프로젝트](#프로젝트) &nbsp; · &nbsp; [연락처](#연락처)

</div>

---

## 소개

생명공학을 공부하며, 생물학적 차이를 데이터로 이해하고 약물 반응을 예측하는 연구에 관심을 갖게 되었습니다. 논문을 읽다가 생긴 궁금증을 작은 개인 프로젝트로 옮겨 보며 모델의 구조와 계산, 평가 방법을 공부하고 있습니다.

현재는 분자 구조를 이용한 유전자 발현 변화 예측과 조직별 약물 이동 모델링을 시도하고 있습니다. 장기적으로는 개인의 특성을 반영한 약물 반응 예측과, 전임상 데이터를 사람의 치료 반응 예측에 활용하는 연구를 하고 싶습니다.

각 프로젝트에는 시작한 이유와 시도한 과정, 이어서 공부하고 싶은 내용을 담았습니다. 프로젝트의 세부 모델 구성과 코드 구현·점검·수정, 결과 분석에는 AI 도구를 활용했습니다.

<details>
<summary>학습 배경</summary>

「파이썬을 활용한 데이터 분석」과 「인공지능의 이해」를 수강했습니다. 2026년 2학기에는 「딥러닝의 이해」에서 학습 원리와 역전파 계산을 배우고 있으며, Stanford CS229 머신러닝 강의와 MIT 18.06 선형대수학 강의도 병행하고 있습니다.

</details>

## 프로젝트

### 01 &nbsp; Mol2Expr

**분자 구조 기반 유전자 발현 변화 예측**<br>
2026–현재 · 개인 프로젝트 진행 중

학습에서 보지 못한 화합물의 반응을 예측할 때, 분자 구조가 도움이 될지 궁금해 시작했습니다. SciPlex3 데이터를 이용해 약물 처리에 따른 조건별 평균 발현 변화를 예측해 보았습니다.

**다음에 살펴볼 점:** 학습에 없는 화합물에서 분자 구조 정보의 유무에 따른 예측 결과 비교

**사용한 도구:** `Python` `PyTorch` `RDKit`

**[프로젝트 읽기 →](projects/drug-response/README.md)** &nbsp; · &nbsp; [모델과 코드 설명](projects/drug-response/MODEL.md) &nbsp; · &nbsp; [코드 예시](projects/drug-response/model_sketch.py)

---

### 02 &nbsp; TissuePK

**조직별 약물 이동 모델링**<br>
2026–현재 · 개인 프로젝트 진행 중

약물이 몸 안에서 이동하고 제거되는 과정을 계산해 보고 싶어 시작했습니다. 장·간·신장 등을 연결한 단순화된 생리학 기반 약동학(PBPK) 모델의 농도 곡선을 PK-DB의 사람 약동학 자료와 비교했습니다.

**다음에 살펴볼 점:** 관측 농도와 차이가 나는 시간대, 그 차이에 영향을 주는 가정과 매개변수

**사용한 도구:** `Python` `NumPy` `SciPy`

**[프로젝트 읽기 →](projects/tissue-modeling/README.md)** &nbsp; · &nbsp; [모델과 코드 설명](projects/tissue-modeling/MODEL.md) &nbsp; · &nbsp; [코드 예시](projects/tissue-modeling/model_sketch.py)

---

### 03 &nbsp; DermaMNIST

**피부 병변 이미지 분류**<br>
2025 · 모델 학습 경험 / 2026 · 코드 정리

첫 딥러닝 프로젝트로 MobileNetV2를 미세 조정해 피부 병변 이미지를 7개 범주로 분류해 보았습니다. 학습 설정을 바꾸고, 체크포인트 앙상블과 Optuna 탐색을 시도하며 결과를 살펴보았습니다.

**다음에 살펴볼 점:** 같은 조건에서 학습 설정을 하나씩 바꿨을 때의 차이

**사용한 도구:** `Python` `PyTorch` `Torchvision` `Optuna`

**[프로젝트 읽기 →](projects/dermamnist/README.md)** &nbsp; · &nbsp; [학습 과정](projects/dermamnist/project-notes.ko.md) &nbsp; · &nbsp; [소스 코드](projects/dermamnist/notebooks/)

---

## 연락처

[vlsi71@konkuk.ac.kr](mailto:vlsi71@konkuk.ac.kr) &nbsp; · &nbsp; 010-2202-8414
