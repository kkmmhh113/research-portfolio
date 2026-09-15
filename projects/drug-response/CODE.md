# 코드 안내

진행 중인 `my_star` 프로젝트에서 공개할 소스 코드를 선별해 정리했습니다. 모델과 학습 코드, 저장된 실험 기록의 용어를 함께 살펴볼 수 있도록 기존 모듈 이름을 유지했습니다.

## 주요 파일

| 경로 | 설명 |
| --- | --- |
| [chemistry.py](src/my_star/chemistry.py) | 구조 표준화, 분자 지문, 분자 기술자, 화학적 골격 식별 키 |
| [model.py](src/my_star/model.py) | 과거 실험의 구조·처리 맥락·기저 발현 모델과 손실 함수 |
| [data.py](src/my_star/data.py), [splits.py](src/my_star/splits.py) | 데이터 준비, 스케일링, 처리 맥락별 사전 정보, 화합물·골격별 분할 |
| [prc_model.py](src/my_star/prc_model.py) | 섭동·처리 맥락·관측값을 분리하는 모델 변형 |
| [paper_a_metrics.py](src/my_star/paper_a_metrics.py) | 화합물 단위 지표와 불확실성 계산 도구. 과거의 관측 행 기준 가중 지표와 구분됨 |
| [scripts](scripts/) | 과거 학습·예측, 후보 확정, 평가 도구 |
| [tests](tests/) | 작은 예제 데이터를 이용한 특징 추출, 모델, 데이터 분할, 손실 함수, 지표 검증 |

`chemistry_v14`, `structure_contract_v14`, `control_fit_scope_v14`는 개발 중인 보조 모듈입니다. 이 파일들이 포함되어 있다고 해서 수정된 v1.4 벤치마크가 완성된 것은 아닙니다. 파일명의 Phase 번호는 내부 실험 단계를 뜻하며 임상시험 단계를 의미하지 않습니다. Phase 8 분석 도구는 과거 지표의 비교용으로 포함했으며, Phase 8의 성능을 주장하지 않습니다.

## 로컬 실행 환경

Python 3.11 이상을 사용해 이 프로젝트 폴더에서 실행합니다.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q
python examples/molecular_features.py --smiles 'CCO'
```

Windows에서는 PowerShell에서 `.venv\Scripts\Activate.ps1`로 가상환경을 활성화합니다. 예제는 분자 특징만 추출하며, 모델을 학습하거나 생물학적 반응을 예측하지 않습니다.

## 학습과 추론

[train_sciplex.py](scripts/train_sciplex.py)에는 과거 실험의 학습 반복 과정이, [train_prc.py](scripts/train_prc.py)에는 후속 분해형 모델의 학습 구현이 들어 있습니다. [predict_sciplex.py](scripts/predict_sciplex.py)는 체크포인트, SMILES, 처리 맥락, 기저 발현 벡터를 입력으로 받습니다. 데이터셋을 불러오지 않고도 다음 명령으로 인자 설명을 확인할 수 있습니다.

```bash
python scripts/train_sciplex.py --help
python scripts/train_prc.py --help
python scripts/predict_sciplex.py --help
```

[Phase 4 설정 파일](configs/sciplex_phase4_v2_final_adaptive.yaml)은 과거 실험 설정을 기록한 것입니다. 발현 테이블, 데이터 분할 파일, 학습된 가중치를 배포하지 않으므로 이 공개 코드만으로는 설정의 `data/` 경로가 연결되지 않습니다. 학습과 벤치마크 재현에는 버전이 일치하는 입력 자료를 별도로 준비해야 합니다. 현재 소스는 이후 개발 내용이 포함된 시점의 코드이며, 모든 과거 점수를 산출한 정확한 코드 버전이라고 주장하지 않습니다.

새 실험에 앞서 수정된 데이터의 정의와 처리 규칙을 확정하고, 모델 선택에는 검증 세트를 사용해야 합니다. 이미 결과를 확인한 최종 평가 세트를 새로운 독립 테스트로 재사용해서는 안 됩니다. 유전자 주석, 구조 매핑, 평가 세트 재사용에 따른 한계는 [상세 평가 기록](docs/evaluation.md)에 정리했습니다.

포함된 테스트는 작은 예제 데이터로 구현 동작을 확인합니다. 벤치마크 모델을 학습하거나 과거 공개 점수를 처음부터 끝까지 검증하는 테스트는 아닙니다.

[프로젝트로 돌아가기](README.md)
