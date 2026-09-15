# 코드 안내

진행 중인 `our_star` 프로젝트에서 수치 모델과 분석 코드를 선별해 공개했습니다. 기존 모듈 이름을 유지했으며, 임상 데이터 수집·불러오기, 연구 자료 접근 관리, 논문 생성 도구, 비공개 결과 보관 자료는 포함하지 않았습니다.

## 주요 파일

| 경로 | 설명 |
| --- | --- |
| [models/segmented.py](src/our_star/models/segmented.py) | 위장관을 구간별로 나눈 PBPK 모델 |
| [models/parent_metabolite.py](src/our_star/models/parent_metabolite.py) | 과거 v0.3 계열에서 사용한 모약물–대사체 모델 |
| [virtual_trial.py](src/our_star/virtual_trial.py) | 상미분방정식(ODE) 적분, 투약 이벤트, 시간에 따른 상태 요약 |
| [models](src/our_star/models/), [chemistry](src/our_star/chemistry/) | 추가 모델 변형, 반응 네트워크, 물질량 수지 계산 |
| [population](src/our_star/population/), [prediction](src/our_star/prediction/) | 가상 인구집단 표본 추출과 예측 도구 |
| [analysis](src/our_star/analysis/), [comparators](src/our_star/comparators/) | 민감도, 식별 가능성, 매개변수 프로파일, 불확실성 분석 및 단순 구획 모델 |
| [configs](configs/) | 합성 예제와 공개 테스트에 필요한 과거 매개변수 설정 |
| [tests](tests/) | 수치적 보존, 모델 동작, 설정 유효성, 지표 검증 |

모델 변형에는 README에 설명한 과거 모델뿐 아니라 이후의 탐색적 개발 내용도 포함됩니다. 코드가 공개되어 있다는 사실이 임상적으로 검증되었다는 의미는 아닙니다. 소스의 매개변수에는 기존 출처와 해석 메모를 유지했습니다.

## 로컬 실행 환경

Python 3.11 이상을 사용해 이 프로젝트 폴더에서 실행합니다.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q
python examples/synthetic_pbpk.py
```

Windows에서는 PowerShell에서 `.venv\Scripts\Activate.ps1`로 가상환경을 활성화합니다. 예제는 가상 화합물 설정을 사용해 CPU에서 실행됩니다. 네트워크 연결이나 임상 관측값 입력 없이 질량수지와 상태값의 비음수성을 확인합니다.

## 보고된 결과와의 관계

[상세 평가 기록](docs/evaluation.md)과 [저장된 집계 결과](results/reviewed_results.json)는 과거 비교 실험을 설명합니다. 이번 공개 코드는 이후 개발 내용이 포함된 시점의 소스이며, 모든 실험을 독립적으로 재현한 배포본은 아닙니다.

예제와 테스트는 원본 임상 데이터셋 없이 실행할 수 있습니다. 수치 계산과 소프트웨어 동작을 확인하지만 사람 데이터의 평가 지표를 재현하지는 않습니다. 특히 합성 예제의 매개변수는 시연용 값이며, 실제 특정 약물의 추정값이 아닙니다. 원본 임상 관측 테이블과 과거 평가의 전체 실행 과정은 배포하지 않습니다.

모든 모델은 단순화한 연구용 모델입니다. 출력값만으로 환자별 용량, 유효성, 안전성을 판단할 수 없습니다.

[프로젝트로 돌아가기](README.md)
