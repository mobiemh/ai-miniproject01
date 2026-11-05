# ADR — 핵심 의사결정 기록

**프로젝트:** 모듈 1 ML/DL 기초 미니프로젝트 — 회귀·분류

> 실험 결과를 반복해서 옮겨 적지 않습니다. **선택 → 근거 → 한계**를 중심으로 각 항목을 1~2문장으로 작성합니다.
>
> 이 파일은 프로젝트 전체의 의사결정을 순서대로 쌓는 기록입니다. 1일차(회귀)는 의사결정 ①·②, 2일차(분류)는 ③·④를 아래에 이어서 작성합니다.

---

## 1일차 — 회귀 (공유자전거 대여 수요 예측)

### 의사결정 ① 입력 특성 선택 (CP-02)

**문제**

예측 시점에 사용할 수 없거나 누수가 발생할 수 있는 특성이 있습니다.

- **검토한 선택지:** casual, registered, atemp 제외 windspeed 포함  (예: `casual`·`registered` 포함 여부, `atemp` 포함 여부, `windspeed` 제거 여부)
- **최종 선택(사용·제외한 특성):** "hr",
    "season",
    "holiday",
    "weekday",
    "weathersit",
    "temp",
    "hum",
    "windspeed",
    "workingday"
- **선택 근거(누수 확인, 예측 시점, 상관관계 등):** casual, registered 는 cnt에 포함되므로 예측시점에 cnt가 유추되므로 데이터 누수, atemp는 temp와 상관관계가 높아서 제외
- **감수한 한계 또는 추후 확인할 점:** temp와 상관관계가 높은 atemp는 제외했는데 이렇게되면 atemp가 주는 미세한 영향은 예측에서 누락될수 있습니다.

**분석 근거:** `01_regression_data_analysis.md`

### 의사결정 ② 최종 회귀 모델 선택 (CP-07)

**문제**

서로 다른 회귀 모델 및 설정 중 하나를 최종 모델로 선택해야 합니다.

- **비교한 모델·설정:**   
DecisionTree max_depth 3,
DecisionTree max_depth 6,
DecisionTree max_depth 12,
RandomForest A {"n_estimators": 50, "max_depth": 10}, 
RandomForest B {"n_estimators": 100, "max_depth": 15},
LinearRegression,
DummyRegressor
- **최종 선택한 모델·설정:** RandomForest B {"n_estimators": 100, "max_depth": 15}
- **선택 근거(Validation MAE, 과대적합 가능성, 시간대별 오류 등):** Valid MAE가 가장 낮게 나왔고 다른 수치또한 다른 모델보다 전체적으로 좋게 나왔습니다, 검증-훈련 mae 차이가 비교적 큰편이라 과대적합 가능성이 있습니다. 훈련, 검증 시간대가 달라서 그 시간대의 영향을 받아 예측과 결과가 달라졌을수 있습니다.
- **선택 모델의 장점:** Valid MAE가 가장 작습니다.
- **남아 있는 한계 또는 개선 방향:** 학습할때 기간과 테스트의 기간이 달라서 예측결과에 영향을 주었을것입니다. 날짜 순서는 유지한채로 각 train, valid, test 데이터의 날짜 구간을 섞어서 개선을 해볼수 있겠습니다.

**분석 근거:** `01_regression_model_analysis.md`

---

## 2일차 — 분류 (온라인 쇼핑 구매 여부 예측 · PyTorch MLP)

### 의사결정 ③ 평가 지표 선택 (CP-04)

**문제**

구매 세션이 전체의 약 15%뿐인 불균형 데이터이므로, Accuracy만으로 구매 클래스를 얼마나 잘 예측하는지 판단하기 어렵습니다.

- **검토한 선택지:** ___ (예: Accuracy, 구매 클래스 F1, Precision, Recall)
- **최종 선택한 주 지표와 보조 지표:** ___
- **선택 근거(DummyClassifier와 LogisticRegression 결과, FP·FN 등):** ___
- **감수한 한계 또는 추후 확인할 점:** ___

**분석 근거:** `02_classification_data_analysis.md`, `02_classification_model_analysis.md`

### 의사결정 ④ PyTorch MLP 설정 및 분류 임계값 선택 (CP-07)

**문제**

기본 손실 함수로 학습한 MLP-A와 클래스 가중 손실을 적용한 MLP-B를 비교하고, 구매 확률을 0/1로 구분하는 임계값(0.3·0.5·0.7)을 선택해야 합니다.

- **검토한 선택지:** ___ (MLP-A / MLP-B의 학습 결과, 임계값 0.3·0.5·0.7)
- **최종 선택한 MLP 설정·임계값·재학습 에포크:** ___
- **선택 근거(Validation F1, Precision·Recall, FP·FN 및 가중 손실의 영향):** ___
- **LogisticRegression과 비교했을 때 선택한 MLP의 특징:** ___
- **감수한 한계 또는 추후 개선 방향:** ___

**분석 근거:** `02_classification_model_analysis.md`
