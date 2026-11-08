# 004. 모델 실험 및 오류 분석

**프로젝트:** 온라인 쇼핑 구매 여부 예측 (PyTorch 신경망 분류)  
**작성 범위:** CP-04~CP-08

> Notebook에서 확인한 실제 수치와 관찰을 기록합니다. `___`는 실제 값 또는 1~2문장으로 바꾸세요. Precision·Recall·F1은 구매 클래스(`Revenue=1`) 기준입니다.

## 1. 기준 모델 및 평가 지표 (CP-04)

- **양성 클래스:** 구매 (`Revenue=1`)
- **주 평가 지표:** F1 (구매 클래스)
- **보조 지표:** Precision, Recall, Accuracy
- **모델 비교에 사용한 데이터:** Validation 데이터
- **DummyClassifier — Validation Accuracy / F1:** 0.845 / 0.000
- **LogisticRegression — Train F1 / Validation F1:** 0.505 / 0.490
- **LogisticRegression — Validation Precision / Recall:** 0.734 / 0.367
- **LogisticRegression의 Validation 혼동행렬:** TN 1526 / FP 38 / FN 181 / TP 105

**분석:** 불균형 데이터에서 기준 모델의 Accuracy와 F1이 다르게 나타나는 이유를 작성합니다.  
불균형 데이터에서는 데이터를 이루는 구매여부 비율에 따라서 정확도가 높게 나올수 있으나, F1은 TN은 제외하므로  미구매가 많아도 점수가 부풀려지지 않습니다.

## 2. PyTorch MLP-A 구현 및 학습 (CP-05)

- **신경망 구조:** 입력 → 64 → 32 → 1 (ReLU, Dropout 0.1)
- **손실 함수 / 옵티마이저:** `BCEWithLogitsLoss` / Adam
- **학습 조건:** 최대 25 epoch, batch size 256, learning rate 0.001
- **Validation F1이 가장 높았던 에포크:** 25
- **선택된 MLP-A의 Train F1 / Validation F1:** 0.709 / 0.633
- **MLP-A Validation Precision / Recall:** 0.717 / 0.566

**분석:** Train loss와 Train·Validation F1 그래프에서 확인한 학습 경향을 작성합니다.  
Train loss 그래프에서는 손실이 초반부터 급격히 줄어드는것으로 보아 잘 학습이 된 것 같습니다. F1 그래프를 보면 Validation 그래프가 에포크가 증가할수록 Train 그래프처럼 완만하게 증가하면서 그래프간 거리가 점점 벌어짐을 볼 수 있습니다. 최대 에포크 25에서는 위로 올라가는것을 보아 에포크가 더 증가할수록 개선될 수도 있어보이나 그래프가 오르락내리락 하므로 단정하기 어렵습니다.

## 3. 클래스 가중 손실과 MLP 비교 (CP-06)

- **MLP-B 변경 사항:** 양성 클래스 가중치(`pos_weight`)를 적용한 `BCEWithLogitsLoss`
- **Train 데이터의 `pos_weight`(미구매 수 ÷ 구매 수):** 5.46 (미구매 7,294 ÷ 구매 1,336)
- **MLP-B의 최적 에포크:** 17
- **MLP-B Train F1 / Validation F1:** 0.692 / 0.645
- **MLP-B Validation Precision / Recall:** 0.575 / 0.734
- **MLP-A와 MLP-B의 Validation F1 차이:** 0.012
- **LogisticRegression과 두 MLP의 성능 비교에서 발견한 점:** 두 MLP보다 LogisticRegression의 F1과 Recall이 낮다
- **Train·Validation F1 차이에서 관찰한 점:** 차이가 크지 않다.MLP-A가 0.077로 차이가 가장 크다

**분석:** 양성 클래스 가중치를 적용한 뒤 Precision·Recall·F1이 어떻게 달라졌는지 작성합니다.  
F1은 거의 변화가 없었고 가중치 적용 뒤 Precision이 컸으나 작아졌고 Recall이 작았으나 커졌다

> 전체 실험 결과는 Notebook의 `results_df`와 `results/02_model_comparison.csv`를 참고합니다.

## 4. 예측 오류 및 임계값 분석 (CP-07)

- **비교한 두 신경망:** MLP-A / MLP-B
- **기본 임계값 0.5에서 MLP-A의 FP / FN:** 64 / 124
- **기본 임계값 0.5에서 MLP-B의 FP / FN:** 155 / 76
- **임계값 0.3에서 Precision·Recall 및 FP·FN 변화:** Precision 0.608→0.459 감소, Recall 0.678→0.839 증가, FP 125→283 증가, FN 92→46 감소
- **임계값 0.7에서 Precision·Recall 및 FP·FN 변화:** Precision 0.835→0.639 감소, Recall 0.353→0.612 증가, FP 20→99 증가 FN 185→111 감소
- **Validation F1이 가장 높은 MLP·임계값 조합:** MLP-B 0.5
- **선택한 MLP의 최적 에포크 수:** 17
- **(선택) PR 곡선 또는 추가 임계값 실험에서 확인한 점:** ___

**분석:** 모델과 임계값의 선택에서 Precision·Recall 및 FP·FN을 어떻게 고려했는지 작성합니다.  
F1이 가장 높은 모델로 먼저 정했고 다른 보델과 비교했을때 Precision·Recall 및 FP·FN이 유난히 값이 튀지 않는 모델로 최종 선택했습니다.

> 임계값별 상세 결과는 `threshold_df`와 `results/02_threshold_comparison.csv`를 참고합니다.

## 5. 최종 신경망과 Test 평가 (CP-08)

- **Test 확인 전에 확정한 MLP 설정 / 임계값:** MLP-B (weighted BCE) / 0.5
- **최종 재학습 데이터(Train + Validation):** 10480행
- **최종 재학습 에포크 수 / 가중 손실 적용 여부:** 17 / 적용함
- **Test Accuracy / Precision / Recall / F1:** 0.858 / 0.526 / 0.843 / 0.648
- **Test 혼동행렬:** TN 1347 / FP 217 / FN 45 / TP 241
- **모델 번들(`02_final_classifier.joblib`)·설정 JSON 저장 및 재로딩 확인:** 저장 전후 예측 일치함
- **`predict_purchase()` 실행 결과:** {'purchase_probability': 0.0058, 'prediction': 0, 'threshold': 0.5}

**해석:** 최종 Test 결과와 동일한 모델·임계값의 Validation 결과를 비교해 알게 된 점을 작성합니다.  
F1은 0.648와 0.645로 비슷한것로 보아 모델 성능이 비숫합니다. Recall이 0.843에서 0.734로 다소 많이 줄었고 Precision이 0.526에서 0.575로 약간 늘었습니다. 이유는 Train+Validation 데이터로 다시 학습하면서 모델 가중치(입력에 곱해서 구매 확률을 계산하는 신경망 안의 숫자)가 달라졌고 평가데이터도 Test로 바뀌었기때문입니다.

> Test 결과는 모델 선택에 사용하지 않습니다. 최종 Test 수치는 Notebook과 이 문서에만 기록합니다.

## 6. 실험 요약

- **가장 중요한 실험 결과:** 데이터에서 차지하는 구매여부 비중에 따라 제대로 예측이 되지 않으므로 구매여부에 가중치를 주었고 다양한 임계값을 준 여러 모델 지표값을 비교하여 모델을   F1이 가장 높은 MLP-B로 선택했습니다. F1이 Test에서도 Vlidation과 비슷한 값을 보여 성능이 비슷하게 보였습니다.

- **남아 있는 모델의 한계:** 선택된 모델의 Precision 수치는 Test 0.526, Validation 0.575 정도로 높은편이 아닙니다. 실제로는 구매하지 않을 세션을 구매로 잘못 예측하는 경우(FP)가 많습니다.

> **최종 MLP 설정·임계값의 선택과 근거**는 `adr.md`의 의사결정 ④에 작성합니다.
