# Validation Report — Fraud Early-Warning Model (Comparison)

**Val set:** val.parquet (labeled rows only, time-based 2018-2019)
**Champion:** CatBoost (PR-AUC=0.8538)

## Head-to-Head Comparison

| Metric | CatBoost | XGBoost |
|--------|--------|--------|
| PR-AUC | **0.8538** | 0.8470 |
| ROC-AUC | **0.9994** | 0.9992 |
| Brier (raw) | 0.000993 | **0.000573** |
| Brier (calibrated) | **0.000527** | 0.000555 |
| Recall@P90 | 0.6000 | **0.6426** |
| KS Statistic | 0.9769 | **0.9792** |
| ECE | 0.000001 | **0.000001** |
| F1 @ T_high (0.7) | 0.6938 | **0.7757** |
| F1 @ T_med (0.3) | **0.7972** | 0.7882 |
| Top-Decile Lift | **10.0** | **10.0** |

> **Bold** = best in column. Lower is better for Brier and ECE; higher is better for all others.

---

## CatBoost (Champion)

**Model file:** `catboost_fraud.cbm`
**Calibrator:** `outputs/models/calibrator_catboost.joblib`

### Metrics (raw)

| Metric | Value |
|--------|--------|
| PR-AUC | 0.8538 |
| ROC-AUC | 0.9994 |
| Brier | 0.000993 |
| Recall@P90 | 0.6000 |

### Calibration (isotonic)

- Brier before: 0.000993 → after: 0.000527

### Global Feature Importance (SHAP)

| Feature | Mean |SHAP| |
|---------|------|
| use_chip | 0.3912 |
| mcc_historical_fraud_rate | 0.3701 |
| time_since_last_tx_card | 0.2142 |
| first_time_merchant_user | 0.1703 |
| zip | 0.1613 |
| card_brand_historical_fraud_rate | 0.1554 |
| merchant_historical_fraud_rate | 0.1382 |
| merchant_state | 0.1125 |
| hour_of_day | 0.1069 |
| card_amount_sum_24h | 0.1013 |
| mcc_str | 0.0975 |
| amount | 0.0863 |
| mcc | 0.0615 |
| errors | 0.0420 |
| user_unique_mcc_24h | 0.0401 |

### Confusion Matrix @ T_high (0.7)

| | Predicted Fraud | Predicted Legit |
|---|---|---|
| **Actual Fraud** | 751 (TP) | 609 (FN) |
| **Actual Legit** | 54 (FP) | 775925 (TN) |

P=0.9329, R=0.5522, F1=0.6938

### Confusion Matrix @ T_med (0.3)

| | Predicted Fraud | Predicted Legit |
|---|---|---|
| **Actual Fraud** | 1093 (TP) | 267 (FN) |
| **Actual Legit** | 289 (FP) | 775690 (TN) |

P=0.7909, R=0.8037, F1=0.7972

### Threshold Counts

HIGH=805, MED=577, LOW=775957

### Sample Reason Codes (high-risk rows)

**Row 1** (calibrated proba=0.932):

  - first_time_merchant_user: 2.3715

  - merchant_state: 2.3118

  - merchant_city: 1.9128

  - mcc_str: 1.4767

  - use_chip: 0.5498

**Row 2** (calibrated proba=0.932):

  - merchant_state: 3.1842

  - merchant_city: 2.2565

  - mcc_historical_fraud_rate: 1.3056

  - mcc_str: 0.6638

  - use_chip: 0.4748

**Row 3** (calibrated proba=0.932):

  - first_time_merchant_user: 2.6825

  - merchant_state: 2.2272

  - merchant_city: 2.0251

  - mcc_str: 0.7573

  - use_chip: 0.5444

**Row 4** (calibrated proba=0.982):

  - merchant_state: 2.5104

  - first_time_merchant_user: 2.0103

  - merchant_city: 1.8287

  - mcc_str: 1.6061

  - mcc_historical_fraud_rate: 0.7190

**Row 5** (calibrated proba=0.932):

  - merchant_state: 3.1304

  - merchant_city: 2.4267

  - mcc_historical_fraud_rate: 1.2278

  - use_chip: 0.5704

  - mcc_str: 0.5607


---

## XGBoost

**Model file:** `xgboost_fraud.json`
**Calibrator:** `outputs/models/calibrator_xgboost.joblib`

### Metrics (raw)

| Metric | Value |
|--------|--------|
| PR-AUC | 0.8470 |
| ROC-AUC | 0.9992 |
| Brier | 0.000573 |
| Recall@P90 | 0.6426 |

### Calibration (isotonic)

- Brier before: 0.000573 → after: 0.000555

### Global Feature Importance (SHAP)

| Feature | Mean |SHAP| |
|---------|------|
| merchant_city | 3.1386 |
| mcc_str | 1.5436 |
| merchant_state | 0.9756 |
| merchant_historical_fraud_rate | 0.7582 |
| first_time_merchant_user | 0.6520 |
| hour_of_day | 0.5643 |
| amount | 0.5381 |
| mcc | 0.5347 |
| use_chip | 0.4654 |
| card_tx_count_7d | 0.4026 |
| first_time_merchant_card | 0.3778 |
| merchant_id | 0.3743 |
| time_since_last_tx_card | 0.3106 |
| card_amount_mean_7d | 0.2774 |
| zip | 0.2732 |

### Confusion Matrix @ T_high (0.7)

| | Predicted Fraud | Predicted Legit |
|---|---|---|
| **Actual Fraud** | 937 (TP) | 423 (FN) |
| **Actual Legit** | 119 (FP) | 775860 (TN) |

P=0.8873, R=0.6890, F1=0.7757

### Confusion Matrix @ T_med (0.3)

| | Predicted Fraud | Predicted Legit |
|---|---|---|
| **Actual Fraud** | 1042 (TP) | 318 (FN) |
| **Actual Legit** | 242 (FP) | 775737 (TN) |

P=0.8115, R=0.7662, F1=0.7882

### Threshold Counts

HIGH=1056, MED=228, LOW=776055

### Sample Reason Codes (high-risk rows)

**Row 1** (calibrated proba=1.000):

  - merchant_city: 3.0518

  - first_time_merchant_user: 1.4380

  - merchant_historical_fraud_rate: 1.2988

  - merchant_state: 1.2136

  - first_time_merchant_card: 1.0135

**Row 2** (calibrated proba=1.000):

  - merchant_city: 3.4103

  - merchant_state: 1.4923

  - merchant_id: 1.4441

  - credit_limit: 1.1455

  - mcc_str: 1.1441

**Row 3** (calibrated proba=1.000):

  - merchant_city: 2.9050

  - first_time_merchant_user: 1.4669

  - merchant_historical_fraud_rate: 1.3271

  - merchant_state: 1.0967

  - mcc: 0.9062

**Row 4** (calibrated proba=1.000):

  - merchant_city: 3.3012

  - mcc_str: 1.6919

  - merchant_id: 1.5326

  - mcc: 1.2068

  - first_time_merchant_user: 1.1037

**Row 5** (calibrated proba=0.983):

  - merchant_city: 3.2557

  - mcc: 2.6702

  - use_chip: -1.3954

  - merchant_state: 0.9758

  - amount: 0.8038


