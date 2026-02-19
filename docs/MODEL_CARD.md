# Model Card — Fraud Early-Warning Classifier

**Model type:** CatBoost binary classifier
**Model file:** `catboost_fraud.cbm`
**Purpose:** Score each transaction with fraud probability; support alert levels (HIGH/MEDIUM/LOW) and reason codes.

## 1. Training data

- **Train:** time-based, date <= 2017-12-31 (~7.2M labeled rows)
- **Target:** is_fraud (0/1); only labeled rows (has_label=True)
- **Features:** 99 total (9 categorical)
- **Categorical encoding:** Native CatBoost handling

## 2. Metrics (validation, 2018-2019)

- **PR-AUC:** 0.8538 (primary)
- **ROC-AUC:** 0.9994
- **Brier:** 0.000993 (raw), 0.000527 (calibrated)
- **Recall@P90:** 0.6000
- **KS Statistic:** 0.9769
- **ECE (calibrated):** 0.000001

## 3. Limitations

- Trained on 2010-2017; validated on 2018-2019. Performance may drift on future data.
- Trained on 3M stratified subsample (all fraud + sampled non-fraud) due to RAM constraints; full 7.2M labeled rows available.
- Class imbalance: fraud rate ~0.15%; PR-AUC and Recall@P90 are primary metrics.
- Behavioral features included: rolling velocity/monetary windows (10m/1h/24h/7d), recency, novelty, z-score. Top SHAP: time_since_last_tx_card, first_time_merchant_user.

## 4. Usage

- Load model: `catboost.CatBoostClassifier(); model.load_model("outputs/models/catboost_fraud.cbm")`
- Load calibrator: `joblib.load("outputs/models/calibrator_catboost.joblib")`
- Apply calibrator for calibrated scores, compare to T_high/T_med for alert level
- SHAP reason codes: `get_global_importance()` / `get_local_reason_codes()` from `src.modeling.explain`
