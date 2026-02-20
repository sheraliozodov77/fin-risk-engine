# Business Logic & ML Domain Knowledge

## Business Context

**Industry:** Banking, Treasury, Financial Services
**Problem:** Real-time fraud detection and transaction risk scoring
**Dataset:** CaixaBank Tech 2024 AI Hackathon — ~13.3M transactions across a decade (2010s)
**Regulatory context:** Treasury-grade systems must comply with SR 11-7 (Model Risk Management), Basel III/IV capital requirements, PSD2 (Strong Customer Authentication), and AML/KYC directives

---

## Core Business Objective

Score every financial transaction with a calibrated fraud probability and route it through a 3-tier decision policy:

| Risk Level | Threshold | Action | Business Impact |
|------------|-----------|--------|-----------------|
| **HIGH** | proba ≥ 0.70 | Block / hold / manual review | Prevents confirmed fraud; delays legitimate tx |
| **MEDIUM** | 0.30 ≤ proba < 0.70 | Step-up authentication / watchlist | Adds friction; catches borderline cases |
| **LOW** | proba < 0.30 | Allow (auto-approve) | No friction; possible missed fraud |

The system outputs:
1. **Risk score** — calibrated P(fraud) from 0 to 1
2. **Alert level** — HIGH / MEDIUM / LOW per threshold policy
3. **Reason codes** — top-k SHAP feature contributions explaining why

---

## Data Model (Star Schema)

```
TRANSACTIONS (fact, ~13.3M rows)
    |-- card_id       --> CARDS (~6,146 rows)
    |                       |-- client_id --> USERS (~2,000 rows)
    |-- client_id     --> USERS
    |-- mcc (string)  --> MCC_CODES (109 codes)
    |-- id (string)   --> FRAUD_LABELS (~8.9M labeled, ~13.3K fraud)
```

**Key Business Rules:**
- **Partial labeling:** Only ~67% of transactions have fraud labels; unlabeled = UNKNOWN (never treated as non-fraud)
- **Fraud rate:** ~0.15% among labeled transactions — extreme class imbalance
- **Negative amounts:** Present (`$-77.00`) — represent refunds/chargebacks, valid data
- **PII exclusion:** `card_number`, `address`, `cvv` never used as model features

---

## Feature Engineering Logic

### Static Features (44 total)

| Feature | Source | Logic |
|---------|--------|-------|
| `amount` | transactions | Parse `$` strings to float |
| `hour_of_day` | date | 0-23 |
| `day_of_week` | date | 0=Mon, 6=Sun |
| `is_weekend` | day_of_week | 1 if Sat/Sun |
| `is_night` | hour_of_day | 1 if 00:00-05:00 |
| `month` | date | 1-12 |
| `has_chip` | cards | YES/NO → 0/1 |
| `card_on_dark_web` | cards | No/Yes → 0/1 |
| `credit_limit` | cards | Parse `$` |
| `acct_age_days` | date − acct_open_date | Account maturity |
| `per_capita_income` | users | Parse `$` |
| `use_chip` | transactions | Chip/swipe/online |

### Behavioral Features (55 total) — Leakage-Safe

All behavioral features use **strictly past-only data** (transactions strictly before current tx time). This prevents target leakage — the most common mistake in fraud model development.

**Rolling windows per entity (card_id and client_id):**

| Window | Features |
|--------|----------|
| 10 min | tx_count, amount_sum, amount_mean, amount_max, unique_merchants, unique_mcc |
| 1 hour | same |
| 24 hours | same |
| 7 days | same |

**Recency:**
- `time_since_last_tx_card` — seconds since cardholder's last tx (**#3 SHAP for CatBoost**)
- `time_since_last_tx_user` — seconds since user's last tx

**Novelty:**
- `first_time_merchant_card` / `first_time_merchant_user` — never transacted at this merchant before (**#4 SHAP for CatBoost, #5 for XGBoost**)
- `first_time_mcc_card` / `first_time_mcc_user` — never used this MCC before

**Deviation:**
- `z_score_amount` — standard deviations from user's 7d rolling mean; high z-score = unusual amount

### Risk Aggregate Features (3 total) — Bayesian Smoothed

```
fraud_rate(entity, t) = (fraud_count_before_t + alpha) / (total_count_before_t + beta)
```

alpha=1.0, beta=10.0 (smoothing prevents extreme rates for rare entities)

| Feature | Entity | SHAP Rank |
|---------|--------|-----------|
| `mcc_historical_fraud_rate` | mcc_str | **#1–2** in both models |
| `merchant_historical_fraud_rate` | merchant_id | **#4** in XGBoost |
| `card_brand_historical_fraud_rate` | card_brand | Top-10 |

**Why Bayesian smoothing matters:** A merchant with 1 fraud out of 1 transaction should not show 100% fraud rate. Smoothing regularizes toward the global mean, critical for decision-making on rare entities.

---

## Modeling Strategy

### Multi-Model Approach — Results on Held-Out Test Set (2020)

| Metric | CatBoost (Champion) | XGBoost | Winner |
|--------|---------------------|---------|--------|
| **PR-AUC** | **0.8538** | 0.8470 | CatBoost |
| ROC-AUC | **0.9994** | 0.9992 | CatBoost |
| Brier (raw) | 0.000993 | **0.000573** | XGBoost |
| Brier (calibrated) | **0.000527** | 0.000555 | CatBoost |
| Recall@P90 | 0.600 | **0.643** | XGBoost |
| KS Statistic | 0.977 | **0.979** | Tie |
| ECE | 0.000001 | 0.000001 | Tie |
| F1 @ T_high (0.7) | 0.694 | **0.776** | XGBoost |
| F1 @ T_med (0.3) | **0.797** | 0.788 | CatBoost |

> Both models validated on truly held-out 2020 test data (777K rows, 1,360 fraud). Val → Test degradation is near zero, confirming no overfitting to the validation set.

### CatBoost (Champion)

- Native categorical handling (no encoding needed for 100K+ merchants)
- Built-in SHAP (required for SR 11-7 regulatory explainability)
- Best calibrated probability output (Brier=0.000527 after isotonic)
- Best at medium-threshold decision making (F1@T_med=0.797)
- **Tuned params** (Optuna 20 trials, 3M subsample with behavioral):
  - `iterations=843`, `lr=0.049`, `depth=9`, `l2_leaf_reg=8.13`, `border_count=141`, `scale_pos_weight=8.34`
  - Best val PR-AUC: 0.8517

### XGBoost

- Requires **Bayesian smoothed target encoding** for categoricals (not LabelEncoder — LabelEncoder loses ~13% PR-AUC)
- Better at high-precision operation (Recall@P90=0.643, F1@T_high=0.776)
- Best raw probability calibration (Brier_raw=0.000573)
- **Tuned params** (Optuna 10 trials, 3M subsample with behavioral):
  - `n_estimators=895`, `lr=0.154`, `max_depth=8`, `reg_lambda=6.41`, `subsample=0.706`, `colsample_bytree=0.563`, `scale_pos_weight=15.85`, `min_child_weight=14`, `gamma=0.829`
  - Best val PR-AUC: 0.8471

### LightGBM

- Native categorical support via `.astype("category")`
- OOM on Mac with 10.7M rows — skip in benchmark or use reduced data

### Training Protocol

1. **Data:** Labeled rows only (`has_label == True`). ~7.2M train, ~935K val, ~777K test
2. **Target:** `is_fraud` (binary 0/1)
3. **Split:** Strictly time-based — train ≤ 2017-12-31, val 2018-2019, test ≥ 2020
4. **Subsampling:** `--subsample-train 3000000` — keep ALL fraud + sample non-fraud to 3M total. Required on <16GB RAM. Uses pyarrow filter pushdown + gc.collect() to avoid memory spikes
5. **Missing handling:** numeric → -999, categorical → `_fill_cat()` (centralized handler, handles pandas 3.0 StringDtype)
6. **Tuning:** Optuna with data precomputed once per study (encode/pool before trials, not inside each trial)
7. **scale_pos_weight:** Must retune when subsampling changes class ratio. At 3M subsample (1:289 ratio): CatBoost ~8, XGBoost ~16. At full 7.2M (1:696 ratio): CatBoost ~21, XGBoost ~2.5

### Optimization History

**CatBoost:**

| Stage | PR-AUC | Dataset | Key Finding |
|-------|--------|---------|-------------|
| Defaults | 0.649 | 7.2M, no behavioral | Overfit at iter 56 — always tune |
| Optuna 10 trials | 0.841 | 7.2M, no behavioral | High SPW (~21) helps |
| Optuna 20 trials | 0.859 | 7.2M, no behavioral | depth=7, lr=0.30 optimal |
| Retune 20 trials | **0.852** | 3M, with behavioral | depth=9, lr=0.049, SPW drops to 8.34 |

**XGBoost:**

| Stage | PR-AUC | Key Finding |
|-------|--------|-------------|
| LabelEncoder | 0.640 | Arbitrary integers, no signal |
| Bayesian target encoding | 0.773 | +20.8% — encoding is critical |
| VIF feature selection (44→31) | 0.722 | WORSE — VIF hurts tree models; keep all features |
| Optuna 10 trials (no behavioral) | 0.817 | Low LR (0.03), moderate SPW |
| Retune 10 trials (behavioral) | **0.847** | High LR (0.154), deep tree (8), SPW=15.85 |

### Probability Calibration

- **Method:** Isotonic regression (non-parametric, fits fraud data better than Platt)
- **Effect:** Brier score reduced by 40-50% for CatBoost; 3% for XGBoost (XGBoost already well-calibrated)
- **Storage:** `calibrator_catboost.joblib`, `calibrator_xgboost.joblib`; `calibrator.joblib` = champion copy for serving API
- Calibrated probabilities are used for all threshold comparisons and reason code reporting

---

## Explainability (SHAP)

### Global Feature Importance — CatBoost (Test Set)

| Rank | Feature | Mean |SHAP| | Business Interpretation |
|------|---------|------|-------------------------|
| 1 | `use_chip` | 0.391 | Swipe/online transactions are highest-risk channel |
| 2 | `mcc_historical_fraud_rate` | 0.370 | Category-level fraud history is strongest historical signal |
| 3 | `time_since_last_tx_card` | 0.214 | Rapid successive transactions signal fraud |
| 4 | `first_time_merchant_user` | 0.170 | Novel merchant = elevated risk |
| 5 | `zip` | 0.161 | Geographic anomaly (out-of-territory) |

### Global Feature Importance — XGBoost (Test Set)

| Rank | Feature | Mean |SHAP| | Business Interpretation |
|------|---------|------|-------------------------|
| 1 | `merchant_city` | 3.139 | City-level geographic risk is dominant signal |
| 2 | `mcc_str` | 1.544 | Merchant category code (raw) |
| 3 | `merchant_state` | 0.976 | State-level geographic concentration |
| 4 | `merchant_historical_fraud_rate` | 0.758 | Merchant-specific fraud history |
| 5 | `first_time_merchant_user` | 0.652 | Confirmed by both models |

**Key insight:** Behavioral features (`time_since_last_tx_card`, `first_time_merchant_user`) appear in top-5 for both models, confirming they add genuine predictive value beyond static features.

### Local Reason Codes (Per-Transaction)

Each HIGH/MEDIUM alert includes top-5 SHAP values:
- **Positive SHAP** → pushes score toward fraud
- **Negative SHAP** → pushes score toward legitimate
- Example: `merchant_city = "LOS ANGELES" → SHAP +3.14 → Strongly increased risk`

**Regulatory requirement:** Reason codes are mandatory for SR 11-7 compliance and customer-facing fraud alerts in Treasury.

---

## Decision Policy

### Threshold Economics

Two costs to balance:
1. **False negative cost** — missed fraud (chargebacks, customer loss, regulatory fines)
2. **False positive cost** — blocking legitimate transactions (customer friction, revenue loss)

**Current thresholds (T_high=0.7, T_med=0.3) on test set:**
- HIGH alerts: ~805–1,056 transactions (block/review queue)
- MEDIUM alerts: ~228–577 transactions (step-up auth)
- LOW (allowed): ~775K–776K transactions

### Alert Routing

```
Transaction → Score → Calibrate → Threshold
                                      |
              HIGH  ───────────────→  Fraud investigation queue
              MEDIUM ─────────────→  Step-up authentication / analyst watchlist
              LOW ───────────────→   Auto-approve
```

---

## Monitoring & Governance

### Data Drift Detection

- **PSI (Population Stability Index):**
  - PSI < 0.10 → stable
  - 0.10 < PSI < 0.25 → moderate drift (investigate)
  - PSI > 0.25 → significant drift (retrain trigger)
- **Missing rate monitoring** — sudden increase = upstream data pipeline issue
- **New category detection** — new merchants/MCCs unseen in training = coverage gap

### Performance Drift

- PR-AUC tracked by month/quarter via `performance_by_period()`
- Alert precision (HIGH alerts that are actual fraud) monitored over time
- Degradation triggers model retraining workflow

---

## Validated Production Readiness (WS1–WS4 Complete)

### What's Been Proven

1. End-to-end ML pipeline: ingestion → features → tuning → training → calibration → evaluation → serving
2. **CatBoost champion:** Test PR-AUC=0.854, ROC-AUC=0.999, KS=0.977, ECE≈0 — no overfitting
3. **XGBoost:** Test PR-AUC=0.847, better Recall@P90=0.643, better F1@T_high=0.776
4. Isotonic calibration (probabilities are well-calibrated: ECE=0.000001)
5. SHAP explainability (global importance + per-transaction reason codes)
6. Time-based splits (no data leakage)
7. Behavioral features validated by SHAP (top-5 in both models)
8. Bayesian-smoothed historical fraud rates (prevents overfitting on rare entities)
9. FastAPI + Gradio serving — interactive stakeholder UI, root `/` opens Gradio directly
10. PSI drift monitoring + performance trending
11. 44 tests across 7 test files
12. **MLflow model registry** — every training run tracked; @champion/@challenger aliases; app.py loads from registry at startup (WS3)
13. **Champion-challenger framework** — promote_champion() called after each evaluation; challenger set automatically (WS3)
14. **Docker + docker-compose** — single-image deployment, artifacts volume-mounted (WS4)
15. **GitHub Actions CI/CD** — lint + test matrix (3.10/3.11) + docker build+push on every merge to main (WS4)

### What's Missing for Bank Treasury Deployment (WS5–WS6)

#### Tier 1: Critical for Go-Live
1. **Real-time feature store** — Redis for behavioral features without batch recompute (WS5)
2. **API authentication** — OAuth2/API key for inter-service communication (WS6)
3. **Audit trail** — every score logged with model version, features hash, reason codes (WS6)

#### Tier 2: Regulatory
4. **SR 11-7 model governance** — independent validation, stress testing, bias testing
5. **Regulatory reporting** — automated model performance reports
6. **W&B experiment tracking** — sweep visualisation alongside MLflow (WS3 remainder)

#### Tier 3: Operational
7. **Kafka streaming** — real-time transaction ingestion (WS5)
8. **Prometheus + Grafana** — observability stack (WS6)
9. **Automated drift alerting** — PSI thresholds trigger PagerDuty/Slack (WS6)
10. **AWS deployment** — ECR + ECS Fargate + ALB + S3 (WS4 remainder)
