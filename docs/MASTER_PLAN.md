# Financial Transaction Risk & Fraud Early-Warning System  
## Master Plan — Execution Blueprint

**System:** Transaction Risk Intelligence Engine  
**Domain:** Banking / Treasury-grade fraud early-warning  
**Dataset:** CaixaBank Tech 2024 AI Hackathon (5 files, 2010s decade)  
**Target:** Binary fraud label with explainable risk scores and alert policy  

---

## 1. Problem Definition (Locked)

| Item | Definition |
|------|------------|
| **Business objective** | Score each transaction and route high-risk ones to alert/review with explainable reason codes. |
| **Prediction target** | `risk_score(t) = P(fraud=1 \| history ≤ t)`; risk_level ∈ {LOW, MEDIUM, HIGH}; top_reasons(t). |
| **Decision policy** | HIGH → block/hold or manual review; MEDIUM → step-up/watchlist; LOW → allow. |
| **Constraints** | Time-aware features only; low latency; class imbalance (PR-AUC, recall@precision); explainability; monitoring. |

---

## 2. Data Model (Verified)

| Entity | Primary key | Source | Row counts (approx.) |
|--------|-------------|--------|----------------------|
| **Transaction** | `id` | transactions_data.csv | ~13.3M |
| **Card** | `id` | cards_data.csv | ~6,146 |
| **User** | `id` | users_data.csv | ~2,000 |
| **MCC** | code (string) | mcc_codes.json | ~109 |
| **Fraud label** | transaction `id` (string) | train_fraud_labels.json | ~8.9M labeled, ~13.3K fraud |

**Join keys (authoritative):**
- `transactions.card_id` → `cards.id`
- `transactions.client_id` → `users.id`
- `transactions.mcc` → `mcc_codes` key (normalize to string)
- `transactions.id` (as string) → `train_fraud_labels["target"]` key

**Label coverage:** Subset of transactions have labels; unlabeled = unknown (do not treat as non-fraud).

---

## 3. Technical Phases (End-to-End)

### Phase 1 — Data loading & schema validation (Week 1, Days 1–2)
- Load all 5 sources; no joins.
- Schema audit: dtypes, null %, cardinality, min/max.
- Parse: timestamps, amounts ($), credit_limit ($), booleans (YES/NO, card_on_dark_web).
- Document in `docs/DATA_CONTRACT.md`.

### Phase 2 — Data quality & sanity (Week 1, Days 2–3)
- Duplicate transaction IDs, negative/zero amounts, impossible timestamps.
- Card/user referential integrity (cards → users).
- Fraud label coverage: % labeled, fraud rate overall and by year.

### Phase 3 — Temporal normalization (Week 1, Day 3)
- All timestamps → UTC; sort transactions by `date`.
- Derive: hour_of_day, day_of_week, is_weekend, month, is_night.

### Phase 4 — Label construction & alignment (Week 1, Day 4)
- Parse labels: "Yes"→1, "No"→0.
- Join to transactions on `id` (string); flags: has_label, is_fraud, is_unlabeled.

### Phase 5 — Entity joining (star schema) (Week 1, Day 4–5)
- Order: transactions → cards → users → mcc → labels.
- Left joins; track row loss and nulls; post-join integrity checks.

### Phase 6 — Static feature preprocessing (Week 2, Days 1–2)
- Numeric: parse $ strings, handle impossible values.
- Categorical: keep raw for CatBoost (card_brand, card_type, mcc, merchant_id, etc.).
- Boolean: normalize to 0/1.

### Phase 7 — Behavioral feature engineering (Week 2, Days 2–4)
- Windows: 10m, 1h, 24h, 7d per card_id and client_id.
- Velocity: tx_count, unique_merchants, unique_mcc.
- Monetary: amount_sum, amount_mean, amount_max; ratios (e.g. 1h/24h).
- Recency/novelty: time_since_last_tx, first_time_merchant, first_time_mcc.
- Baseline deviation: z_score_amount vs user rolling mean/std (7d/30d).

### Phase 8 — Leakage-safe historical risk features (Week 2, Day 4)
- Merchant/MCC fraud rates using only past data at time t; Bayesian smoothing.

### Phase 9 — Gold table & splits (Week 2, Day 5)
- One row per transaction; all features + target + event_time.
- Persist: `data/processed/gold_transactions.parquet`.
- Time-based split: Train (e.g. 2010–2017), Val (2018–2019), Test/OOT (2020+).

### Phase 10 — Modeling (Week 3)
- CatBoost classifier (primary); XGBoost benchmark.
- Metrics: PR-AUC, ROC-AUC, recall@precision, calibration, cost-based threshold.
- Calibration (e.g. Platt/isotonic) for risk scores.

### Phase 11 — Explainability & decisioning (Week 3–4)
- SHAP: global top features; local top-5 reason codes per alert.
- Alert policy: thresholds T_high, T_med; alert volume and cost model.

### Phase 12 — Monitoring & governance (Week 4)
- Data drift: PSI on key features; missing rates; category explosion.
- Performance drift: PR-AUC by period; alert precision trend.
- Model card, data card, validation report.

### Phase 13 — Serving & reproducibility (Week 4–5)
- FastAPI scoring service; optional streaming simulation.
- Reproducible pipelines; config-driven; README and docs.

---

## 4. Timeline Summary

| Week | Focus |
|------|--------|
| **1** | Load, validate, join, temporal norm, label alignment, gold table start |
| **2** | Static + behavioral features, leakage-safe aggregates, gold table, splits |
| **3** | Train/val CatBoost, calibration, SHAP, threshold policy |
| **4** | Monitoring, model/data cards, validation report |
| **5** | API, reproducibility, optional Docker |

| Phase | Week | Deliverable |
|-------|------|-------------|
| 1–5 | 1 | Data load, schema, quality, gold join, labels |
| 6–8 | 2 | Static + behavioral + risk features |
| 9–10 | 2 | gold_transactions.parquet; train/val/test splits |
| 10–11 | 3 | CatBoost train; PR-AUC, calibration, SHAP, thresholds |
| 12–13 | 4–5 | Monitoring, MODEL_CARD, VALIDATION_REPORT, API |

---

## 4b. Pre-modeling & modeling reference (core)

**Inputs:** `train.parquet`, `val.parquet` from `data/processed/` (time-based splits).  
**Target:** `is_fraud` (0/1); use only rows with `has_label == True`.  
**Features:** All static + risk (and behavioral if built). Exclude: `id`, `date`, `transaction_id_str`, `has_label`, `is_fraud`, `card_number`, `address`, `mcc_description`. Categorical auto-detected (string/object/category/bool).  
**KPIs:** PR-AUC (primary), ROC-AUC, Brier, Recall@P90. Config: `config/model_config.yaml` (target, catboost params, metrics, thresholds T_high=0.7, T_med=0.3).  
**Commands:**  
`PYTHONPATH=. python scripts/build_features_and_splits.py [--skip-behavioral]`  
`PYTHONPATH=. python scripts/train_model.py [--iterations N]`

---

## 5. Codebase Structure

```
financial_transactional_data/
├── config/
│   ├── paths.yaml          # Data paths, output dirs
│   ├── schema.yaml         # Column names, dtypes, keys (optional)
│   └── model_config.yaml   # Splits, thresholds, CatBoost params
├── data/
│   ├── raw/                # Symlinks or copies of original 5 files (optional)
│   └── processed/          # gold_transactions.parquet, splits
├── docs/
│   ├── MASTER_PLAN.md      # This file
│   ├── DATA_CONTRACT.md    # Schema, keys, data dictionary
│   ├── MODEL_CARD.md       # Model card (after training)
│   └── VALIDATION_REPORT.md
├── notebooks/
│   ├── 01_data_loading_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling_catboost.ipynb
├── src/
│   ├── __init__.py
│   ├── config/             # Load paths and config
│   ├── ingestion/          # Load CSVs/JSON, schema validation
│   ├── features/            # Build static + behavioral + risk features
│   ├── modeling/            # Train, evaluate, calibrate, explain
│   ├── serving/            # FastAPI scoring
│   └── monitoring/         # Drift, metrics
├── tests/
│   └── test_ingestion.py
├── reports/                # Generated charts, tables (optional)
├── README.md
├── requirements.txt
└── pyproject.toml
```

---

## 6. Documentation Deliverables

| Document | Purpose |
|----------|---------|
| **MASTER_PLAN.md** | This file — phases, timeline, structure, modeling reference. |
| **DATA_CONTRACT.md** | Schemas, join keys, data dictionary, known issues. |
| **EXECUTION.md** | Phase checklist and key commands. |
| **MODEL_CARD.md** | (After training) Model type, data, metrics, limitations. |
| **VALIDATION_REPORT.md** | (After validation) Split strategy, metrics, threshold analysis. |
| **README.md** | How to run, env setup, quick start. |

---

## 7. Inference Workflow (Reference)

1. **Ingest** transaction → validate schema, parse timestamp, amount.
2. **Fetch** card, user, MCC by keys.
3. **Feature store** → rolling windows (10m, 1h, 24h, 7d) for card/user.
4. **Assemble** feature vector (same as training).
5. **Score** CatBoost → probability; optionally calibrate.
6. **Decision** LOW/MED/HIGH via thresholds.
7. **Explain** (alerts only): SHAP → top reason codes.
8. **Persist** scored txn, alert queue; **update** feature store.

---

## 8. Success Criteria

- [ ] Time-based splits only; no random split for fraud.
- [ ] Behavioral windows implemented leakage-free.
- [ ] Historical fraud rates computed with time cutoff.
- [ ] PR-AUC and calibration documented.
- [ ] SHAP reason codes for alerts.
- [ ] Drift monitoring (PSI + performance) in place.
- [ ] Single-command gold table build and train from config.
- [ ] Model card and validation report written.

---

*Document version: 1.0 — Kick-off.*
