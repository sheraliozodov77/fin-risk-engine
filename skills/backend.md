# Backend Architecture & Infrastructure

## System Overview

**Project:** Financial Transaction Risk & Fraud Early-Warning System (`fin-risk-engine` v0.1.0)
**Runtime:** Python 3.10+ | CatBoost (champion, PR-AUC=0.854) + XGBoost (PR-AUC=0.847) | FastAPI + Gradio serving layer
**Domain:** Banking / Treasury-grade real-time transaction risk scoring
**Virtual env:** `.finvenv/` (not `.venv/`)

---

## Architecture Layers

### 1. Configuration Layer (`src/config/`, `config/`)

- **`config/paths.yaml`** — all data/output paths (relative to project root); auto-resolved at runtime
- **`config/model_config.yaml`** — splits, thresholds (T_high=0.7, T_med=0.3), CatBoost hyperparams, behavioral windows, serving file names
- **`src/config/load.py`** — `get_paths()` and `get_model_config()` load YAML with automatic path resolution via `_resolve()`
- Config is injected into every pipeline stage; no hardcoded paths

### 2. Data Ingestion Layer (`src/ingestion/`)

- **`load.py`** — 5 loaders: `load_transactions()`, `load_cards()`, `load_users()`, `load_mcc_codes()`, `load_fraud_labels()`
  - All resolve paths from `config/paths.yaml` via `_resolve_path(key)`
  - Transactions support `nrows` sampling and optional date parsing
  - Fraud labels parse `"Yes"/"No"` to `1/0`
- **`gold_table.py`** — `build_gold_table()` orchestrates the star-schema join:
  - transactions → cards (on `card_id`) → users (on `client_id`) → MCC codes (on `mcc_str`) → fraud labels (on `transaction_id_str`)
  - Adds `has_label` and `is_fraud` columns; persists as Parquet

### 3. Feature Engineering Layer (`src/features/`)

Three sub-modules, all leakage-safe (past-only at time t):

- **`build.py`** (Static features):
  - `parse_currency()` — `$-77.00` → float
  - `parse_card_bools()` — YES/NO → 0/1 for `has_chip`, `card_on_dark_web`
  - `add_time_features()` — `hour_of_day`, `day_of_week`, `is_weekend`, `month`, `is_night`
  - `prepare_static_transaction()`, `prepare_cards()`, `prepare_users()`, `add_acct_age_days()`

- **`behavioral.py`** (Rolling window features — 55 features from 4 windows):
  - Per-entity (card_id, client_id) rolling windows: 10min, 1h, 24h, 7d
  - Past-only: tx_count, amount_sum, amount_mean, amount_max, unique_merchants, unique_mcc
  - `add_time_since_last_tx()` — seconds since last tx per card/user
  - `add_novelty()` — first_time_merchant/mcc flags for card and user
  - `add_z_score_amount()` — amount deviation from user's rolling 7d mean
  - `add_behavioral_features()` — orchestrator with `--behavioral-batch 400` for entity-batching

- **`risk_aggregates.py`** (Historical fraud rates):
  - `add_merchant_fraud_rate()`, `add_mcc_fraud_rate()`, `add_card_brand_fraud_rate()`
  - Bayesian smoothing: `(fraud_past + alpha) / (total_past + beta)` with alpha=1, beta=10
  - Strictly past-only: cumsum + shift(1) pattern

- **`splits.py`** (Time-based splitting):
  - `time_based_splits()` — train ≤ 2017, val 2018-2019, test ≥ 2020
  - `split_fraud_rates()` — stats per split
  - `save_splits()` — writes train/val/test Parquet files

- **`selection.py`** (Feature selection):
  - `correlation_filter()` — drop highly-correlated pairs (threshold=0.95)
  - `vif_filter()` — iterative VIF reduction (max_vif=10)
  - `permutation_importance_filter()` — post-training feature pruning
  - Note: VIF filter hurts tree models; keep all 44–99 features for CatBoost/XGBoost

### 4. Modeling Layer (`src/modeling/`)

- **`train.py`**:
  - `get_feature_columns()` — auto-detect numeric/categorical; exclude IDs, PII, target
  - `_fill_cat(series)` — centralized NaN→`"__missing__"` handler (handles pandas 3.0 StringDtype)
  - `_prepare_X(df, feature_cols, cat_features)` — shared feature matrix preparation
  - `train_catboost()` — CatBoost binary classifier; `val_df` → `cb.Pool` eval_set for early stopping; `train_dir="outputs/catboost_info"`
  - `evaluate_val()` — PR-AUC, ROC-AUC, Brier, Recall@Precision90

- **`train_xgboost.py`**:
  - `_target_encode_fit(X, y, cat_features)` — Bayesian smoothed target encoding: `(n*cat_mean + m*global_mean) / (n+m)`, m=100
  - `_target_encode_transform(X, cat_features, encoders)` — apply fitted encoders; unseen categories → global mean
  - `train_xgboost()` — XGBoost with target encoding + early stopping via `eval_set`
  - Returns `(model, encoders)` tuple (encoders required for inference)

- **`train_lightgbm.py`**:
  - `train_lightgbm()` — LightGBM with native `.astype("category")` support
  - Note: OOM on Mac with 10.7M rows; use benchmark with reduced data

- **`tuning.py`** (Optuna hyperparameter optimization):
  - Data precomputed ONCE before study (not per-trial) — avoids N×encoding overhead
  - `_catboost_objective_fast()` — uses pre-built `cb.Pool`; `train_dir="outputs/catboost_info"`
  - `_xgboost_objective_fast()` — uses pre-encoded data
  - `_lightgbm_objective_fast()` — LightGBM with category dtype
  - `run_tuning()` — main entry: precompute → create study → optimize → return best params
  - Best CatBoost: PR-AUC=0.852 (20 trials, saved to `outputs/tuning/catboost_best_params.json`)
  - Best XGBoost: PR-AUC=0.847 (10 trials, saved to `outputs/tuning/xgboost_best_params.json`)

- **`benchmark.py`** (Multi-model comparison):
  - `ModelResult` dataclass: name, model, pr_auc, roc_auc, brier, recall@p90, train_time, params
  - `run_benchmark()` — trains selected models, saves partial JSON after each (crash protection)
  - `compare_models()` — comparison DataFrame sorted by PR-AUC
  - `--use-tuned` flag loads best params from `outputs/tuning/`

- **`calibrate.py`**:
  - `fit_calibrator()` — isotonic (default) or Platt scaling
  - `apply_calibrator()` — transform raw proba to calibrated
  - `brier_score()` — calibration quality metric
  - Per-model calibrators: `calibrator_catboost.joblib`, `calibrator_xgboost.joblib`
  - `calibrator.joblib` = champion calibrator copy (used by serving API)

- **`explain.py`** (SHAP explainability):
  - `get_global_importance()` — mean |SHAP| across sample; CatBoost uses `ShapValues` via Pool
  - `get_local_reason_codes()` — per-row top-k (feature, shap_value) for alert explanation
  - `get_local_reason_codes_batch()` — batch SHAP for multiple rows

- **`advanced_metrics.py`**:
  - `confusion_matrix_at_threshold(y, proba, threshold)` — TP/FP/TN/FN + P/R/F1
  - `ks_statistic(y, proba)` — Kolmogorov-Smirnov separation
  - `lift_chart_data(y, proba, n_bins=10)` — decile lift
  - `expected_calibration_error(y, proba, n_bins=10)` — ECE

### 5. Serving Layer (`src/serving/`)

- **`app.py`** — FastAPI + Gradio:
  - **Startup:** `load_artifacts()` loads model (.cbm), `calibrator.joblib` (champion), `feature_metadata.json`, thresholds
  - **Endpoints:** `GET /health`, `GET /model_info`, `GET /sample_from_test`, `POST /score`, `POST /score_batch`
  - **Gradio UI at `/gradio`:** interactive scoring, risk gauge chart, SHAP bar chart, load sample button
  - **Scoring pipeline:** features dict → DataFrame → predict_proba → calibrate → threshold → SHAP reason codes

### 6. Monitoring Layer (`src/monitoring/`)

- **`drift.py`** — PSI numeric/categorical, missing rate, new category share, `compute_drift()` orchestrator
- **`performance.py`** — `performance_by_period()`: PR-AUC + alert precision by month/quarter/week

---

## Scripts (Orchestration)

| Script | Purpose |
|--------|---------|
| `scripts/build_features_and_splits.py` | Gold table → static → behavioral → risk → train/val/test Parquet |
| `scripts/train_model.py` | Train CatBoost or XGBoost; `--use-tuned`, `--subsample-train`, `--model-type` |
| `scripts/verify_model.py` | Sanity check: load model, run one prediction |
| `scripts/evaluate_model.py` | Calibration + SHAP + advanced metrics + reports; `--compare` for head-to-head |
| `scripts/run_pipeline.py` | Single command: build → train → verify → evaluate |
| `scripts/run_monitoring.py` | Drift + performance monitoring |
| `scripts/run_serve.py` | Launch FastAPI + Gradio server |
| `scripts/score_test_data.py` | Batch scoring to `outputs/artifacts/predictions.csv` |
| `scripts/run_feature_selection.py` | Correlation + VIF feature filtering |
| `scripts/run_tuning.py` | Optuna tuning; `--model-type`, `--n-trials`, `--subsample-train` |
| `scripts/run_benchmark.py` | 3-model benchmark; `--models`, `--use-tuned` |

---

## Data Flow

```
Raw CSV/JSON (5 files)
    |
    v
[Ingestion] load_* functions → star-schema join
    |
    v
[Gold Table] gold_transactions.parquet (108 columns)
    |
    v
[Features] Static (44) → Behavioral windows (55) → Risk aggregates (+3)
           = 99 features total, 9 categorical
    |
    v
[Splits] Time-based: train ≤ 2017 | val 2018-2019 | test ≥ 2020
    |
    v
[Tuning] Optuna (precomputed data, N trials) → best_params.json
    |
    v
[Training] CatBoost/XGBoost on 3M stratified subsample (all fraud + sampled non-fraud)
    |
    v
[Evaluation] Calibration + SHAP + advanced metrics → VALIDATION_REPORT.md
             --compare: head-to-head both models + MODEL_CARD.md
    |
    v
[Serving] FastAPI /score + Gradio UI
[Monitoring] PSI drift + PR-AUC by period
[Batch] score_test_data.py → predictions.csv
```

---

## Key Artifacts

| Path | Format | Contents |
|------|--------|----------|
| `data/processed/train.parquet` | Parquet | Training split (≤ 2017), 10.7M rows, 7.2M labeled |
| `data/processed/val.parquet` | Parquet | Validation split (2018-2019), 1.4M rows, 935K labeled |
| `data/processed/test.parquet` | Parquet | Test/OOT split (≥ 2020), 1.16M rows, 777K labeled |
| `outputs/models/catboost_fraud.cbm` | CatBoost | Champion model binary |
| `outputs/models/xgboost_fraud.json` | XGBoost | Runner-up model |
| `outputs/models/calibrator_catboost.joblib` | Joblib | CatBoost isotonic calibrator |
| `outputs/models/calibrator_xgboost.joblib` | Joblib | XGBoost isotonic calibrator |
| `outputs/models/calibrator.joblib` | Joblib | Champion calibrator copy (for serving API) |
| `outputs/models/feature_metadata_catboost.json` | JSON | CatBoost feature cols + cat cols (stable) |
| `outputs/models/feature_metadata_xgboost.json` | JSON | XGBoost feature cols + cat cols (stable) |
| `outputs/models/feature_metadata.json` | JSON | Last-trained model metadata (backward compat) |
| `outputs/models/xgboost_target_encoders.joblib` | Joblib | XGBoost Bayesian target encoders |
| `outputs/tuning/catboost_best_params.json` | JSON | CatBoost Optuna best params (PR-AUC=0.852) |
| `outputs/tuning/xgboost_best_params.json` | JSON | XGBoost Optuna best params (PR-AUC=0.847) |
| `outputs/catboost_info/` | CatBoost logs | Training curves (redirected from root via train_dir) |
| `outputs/artifacts/predictions.csv` | CSV | Batch prediction results |

---

## Critical Implementation Notes

- **NaN in categoricals:** ALWAYS use `_fill_cat()` from `train.py` — never `.astype(str).fillna()` which misses `"nan"` strings (pandas 3.0 bug)
- **CatBoost early stopping:** MUST pass `val_df` to `train_catboost()` which builds `cb.Pool` eval_set; without it, model overfits (PR-AUC drops from 0.85 → 0.65)
- **XGBoost categoricals:** Bayesian target encoding only — LabelEncoder loses 13% PR-AUC
- **scale_pos_weight:** Retune when subsampling changes class ratio. CatBoost ~8 (at 3M), XGBoost ~16 (at 3M). Very different from each other
- **LightGBM OOM:** 10.7M rows + 99 features OOMs on Mac; skip in benchmark
- **Subsampling:** `--subsample-train 3000000` keeps ALL fraud rows + samples non-fraud; use pyarrow filter pushdown + `gc.collect()` to avoid peak memory spike
- **Feature metadata:** Per-model files (`feature_metadata_catboost.json`, `feature_metadata_xgboost.json`) are stable; `feature_metadata.json` = last-trained (backward compat). `evaluate_model.py` loads from metadata (avoids loading 10.7M train rows)
- **CatBoost training logs:** Written to `outputs/catboost_info/` via `train_dir=` param (not project root)

---

## Dependencies

**Core:** pandas 3.0, numpy, pyyaml, pyarrow
**ML:** catboost, xgboost, lightgbm, scikit-learn, joblib, optuna, scipy, statsmodels
**Serving:** fastapi, uvicorn, gradio
**Viz:** matplotlib, seaborn
**Logging:** structlog
**Testing:** pytest, pytest-cov

---

## What's Next (WS3–WS6)

| Work Stream | Priority | Key Components |
|-------------|----------|----------------|
| **WS3: Experiment Tracking** | Next | MLflow model registry, W&B sweep, champion-challenger promotion |
| **WS4: Infrastructure** | Parallel with WS3 | Docker, CI/CD (GitHub Actions), AWS ECS/ECR |
| **WS5: Streaming** | After WS4 | Kafka producer/consumer, Redis feature store |
| **WS6: Observability** | After WS4 | Prometheus /metrics, drift alerting, audit trail, API auth |
