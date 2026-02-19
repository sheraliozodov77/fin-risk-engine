# CLAUDE.md -- Financial Transaction Risk & Fraud Early-Warning System

## Project Identity

- **Name:** `fin-risk-engine` v0.1.0
- **Domain:** Banking / Treasury-grade fraud detection and transaction risk scoring
- **Dataset:** CaixaBank Tech 2024 AI Hackathon (5 files, ~13.3M transactions, 2010s decade)
- **Language:** Python 3.10+
- **ML Framework:** CatBoost (champion PR-AUC=0.854 test), XGBoost (PR-AUC=0.847 test), LightGBM, scikit-learn (calibration/metrics)
- **Tuning:** Optuna hyperparameter optimization (precomputed data, per-trial logging)
- **Serving:** FastAPI + Gradio

## Quick Commands

```bash
# Activate environment
source .finvenv/bin/activate

# Run full pipeline (build -> train -> verify -> evaluate)
PYTHONPATH=. python scripts/run_pipeline.py

# Run pipeline without slow behavioral features
PYTHONPATH=. python scripts/run_pipeline.py --skip-behavioral

# Run pipeline on sample data (fast iteration)
PYTHONPATH=. python scripts/run_pipeline.py --nrows 50000 --skip-behavioral

# Step-by-step
PYTHONPATH=. python scripts/build_features_and_splits.py --skip-behavioral
PYTHONPATH=. python scripts/train_model.py
PYTHONPATH=. python scripts/verify_model.py
PYTHONPATH=. python scripts/evaluate_model.py

# Serve (after training)
PYTHONPATH=. python scripts/run_serve.py

# Monitoring
PYTHONPATH=. python scripts/run_monitoring.py

# Batch scoring
PYTHONPATH=. python scripts/score_test_data.py

# Tests
PYTHONPATH=. pytest tests/ -v

# Feature selection
PYTHONPATH=. python scripts/run_feature_selection.py

# Train with tuned params + subsampling (recommended for RAM-constrained machines)
PYTHONPATH=. python scripts/train_model.py --use-tuned --subsample-train 3000000
PYTHONPATH=. python scripts/train_model.py --model-type xgboost --use-tuned --subsample-train 3000000

# Evaluate both models (head-to-head comparison)
PYTHONPATH=. python scripts/evaluate_model.py --compare
PYTHONPATH=. python scripts/evaluate_model.py --compare --val data/processed/test.parquet

# Hyperparameter tuning
PYTHONPATH=. python scripts/run_tuning.py --model-type xgboost --n-trials 10 --subsample-train 3000000
PYTHONPATH=. python scripts/run_tuning.py --model-type catboost --n-trials 20 --subsample-train 3000000

# 3-model benchmark (CatBoost vs XGBoost vs LightGBM)
PYTHONPATH=. python scripts/run_benchmark.py --models xgboost --use-tuned
PYTHONPATH=. python scripts/run_benchmark.py --models catboost,xgboost --use-tuned
```

## Repository Structure

```
config/                    # YAML configuration (paths, model params, thresholds)
  paths.yaml               # Data file paths, output directories
  model_config.yaml        # Splits, thresholds, CatBoost params, windows

src/
  config/load.py           # Load and resolve YAML configs
  ingestion/load.py        # Load 5 raw data files (CSV/JSON)
  ingestion/gold_table.py  # Star-schema join -> gold table
  features/build.py        # Static feature parsing (amounts, booleans, time)
  features/behavioral.py   # Rolling window features (past-only, leakage-safe)
  features/risk_aggregates.py  # Historical fraud rates (Bayesian smoothed)
  features/splits.py       # Time-based train/val/test splitting
  features/selection.py     # Feature selection (correlation, VIF, permutation importance)
  modeling/train.py         # CatBoost training + evaluation (auto scale_pos_weight)
  modeling/train_xgboost.py # XGBoost trainer (Bayesian smoothed target encoding)
  modeling/train_lightgbm.py # LightGBM trainer (native categorical support)
  modeling/benchmark.py     # Unified 3-model benchmarking
  modeling/tuning.py        # Optuna tuning (precomputed data, per-trial callback logging)
  modeling/advanced_metrics.py # KS, lift, gain, ECE, confusion matrix at threshold
  modeling/calibrate.py     # Isotonic/Platt probability calibration
  modeling/explain.py       # SHAP global importance + local reason codes
  logging_config.py         # Structured logging (structlog)
  serving/app.py            # FastAPI endpoints + Gradio UI
  monitoring/drift.py       # PSI, missing rate, new category detection
  monitoring/performance.py # PR-AUC by period, alert precision trending
  eda/                      # Schema audit, summary statistics

scripts/                   # Pipeline orchestration scripts
  run_pipeline.py           # Single-command: build -> train -> verify -> evaluate
  build_features_and_splits.py  # Gold table + features + splits
  train_model.py            # Train CatBoost or XGBoost (--model-type, --use-tuned)
  verify_model.py           # Sanity check model
  evaluate_model.py         # Calibration + SHAP + advanced metrics + reports (--compare for head-to-head)
  run_serve.py              # Launch FastAPI server
  run_monitoring.py         # Run drift + performance monitoring
  score_test_data.py        # Batch predictions to CSV
  run_feature_selection.py  # Correlation + VIF feature filtering
  run_tuning.py             # Optuna hyperparameter tuning
  run_benchmark.py          # 3-model benchmark comparison

docs/                      # Documentation
  MASTER_PLAN.md            # 13-phase execution blueprint
  DATA_CONTRACT.md          # Schema, join keys, data dictionary
  EXECUTION.md              # Phase checklist with commands
  VALIDATION_REPORT.md      # Model metrics, SHAP importance, threshold counts
  MODEL_CARD.md             # Model card (type, data, metrics, limitations)
  THRESHOLD_POLICY.md       # HIGH/MEDIUM/LOW decision thresholds

skills/                    # Claude Code skill files
  backend.md                # Backend architecture reference
  business-logic.md         # Business logic + ML domain knowledge

tests/
  conftest.py               # Shared fixtures (synthetic DataFrames, mock model)
  test_ingestion.py         # Ingestion + config tests
  test_features_build.py    # Feature parsing, time features, static prep
  test_features_splits.py   # Time-based splitting, fraud rate stats
  test_modeling_train.py    # Feature column detection, evaluate_val
  test_modeling_calibrate.py # Isotonic/Platt calibration
  test_monitoring_drift.py  # PSI, missing rate, new categories
```

## What Has Been Completed

### Phase 1-5: Data Foundation (DONE)
- [x] All 5 raw data sources loaded and parsed (transactions, cards, users, MCC codes, fraud labels)
- [x] Schema audit and DATA_CONTRACT.md documenting all columns, types, join keys
- [x] Star-schema gold table join (transactions -> cards -> users -> MCC -> labels)
- [x] Label construction: "Yes"->1, "No"->0; `has_label` flag for partial labeling
- [x] Temporal normalization: date parsing, sorting, time feature extraction

### Phase 6-9: Feature Engineering (DONE)
- [x] Static features: amount parsing, boolean normalization, time features, acct_age_days
- [x] Behavioral features: rolling windows (10m/1h/24h/7d), velocity, monetary, recency, novelty, z-score
- [x] Risk aggregates: merchant/MCC/card_brand historical fraud rates with Bayesian smoothing
- [x] Gold table persisted as Parquet; time-based splits (train<=2017, val 2018-2019, test>=2020)

### Phase 10-11: Modeling & Explainability (DONE)
- [x] CatBoost classifier trained (PR-AUC=0.80, ROC-AUC=0.99, Brier=0.0007)
- [x] Isotonic calibration (Brier improved from 0.000694 to 0.000633)
- [x] SHAP global importance computed (top: merchant_fraud_rate, mcc_fraud_rate, day_of_week)
- [x] Local reason codes: per-row top-k SHAP for alert explanation
- [x] Threshold policy: T_high=0.7, T_med=0.3; VALIDATION_REPORT.md and MODEL_CARD.md generated

### Phase 12: Monitoring (DONE)
- [x] Data drift: PSI (numeric + categorical), missing rate monitoring, new category detection
- [x] Performance drift: PR-AUC by time period, alert precision trending
- [x] run_monitoring.py script with configurable reference/current datasets

### Phase 13: Serving (DONE)
- [x] FastAPI scoring API (/health, /model_info, /score, /score_batch, /sample_from_test)
- [x] Gradio interactive UI at /gradio (risk gauge, SHAP bar chart, sample loading)
- [x] Batch scoring script (score_test_data.py -> predictions.csv)
- [x] Single-command pipeline (run_pipeline.py)

### WS1: Foundation Hardening (DONE)
- [x] Structured logging (structlog) -- src/logging_config.py
- [x] Fixed critical bare except blocks with proper error logging
- [x] Fixed pandas 3.0 compatibility (StringDtype handling in build.py)
- [x] 48 tests across 7 test files (features, modeling, monitoring)
- [x] pyproject.toml with optional dependency groups [ml, serve, viz, dev, all]
- [x] requirements.txt updated with all new dependencies

### WS2: ML Robustness (DONE)
- [x] Feature selection: correlation filter, VIF filter, permutation importance (src/features/selection.py)
- [x] Class imbalance: scale_pos_weight defaults to 1.0, Optuna searches [1, 50] range
- [x] Optuna hyperparameter tuning with precomputed data (src/modeling/tuning.py)
- [x] XGBoost trainer with Bayesian smoothed target encoding (src/modeling/train_xgboost.py)
- [x] LightGBM trainer with native categoricals (src/modeling/train_lightgbm.py)
- [x] Unified benchmarking framework with partial results saving (src/modeling/benchmark.py)
- [x] Advanced metrics: KS statistic, lift/gain charts, ECE, confusion matrix at threshold
- [x] evaluate_model.py updated with advanced metrics in VALIDATION_REPORT.md
- [x] Critical NaN→"nan" bug fixed across all 13 files (centralized _fill_cat() helper)
- [x] CatBoost early stopping fix: val_df param with cb.Pool eval_set
- [x] CatBoost Optuna tuning: 20 trials with behavioral features + 3M subsample (PR-AUC=0.852 val)
- [x] XGBoost Optuna tuning: 10 trials with behavioral features + 3M subsample (PR-AUC=0.847 val)
- [x] --subsample-train flag: pyarrow filter pushdown + stratified sampling + gc.collect() (avoids OOM)
- [x] Benchmark script: --use-tuned flag loads best params from outputs/tuning/
- [x] evaluate_model.py: --compare flag for head-to-head comparison report
- [x] Per-model calibrators (calibrator_catboost.joblib, calibrator_xgboost.joblib)
- [x] train_catboost() fixed: now passes l2_leaf_reg and border_count from tuned params
- [x] Per-model feature metadata: feature_metadata_catboost.json + feature_metadata_xgboost.json (no overwrite)
- [x] evaluate_model.py: loads feature cols from metadata (no train.parquet load at eval time)
- [x] CatBoost train_dir fixed: logs write to outputs/catboost_info/ (not project root)
- [x] Test set (OOT 2020) evaluation: CatBoost PR-AUC=0.854, XGBoost=0.847 -- no overfitting confirmed

## Current Model Performance

### Head-to-Head Comparison (Held-Out Test Set, 2020)

| Metric | CatBoost (Champion) | XGBoost | Winner |
|--------|---------------------|---------|--------|
| PR-AUC | **0.8538** | 0.8470 | CatBoost |
| ROC-AUC | **0.9994** | 0.9992 | CatBoost |
| Brier (raw) | 0.000993 | **0.000573** | XGBoost |
| Brier (calibrated) | **0.000527** | 0.000555 | CatBoost |
| Recall@P90 | 0.6000 | **0.6426** | XGBoost |
| KS Statistic | 0.9769 | **0.9792** | Tie |
| ECE | 0.000001 | 0.000001 | Tie |
| F1@T_high | 0.6938 | **0.7757** | XGBoost |
| F1@T_med | **0.7972** | 0.7882 | CatBoost |

> Val → Test degradation is near zero for both models. No overfitting confirmed.

### CatBoost (Optuna-Tuned, Behavioral + 3M Subsample) -- Champion

| Metric | Val | Test |
|--------|-----|------|
| PR-AUC | 0.8517 | 0.8538 |
| ROC-AUC | 0.9995 | 0.9994 |
| Brier (calibrated) | 0.000553 | 0.000527 |
| Recall@P90 | 0.5953 | 0.6000 |

**Tuned Hyperparameters** (outputs/tuning/catboost_best_params.json):
- iterations=843, learning_rate=0.049, depth=9
- l2_leaf_reg=8.13, border_count=141
- scale_pos_weight=8.34 (tuned for 3M subsample, 1:289 fraud ratio)
- early_stopping_rounds=50

**CatBoost Optimization Journey:**
- Defaults (no tuning): PR-AUC=0.649 (overfit at iter 56)
- Optuna 10 trials (no behavioral, 7.2M): PR-AUC=0.841
- Optuna 20 trials (no behavioral, 7.2M): PR-AUC=0.859 (depth=7, lr=0.30, SPW=21.25)
- Optuna 20 trials (behavioral, 3M subsample): PR-AUC=0.852 (depth=9, lr=0.049, SPW=8.34)

### XGBoost (Optuna-Tuned, Behavioral + 3M Subsample)

| Metric | Val | Test |
|--------|-----|------|
| PR-AUC | 0.8471 | 0.8470 |
| ROC-AUC | 0.9992 | 0.9992 |
| Brier (calibrated) | 0.000547 | 0.000555 |
| Recall@P90 | 0.6309 | 0.6426 |

**Tuned Hyperparameters** (outputs/tuning/xgboost_best_params.json):
- n_estimators=895, learning_rate=0.154, max_depth=8
- reg_lambda=6.41, subsample=0.706, colsample_bytree=0.563
- scale_pos_weight=15.85, min_child_weight=14, gamma=0.829
- tree_method=hist, early_stopping_rounds=50

**XGBoost Optimization Journey:**
- LabelEncoder baseline: PR-AUC=0.640
- Bayesian target encoding: PR-AUC=0.773 (+20.8%)
- Optuna 10 trials (no behavioral, 7.2M): PR-AUC=0.817
- Optuna 10 trials (behavioral, 3M subsample): PR-AUC=0.847
- VIF feature selection (44→31): PR-AUC=0.722 (WORSE -- VIF hurts tree models)

### Dataset Stats
| Stat | Value |
|------|-------|
| Fraud rate (labeled) | ~0.15% |
| Train rows (total / labeled) | 10.7M / 7.2M |
| Val rows (total / labeled) | 1.4M / 935K |
| Test rows (total / labeled) | 1.16M / 777K |
| Training subsample | 3M (all 10,332 fraud + 2.99M non-fraud) |
| Features | 99 total (44 static + 55 behavioral), 9 categorical |

## What Needs to Be Done (Treasury/Finance Elevation)

### WS3: Experiment Tracking & Model Registry (NEXT)
- [ ] MLflow integration (src/tracking/mlflow_tracker.py)
- [ ] Weights & Biases integration (src/tracking/wandb_tracker.py)
- [ ] Champion-challenger model promotion
- [ ] Auto-generated model card from MLflow

### WS4: Infrastructure & Deployment
- [ ] Dockerize (Dockerfile + docker-compose with API, Redis, Kafka, MLflow)
- [ ] Makefile with standard targets
- [ ] CI/CD pipeline (GitHub Actions: lint, test, docker-build, deploy)
- [ ] AWS deployment (ECR + ECS Fargate + ALB + S3)

### WS5: Real-Time Streaming
- [ ] Kafka transaction producer/consumer
- [ ] Redis feature store for real-time behavioral features
- [ ] Real-time scoring pipeline

### WS6: Production Monitoring & Observability
- [ ] Prometheus metrics (/metrics endpoint)
- [ ] Automated drift alerting
- [ ] Structured audit trail (every score logged)
- [ ] API authentication & rate limiting

## Coding Conventions

- **PYTHONPATH:** Always set `PYTHONPATH=.` when running scripts from project root
- **Imports:** Use `from src.module import function` pattern
- **Missing values:** Numeric -> -999, Categorical -> use `_fill_cat()` from train.py (handles NaN→"nan" bug)
- **XGBoost categoricals:** Use Bayesian smoothed target encoding (`_target_encode_fit/transform` in train_xgboost.py), NOT LabelEncoder
- **CatBoost categoricals:** Native handling via `cat_features` param + `cb.Pool`
- **Feature leakage:** All behavioral/risk features MUST use past-only data (strictly before current transaction time)
- **Splits:** NEVER use random splits; always time-based for fraud detection
- **PII:** NEVER use card_number, address, cvv as model features
- **Config-driven:** All paths, hyperparameters, thresholds come from `config/` YAML files
- **Parquet for data:** All processed data stored as Parquet (not CSV)

## Important Notes

- The project uses `.finvenv/` as its virtual environment (not `.venv/`)
- Large data files (transactions_data.csv ~1.2GB, train_fraud_labels.json ~159MB) live in `data/raw/` (configured in `config/paths.yaml`)
- `data/processed/` and `outputs/` are gitignored (generated artifacts); `outputs/tuning/*.json` is tracked (Optuna results)
- CatBoost saves its training info in `outputs/catboost_info/` (redirected via `train_dir=` param -- not project root)
- Behavioral features are slow on full 13.3M rows; use `--skip-behavioral` or `--nrows` for fast iteration
- `--subsample-train 3000000` required on <16GB RAM: keeps all fraud + samples non-fraud to 3M total
- scale_pos_weight must be retuned when subsampling changes class ratio (3M → SPW~8-16; 7.2M → SPW~2-21)
- `evaluate_model.py` loads feature columns from per-model metadata files (not train.parquet) -- fast and memory-safe
- Per-model artifacts: `feature_metadata_catboost.json`, `calibrator_catboost.joblib`, `feature_metadata_xgboost.json`, `calibrator_xgboost.joblib` -- no shared copies
- Champion model for serving is set in `config/model_config.yaml` under `serving.champion_model`
- The `--behavioral-batch 400` flag processes behavioral features in entity batches to avoid memory issues                                                                                                                 
