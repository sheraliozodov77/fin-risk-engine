# Financial Transaction Risk & Fraud Early-Warning System

**fin-risk-engine** — treasury-grade fraud detection pipeline: score every transaction, route high-risk to alert/review with explainable reason codes. Built on the CaixaBank Tech 2024 AI Hackathon dataset (~13.3M transactions).

## Results (Held-out Test Set, 2020 — never seen during training or tuning)

| Metric | CatBoost (Champion) | XGBoost |
|--------|---------------------|---------|
| **PR-AUC** | **0.854** | 0.847 |
| ROC-AUC | **0.999** | 0.999 |
| Brier (calibrated) | **0.000527** | 0.000555 |
| Recall @ 90% Precision | 0.600 | **0.643** |
| KS Statistic | 0.977 | **0.979** |
| F1 @ T_high (0.7) | 0.694 | **0.776** |
| F1 @ T_med (0.3) | **0.797** | 0.788 |

Both models validated on 777K labeled test rows (1,360 fraud cases, 0.175% fraud rate). No degradation from val → test confirms no overfitting.

---

## Repository Structure

```
config/                   # paths.yaml, model_config.yaml
data/processed/           # gold table, train/val/test splits (generated)
docs/                     # MASTER_PLAN.md, DATA_CONTRACT.md, VALIDATION_REPORT.md,
                          # MODEL_CARD.md, THRESHOLD_POLICY.md
outputs/
  models/                 # catboost_fraud.cbm, xgboost_fraud.json,
                          # calibrator_{catboost,xgboost}.joblib, calibrator.joblib (champion),
                          # feature_metadata_{catboost,xgboost}.json,
                          # xgboost_target_encoders.joblib
  tuning/                 # catboost_best_params.json, xgboost_best_params.json
  catboost_info/          # CatBoost training logs
  artifacts/              # predictions.csv (from batch scoring)
scripts/                  # pipeline orchestration (see Commands below)
src/
  config/                 # load paths and model config from YAML
  ingestion/              # load 5 raw files, build gold table (star schema)
  features/               # static + behavioral (rolling windows) + risk aggregates
  modeling/               # train, tune, calibrate, explain (SHAP), advanced metrics
  serving/                # FastAPI + Gradio UI
  monitoring/             # PSI drift, performance by period
tests/
skills/                   # Claude Code reference files (backend, business-logic)
pyproject.toml
requirements.txt
```

---

## Data (5 raw files at project root)

| File | Role |
|------|------|
| `transactions_data.csv` | Fact table (~13.3M rows) |
| `cards_data.csv` | Card dimension |
| `users_data.csv` | User dimension |
| `mcc_codes.json` | MCC code descriptions |
| `train_fraud_labels.json` | Fraud labels (~8.9M labeled, ~13.3K fraud) |

See `docs/DATA_CONTRACT.md` for schema, join keys, and column definitions.

---

## Quick Start

```bash
# 1. Environment
python -m venv .finvenv
source .finvenv/bin/activate
pip install -r requirements.txt

# 2. Full pipeline: build features → train → verify → evaluate
PYTHONPATH=. python scripts/run_pipeline.py

# Fast iteration (skip behavioral features, use 50K sample)
PYTHONPATH=. python scripts/run_pipeline.py --skip-behavioral --nrows 50000
```

---

## Commands

### Build features and splits

```bash
# Full build with behavioral features (slow, ~2h on full data)
PYTHONPATH=. python scripts/build_features_and_splits.py

# Fast build (no behavioral, good for initial setup)
PYTHONPATH=. python scripts/build_features_and_splits.py --skip-behavioral
```

### Train models

```bash
# CatBoost with Optuna-tuned params (recommended)
PYTHONPATH=. python scripts/train_model.py --use-tuned --subsample-train 3000000

# XGBoost with tuned params
PYTHONPATH=. python scripts/train_model.py --model-type xgboost --use-tuned --subsample-train 3000000

# With explicit params file
PYTHONPATH=. python scripts/train_model.py --params-file outputs/tuning/catboost_best_params.json
```

`--subsample-train 3000000` keeps all fraud rows + samples non-fraud to 3M total. Required on machines with <16GB RAM.

### Hyperparameter tuning (Optuna)

```bash
PYTHONPATH=. python scripts/run_tuning.py --model-type catboost --n-trials 20 --subsample-train 3000000
PYTHONPATH=. python scripts/run_tuning.py --model-type xgboost --n-trials 10 --subsample-train 3000000
```

### Evaluate

```bash
# Compare both models on val set (writes VALIDATION_REPORT.md + MODEL_CARD.md)
PYTHONPATH=. python scripts/evaluate_model.py --compare

# Evaluate on held-out test set
PYTHONPATH=. python scripts/evaluate_model.py --compare --val data/processed/test.parquet

# Single model
PYTHONPATH=. python scripts/evaluate_model.py --model-type catboost
```

### Benchmark (3 models)

```bash
PYTHONPATH=. python scripts/run_benchmark.py --models catboost,xgboost --use-tuned
```

### Serve

```bash
PYTHONPATH=. python scripts/run_serve.py
# FastAPI at http://127.0.0.1:8000
# Gradio UI at http://127.0.0.1:8000/gradio
```

### Monitor drift

```bash
PYTHONPATH=. python scripts/run_monitoring.py --current data/processed/test.parquet
```

### Batch score

```bash
PYTHONPATH=. python scripts/score_test_data.py --reason-codes
```

### Tests

```bash
PYTHONPATH=. pytest tests/ -v
```

---

## Pipeline Steps (Reference)

| Step | Script | Output |
|------|--------|--------|
| 1. Build features & splits | `build_features_and_splits.py` | `data/processed/train.parquet`, `val.parquet`, `test.parquet` |
| 2. Train | `train_model.py` | `outputs/models/catboost_fraud.cbm` (or `.json`), `feature_metadata_{model}.json` |
| 3. Verify | `verify_model.py` | Sanity-check prediction |
| 4. Evaluate | `evaluate_model.py --compare` | `docs/VALIDATION_REPORT.md`, `docs/MODEL_CARD.md`, calibrators |
| 5. Serve | `run_serve.py` | FastAPI + Gradio at port 8000 |
| 6. Monitor | `run_monitoring.py` | PSI drift + PR-AUC by period |
| 7. Batch score | `score_test_data.py` | `outputs/artifacts/predictions.csv` |

---

## Model Details

### CatBoost (Champion)

- Native categorical handling (no encoding needed)
- Optuna-tuned (20 trials on 3M subsample with behavioral features)
- Params: `iterations=843`, `lr=0.049`, `depth=9`, `l2_leaf_reg=8.13`, `border_count=141`, `scale_pos_weight=8.34`
- Top SHAP features: `use_chip`, `mcc_historical_fraud_rate`, `time_since_last_tx_card`, `first_time_merchant_user`, `zip`

### XGBoost

- Bayesian smoothed target encoding for categoricals (`m=100`)
- Optuna-tuned (10 trials): `n_estimators=895`, `lr=0.154`, `depth=8`, `SPW=15.85`
- Better Recall@P90 (0.643) and F1@T_high (0.776) than CatBoost
- Top SHAP features: `merchant_city`, `mcc_str`, `merchant_state`, `merchant_historical_fraud_rate`, `first_time_merchant_user`

### Decision Policy

| Level | Threshold | Action |
|-------|-----------|--------|
| HIGH | proba ≥ 0.70 | Block / manual review |
| MEDIUM | 0.30 ≤ proba < 0.70 | Step-up authentication |
| LOW | proba < 0.30 | Auto-approve |

Probabilities are isotonic-calibrated. Per-transaction SHAP reason codes explain every HIGH/MEDIUM alert.

---

## Key Artifacts

| Path | Contents |
|------|----------|
| `outputs/models/catboost_fraud.cbm` | CatBoost model |
| `outputs/models/xgboost_fraud.json` | XGBoost model |
| `outputs/models/calibrator_catboost.joblib` | CatBoost isotonic calibrator |
| `outputs/models/calibrator_xgboost.joblib` | XGBoost isotonic calibrator |
| `outputs/models/calibrator.joblib` | Champion calibrator (copy, used by serving API) |
| `outputs/models/feature_metadata_catboost.json` | CatBoost feature list (stable) |
| `outputs/models/feature_metadata_xgboost.json` | XGBoost feature list (stable) |
| `outputs/models/xgboost_target_encoders.joblib` | XGBoost Bayesian target encoders |
| `outputs/tuning/catboost_best_params.json` | CatBoost Optuna best params |
| `outputs/tuning/xgboost_best_params.json` | XGBoost Optuna best params |
| `docs/VALIDATION_REPORT.md` | Head-to-head metrics + SHAP + confusion matrices |
| `docs/MODEL_CARD.md` | Champion model card |

---

## Citation

Dataset: CaixaBank Tech, 2024 AI Hackathon — Financial Transactions Dataset.

---

*Time-based splits · Leakage-free behavioral features · Multi-model (CatBoost + XGBoost) · Optuna tuning · SHAP explainability · Isotonic calibration · Drift monitoring · FastAPI + Gradio*
