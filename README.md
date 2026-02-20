# fin-risk-engine

> **Treasury-grade financial fraud detection** — score every transaction in real time, route high-risk alerts with calibrated probabilities and per-transaction SHAP explanations.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![CatBoost](https://img.shields.io/badge/CatBoost-champion-brightgreen)
![XGBoost](https://img.shields.io/badge/XGBoost-challenger-orange)
![PR--AUC](https://img.shields.io/badge/PR--AUC-0.854-success)
[![CI/CD](https://github.com/sheraliozodov77/fin-risk-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/sheraliozodov77/fin-risk-engine/actions/workflows/ci.yml)
![Tests](https://img.shields.io/badge/tests-44%20passing-success)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## What It Does

fin-risk-engine ingests raw bank transaction data, engineers 99 features (static + behavioral rolling windows), trains and calibrates two ML models (CatBoost + XGBoost), and serves a REST API that returns a fraud probability, risk level (HIGH / MEDIUM / LOW), and the top-5 reason codes driving each decision — all in a single `/score` call.

Built on the **CaixaBank Tech 2024 AI Hackathon** dataset: ~13.3M transactions across a decade, 8.9M labeled rows, 0.15% fraud rate.

---

## Results

Evaluated on a **held-out test set (2020)** — never seen during training or tuning.

| Metric | CatBoost (Champion) | XGBoost |
|--------|:-------------------:|:-------:|
| **PR-AUC** | **0.854** | 0.847 |
| ROC-AUC | **0.999** | 0.999 |
| Brier (calibrated) | **0.000527** | 0.000555 |
| Recall @ 90% Precision | 0.600 | **0.643** |
| KS Statistic | 0.977 | **0.979** |
| F1 @ T_high (0.7) | 0.694 | **0.776** |
| F1 @ T_med (0.3) | **0.797** | 0.788 |

Zero val→test degradation confirms no overfitting across a 2-year temporal gap.

---

## Key Features

- **Leakage-free behavioral features** — 55 rolling-window signals (10 min / 1 h / 24 h / 7 d) computed strictly from past transactions
- **Bayesian risk aggregates** — smoothed historical fraud rates per merchant, MCC, card brand
- **Optuna hyperparameter tuning** — 20 trials (CatBoost) + 10 trials (XGBoost) stored in `outputs/tuning/`
- **Isotonic calibration** — probabilities are reliable for threshold-based routing
- **SHAP explainability** — global feature importance + per-transaction reason codes on every alert
- **Drift monitoring** — PSI for numeric/categorical features, performance drift by time period
- **MLflow model registry** — every run tracked; `@champion`/`@challenger` aliases; serving API loads from registry automatically
- **FastAPI + Gradio** — REST API + interactive stakeholder demo UI (opens at `/` by default)
- **Docker + CI/CD** — single-image deployment; GitHub Actions lint + test + push to ghcr.io on every merge

---

## Quick Start

```bash
# 1. Clone and set up environment
git clone https://github.com/<your-username>/fin-risk-engine.git
cd fin-risk-engine
python -m venv .finvenv && source .finvenv/bin/activate
pip install -r requirements.txt

# 2. Place raw data files in data/raw/
#    transactions_data.csv, cards_data.csv, users_data.csv,
#    mcc_codes.json, train_fraud_labels.json

# 3. Build features and splits (add --skip-behavioral for speed)
PYTHONPATH=. python scripts/build_features_and_splits.py

# 4. Train both models
PYTHONPATH=. python scripts/train_model.py --use-tuned --subsample-train 3000000
PYTHONPATH=. python scripts/train_model.py --model-type xgboost --use-tuned --subsample-train 3000000

# 5. Evaluate head-to-head
PYTHONPATH=. python scripts/evaluate_model.py --compare

# 6. Serve
PYTHONPATH=. python scripts/run_serve.py
# → API:    http://127.0.0.1:8000
# → UI:     http://127.0.0.1:8000/gradio
# → Docs:   http://127.0.0.1:8000/docs
```

Or run the full pipeline in one command:

```bash
PYTHONPATH=. python scripts/run_pipeline.py
```

> **RAM-constrained?** `--subsample-train 3000000` keeps all fraud rows + samples non-fraud to 3M total. Required on machines with <16 GB RAM.

---

## How It Works

```
Raw Data (5 files)
      │
      ▼
  Gold Table          ← star-schema join: transactions + cards + users + MCC + labels
      │
      ▼
Feature Engineering
  ├── Static (44)     ← amount, time-of-day, day-of-week, account age, boolean flags
  ├── Behavioral (55) ← rolling velocity/spend/recency/novelty per card & merchant
  └── Risk Aggregates ← Bayesian-smoothed fraud rates (merchant, MCC, card brand)
      │
      ▼
Time-Based Splits     ← train ≤ 2017 │ val 2018–2019 │ test ≥ 2020
      │
      ▼
Model Training        ← CatBoost (native categoricals) + XGBoost (target encoding)
      │
      ▼
Optuna Tuning         ← PR-AUC objective, precomputed data, 20+10 trials
      │
      ▼
Isotonic Calibration  ← reliable probabilities for threshold routing
      │
      ▼
Serving               ← FastAPI /score → {probability, level, reason_codes}
      │
      ▼
Monitoring            ← PSI drift + PR-AUC degradation alerts
```

---

## Scoring API

```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 284.50,
    "merchant_id": "M123",
    "use_chip": true,
    ...
  }'
```

```json
{
  "fraud_probability": 0.847,
  "risk_level": "HIGH",
  "reason_codes": [
    {"feature": "time_since_last_tx_card", "impact": 0.31},
    {"feature": "first_time_merchant_user", "impact": 0.24},
    {"feature": "mcc_historical_fraud_rate", "impact": 0.19}
  ]
}
```

---

## Decision Policy

| Level | Threshold | Action |
|-------|:---------:|--------|
| **HIGH** | proba ≥ 0.70 | Block / route to manual review |
| **MEDIUM** | 0.30 ≤ proba < 0.70 | Step-up authentication |
| **LOW** | proba < 0.30 | Auto-approve |

---

## Common Commands

```bash
make train          # CatBoost with tuned params
make train-xgboost  # XGBoost with tuned params
make evaluate       # head-to-head comparison on val set
make evaluate-test  # final evaluation on held-out test set
make serve          # launch API + Gradio UI  →  http://localhost:8000
make test           # run all tests (44 passing)
make tune           # re-run Optuna tuning (CatBoost)
make monitor        # drift + performance report
make mlflow-ui      # MLflow experiment browser  →  http://localhost:5001
make docker-build   # build Docker image
make docker-up      # start API (:8000) + MLflow (:5001) via docker-compose
make docker-down    # stop all containers
```

---

## Dataset

CaixaBank Tech, **2024 AI Hackathon** — Financial Transactions Dataset.

| Stat | Value |
|------|-------|
| Total transactions | ~13.3M |
| Labeled rows | ~8.9M |
| Fraud cases (labeled) | ~13.3K (0.15%) |
| Date range | 2010 – 2020 |
| Features (final) | 99 (44 static + 55 behavioral) |
