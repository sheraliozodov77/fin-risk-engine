# Execution — Checklist & Commands

**For phase detail:** MASTER_PLAN.md. **For schema:** DATA_CONTRACT.md.

---

## Phases 1–5 (Data foundation)

- [ ] Load all 5 files; schema audit; DATA_CONTRACT updated
- [ ] Data quality: duplicate IDs, amounts, timestamps; card→user integrity; label coverage
- [ ] Temporal norm: sort by `date`; hour, dow, is_weekend, is_night
- [ ] Labels: Yes→1, No→0; join on `id` (string); has_label, is_fraud
- [ ] Gold join: transactions → cards → users → mcc → labels; row count = transaction count

```bash
PYTHONPATH=. python scripts/build_gold_table.py [--nrows N] [--out path]
```

---

## Phases 6–10 (Features & splits)

- [ ] Static: parse $, booleans; acct_age_days
- [ ] Behavioral (optional): 10m/1h/24h/7d windows; velocity, recency, novelty, z_score
- [ ] Risk: merchant/MCC/card_brand fraud rates (past-only, Bayesian)
- [ ] Gold table saved; time-based splits: train ≤2017, val 2018–2019, test ≥2020

```bash
PYTHONPATH=. python scripts/build_features_and_splits.py [--skip-behavioral] [--behavioral-batch 400]
```

---

## Phases 10–11 (Modeling & explainability)

- [ ] Train CatBoost; PR-AUC, ROC-AUC, Brier, Recall@P90 on val
- [ ] Calibration (isotonic); Brier before/after; save calibrator
- [ ] SHAP: global importance; sample local reason codes (high-risk rows)
- [ ] T_high, T_med from config; VALIDATION_REPORT.md, MODEL_CARD.md, THRESHOLD_POLICY.md

```bash
PYTHONPATH=. python scripts/train_model.py [--iterations N]
PYTHONPATH=. python scripts/verify_model.py
PYTHONPATH=. python scripts/evaluate_model.py
```

---

## Phases 12–13 (Monitoring & serving)

- [ ] Data drift: PSI on amount and key categoricals; missing rate; new merchants
- [ ] Performance drift: PR-AUC by time period; alert precision trend
- [ ] FastAPI scoring: feature vector → risk score + level (HIGH/MED/LOW) + top-k reason codes
- [ ] Single-command pipeline (build → train → verify → evaluate); README complete

```bash
# Monitoring (reference=train, current=val; optional --current test.parquet, --period Q)
PYTHONPATH=. python scripts/run_monitoring.py

# Serving (loads model + calibrator from config; /health, /score, /score_batch)
PYTHONPATH=. python scripts/run_serve.py [--host 127.0.0.1] [--port 8000]
```

---

## Single-command pipeline

```bash
# Full pipeline: build → train → verify → evaluate (no monitoring/serving)
PYTHONPATH=. python scripts/run_pipeline.py [--skip-behavioral] [--nrows N] [--skip-evaluate]
```

---

## Quick start

```bash
pip install -r requirements.txt
PYTHONPATH=. python scripts/run_pipeline.py --skip-behavioral
# Or step-by-step:
PYTHONPATH=. python scripts/build_features_and_splits.py --skip-behavioral
PYTHONPATH=. python scripts/train_model.py
PYTHONPATH=. python scripts/verify_model.py
PYTHONPATH=. python scripts/evaluate_model.py
PYTHONPATH=. pytest tests/ -v
```
