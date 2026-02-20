#!/usr/bin/env python3
"""
Post-train evaluation: calibration, SHAP (global + sample local reason codes),
threshold counts, advanced metrics, VALIDATION_REPORT.md, MODEL_CARD.md.

Supports both CatBoost and XGBoost models (auto-detected from feature_metadata.json).
Use --compare to evaluate both models and generate a head-to-head comparison report.

Usage:
  # Auto-detect model type from feature_metadata.json
  PYTHONPATH=. python scripts/evaluate_model.py

  # Explicit model type
  PYTHONPATH=. python scripts/evaluate_model.py --model-type xgboost
  PYTHONPATH=. python scripts/evaluate_model.py --model-type catboost

  # Compare both models side-by-side
  PYTHONPATH=. python scripts/evaluate_model.py --compare
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd

from src.config import get_paths, get_model_config
from src.logging_config import setup_logging, get_logger
from src.modeling.train import get_feature_columns
from src.tracking.mlflow_tracker import MLflowTracker
from src.modeling.calibrate import fit_calibrator, apply_calibrator, brier_score
from src.modeling.explain import get_global_importance, get_local_reason_codes_batch
from src.modeling.advanced_metrics import (
    confusion_matrix_at_threshold,
    ks_statistic,
    lift_chart_data,
    expected_calibration_error,
)

logger = get_logger(__name__)


def _detect_model_type(models_dir: Path) -> str:
    """Auto-detect model type from per-model metadata files or model binaries."""
    # Prefer config champion_model setting
    cfg = get_model_config()
    champion = cfg.get("serving", {}).get("champion_model")
    if champion in ("catboost", "xgboost"):
        logger.info("detected_model_type", model_type=champion, source="model_config.yaml")
        return champion
    # Fall back to checking which model files exist
    if (models_dir / "xgboost_fraud.json").exists() and not (models_dir / "catboost_fraud.cbm").exists():
        return "xgboost"
    return "catboost"


def _load_model(model_type: str, models_dir: Path):
    """Load model and optional encoders based on type."""
    if model_type == "xgboost":
        import xgboost as xgb
        model_path = models_dir / "xgboost_fraud.json"
        if not model_path.exists():
            logger.error("model_not_found", path=str(model_path))
            return None, None
        model = xgb.XGBClassifier()
        model.load_model(str(model_path))
        enc_path = models_dir / "xgboost_target_encoders.joblib"
        encoders = joblib.load(str(enc_path)) if enc_path.exists() else None
        logger.info("loaded_xgboost", model=str(model_path), has_encoders=encoders is not None)
        return model, encoders
    else:
        import catboost as cb
        model_path = models_dir / "catboost_fraud.cbm"
        if not model_path.exists():
            logger.error("model_not_found", path=str(model_path))
            return None, None
        model = cb.CatBoostClassifier()
        model.load_model(str(model_path))
        logger.info("loaded_catboost", model=str(model_path))
        return model, None


def _prepare_val(val_df, feature_cols, cat_cols, model_type, encoders=None):
    """Prepare validation data based on model type."""
    from src.modeling.train import _fill_cat

    has_label_col = "has_label"
    target_col = "is_fraud"

    if has_label_col in val_df.columns:
        val_df = val_df.loc[val_df[has_label_col]].copy()
    val_df = val_df.dropna(subset=[target_col])

    X = val_df[feature_cols].copy()
    y = val_df[target_col].astype(int)

    if model_type == "xgboost" and encoders is not None:
        from src.modeling.train_xgboost import _target_encode_transform
        for c in cat_cols:
            if c in X.columns:
                X[c] = _fill_cat(X[c])
        X = _target_encode_transform(X, cat_cols, encoders)
        for c in feature_cols:
            if c not in cat_cols and c in X.columns and pd.api.types.is_numeric_dtype(X[c]):
                X[c] = X[c].fillna(-999)
    else:
        for c in feature_cols:
            if c in cat_cols and c in X.columns:
                X[c] = _fill_cat(X[c])
            elif c in X.columns and pd.api.types.is_numeric_dtype(X[c]):
                X[c] = X[c].fillna(-999)

    return X, y


def _evaluate_single(
    model,
    model_type: str,
    encoders,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    cat_cols: list[str],
    models_dir: Path,
    calibration_method: str = "isotonic",
    shap_sample: int = 2000,
) -> dict:
    """
    Run full evaluation for one model. Returns dict with all metrics,
    SHAP importance, reason codes, and calibrator.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, precision_recall_curve

    cfg = get_model_config()
    t_high = cfg.get("thresholds", {}).get("high", 0.7)
    t_med = cfg.get("thresholds", {}).get("medium", 0.3)

    model_name = "XGBoost" if model_type == "xgboost" else "CatBoost"
    print(f"\n{'='*50}")
    print(f"  Evaluating {model_name}")
    print(f"{'='*50}")

    # Prepare val data
    X_val, y_val = _prepare_val(val_df, feature_cols, cat_cols, model_type, encoders)
    if len(X_val) == 0:
        logger.error("no_labeled_val_rows", model_type=model_type)
        return {}
    logger.info("val_prepared", model_type=model_type, rows=len(X_val), fraud=int(y_val.sum()))

    # Raw predictions
    proba_raw = model.predict_proba(X_val)[:, 1]
    pr_auc = average_precision_score(y_val, proba_raw)
    roc_auc = roc_auc_score(y_val, proba_raw)
    brier_raw = brier_score_loss(y_val, proba_raw)
    precision_arr, recall_arr, _ = precision_recall_curve(y_val, proba_raw)
    mask = precision_arr >= 0.9
    r_at_p90 = float(recall_arr[mask].max()) if mask.any() else float("nan")

    print(f"  PR-AUC:      {pr_auc:.6f}")
    print(f"  ROC-AUC:     {roc_auc:.6f}")
    print(f"  Brier (raw): {brier_raw:.6f}")
    print(f"  Recall@P90:  {r_at_p90:.4f}")
    logger.info("raw_metrics", model_type=model_type,
                pr_auc=round(pr_auc, 6), roc_auc=round(roc_auc, 6),
                brier=round(brier_raw, 6), recall_at_p90=round(r_at_p90, 4))

    # Calibration
    print(f"  Calibrating ({calibration_method})...")
    calibrator = fit_calibrator(proba_raw, y_val.values, method=calibration_method)
    proba_cal = apply_calibrator(proba_raw, calibrator)
    brier_cal = brier_score(y_val.values, proba_cal)
    print(f"    Brier before: {brier_raw:.6f}  after: {brier_cal:.6f}")

    # Save calibrator per model
    cal_path = models_dir / f"calibrator_{model_type}.joblib"
    joblib.dump(calibrator, cal_path)
    logger.info("saved_calibrator", path=str(cal_path))

    # Thresholds
    n_high = int((proba_cal >= t_high).sum())
    n_med = int(((proba_cal >= t_med) & (proba_cal < t_high)).sum())
    n_low = int((proba_cal < t_med).sum())
    print(f"  Thresholds: HIGH={n_high} MED={n_med} LOW={n_low}")

    # Advanced metrics
    ks_result = ks_statistic(y_val.values, proba_cal)
    cm_high = confusion_matrix_at_threshold(y_val.values, proba_cal, threshold=t_high)
    cm_med = confusion_matrix_at_threshold(y_val.values, proba_cal, threshold=t_med)
    ece = expected_calibration_error(y_val.values, proba_cal)
    lift_data = lift_chart_data(y_val.values, proba_cal)
    top_lift = lift_data[0]["lift"] if lift_data else 0

    print(f"  KS statistic: {ks_result['ks_stat']:.4f}")
    print(f"  ECE: {ece:.6f}")
    print(f"  F1@T_high: {cm_high['f1']:.4f}  F1@T_med: {cm_med['f1']:.4f}")
    if lift_data:
        print(f"  Top-decile lift: {top_lift:.1f}x")

    # SHAP
    print(f"  SHAP global importance (sample={shap_sample})...")
    X_sample = X_val.head(shap_sample)
    shap_cat_cols = cat_cols if model_type == "catboost" else None
    global_imp = get_global_importance(model, X_sample, feature_cols, cat_cols=shap_cat_cols, top_n=15)
    for name, val in global_imp[:5]:
        print(f"    {name}: {val:.4f}")

    # Local reason codes for high-risk rows
    high_risk_idx = np.where(proba_cal >= t_high)[0]
    sample_high = min(5, len(high_risk_idx))
    local_codes = {}
    if sample_high > 0:
        high_indices = high_risk_idx[:sample_high]
        X_high = X_val.iloc[high_indices].reset_index(drop=True)
        local_codes = get_local_reason_codes_batch(model, X_high, feature_cols, cat_cols=shap_cat_cols, top_k=5, max_rows=sample_high)

    return {
        "model_type": model_type,
        "model_name": model_name,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "brier_raw": brier_raw,
        "brier_cal": brier_cal,
        "recall_at_p90": r_at_p90,
        "ks_stat": ks_result["ks_stat"],
        "ks_threshold": ks_result["ks_threshold"],
        "ece": ece,
        "top_lift": top_lift,
        "cm_high": cm_high,
        "cm_med": cm_med,
        "n_high": n_high,
        "n_med": n_med,
        "n_low": n_low,
        "global_imp": global_imp,
        "local_codes": local_codes,
        "high_risk_idx": high_risk_idx[:sample_high] if sample_high > 0 else [],
        "proba_cal": proba_cal,
        "calibrator": calibrator,
        "calibration_method": calibration_method,
    }


def _write_single_report(result: dict, feature_cols, cat_cols, docs_dir: Path, t_high, t_med):
    """Write VALIDATION_REPORT.md and MODEL_CARD.md for a single model."""
    m = result
    model_file = "xgboost_fraud.json" if m["model_type"] == "xgboost" else "catboost_fraud.cbm"
    global_lines = "\n".join(f"| {n} | {v:.4f} |" for n, v in m["global_imp"][:15])

    reason_codes_section = _format_reason_codes(m)

    report = f"""# Validation Report — Fraud Early-Warning Model

**Model:** {m['model_name']} classifier (`{model_file}`)
**Val set:** val.parquet (labeled rows only, time-based 2018-2019)

## 1. Metrics (raw probabilities)

| Metric | Value |
|--------|--------|
| PR-AUC | {m['pr_auc']:.4f} |
| ROC-AUC | {m['roc_auc']:.4f} |
| Brier | {m['brier_raw']:.6f} |
| Recall@P90 | {m['recall_at_p90']:.4f} |

## 2. Calibration

- Method: {m['calibration_method']}
- Brier before: {m['brier_raw']:.6f}
- Brier after: {m['brier_cal']:.6f}
- Calibrator saved: `outputs/models/calibrator_{m['model_type']}.joblib`

## 3. Global feature importance (SHAP, mean |impact|)

| Feature | Mean |SHAP| |
|---------|------|
{global_lines}

## 4. Advanced Metrics

| Metric | Value |
|--------|--------|
| KS Statistic | {m['ks_stat']:.4f} |
| KS Threshold | {m['ks_threshold']:.3f} |
| ECE (calibrated) | {m['ece']:.6f} |
| Top-Decile Lift | {m['top_lift']:.1f}x |

### Confusion Matrix @ T_high ({t_high})

| | Predicted Fraud | Predicted Legit |
|---|---|---|
| **Actual Fraud** | {m['cm_high']['tp']} (TP) | {m['cm_high']['fn']} (FN) |
| **Actual Legit** | {m['cm_high']['fp']} (FP) | {m['cm_high']['tn']} (TN) |

Precision={m['cm_high']['precision']:.4f}, Recall={m['cm_high']['recall']:.4f}, F1={m['cm_high']['f1']:.4f}

### Confusion Matrix @ T_med ({t_med})

| | Predicted Fraud | Predicted Legit |
|---|---|---|
| **Actual Fraud** | {m['cm_med']['tp']} (TP) | {m['cm_med']['fn']} (FN) |
| **Actual Legit** | {m['cm_med']['fp']} (FP) | {m['cm_med']['tn']} (TN) |

Precision={m['cm_med']['precision']:.4f}, Recall={m['cm_med']['recall']:.4f}, F1={m['cm_med']['f1']:.4f}

## 5. Threshold policy

- **T_high** = {t_high} → HIGH alert (block/review)
- **T_med** = {t_med} → MEDIUM alert (step-up/watchlist)
- Below T_med → LOW (allow)

On validation set (labeled): HIGH={m['n_high']}, MED={m['n_med']}, LOW={m['n_low']}.
{reason_codes_section}
"""
    (docs_dir / "VALIDATION_REPORT.md").write_text(report, encoding="utf-8")
    print("Wrote docs/VALIDATION_REPORT.md")

    _write_model_card(m, feature_cols, cat_cols, docs_dir)


def _write_comparison_report(results: list[dict], feature_cols, cat_cols, docs_dir: Path, t_high, t_med):
    """Write combined VALIDATION_REPORT.md with comparison table + per-model details."""
    # Sort by PR-AUC descending — champion first
    results = sorted(results, key=lambda r: r["pr_auc"], reverse=True)
    champion = results[0]

    # Comparison table
    header = "| Metric |"
    sep = "|--------|"
    for r in results:
        header += f" {r['model_name']} |"
        sep += "--------|"

    def _row(label, key, fmt=".4f"):
        row = f"| {label} |"
        vals = [r[key] for r in results]
        best = max(vals) if "brier" not in key and "ece" not in key else min(vals)
        for r in results:
            v = r[key]
            formatted = f"{v:{fmt}}"
            if v == best and len(results) > 1:
                formatted = f"**{formatted}**"
            row += f" {formatted} |"
        return row

    comparison_table = "\n".join([
        header, sep,
        _row("PR-AUC", "pr_auc", ".4f"),
        _row("ROC-AUC", "roc_auc", ".4f"),
        _row("Brier (raw)", "brier_raw", ".6f"),
        _row("Brier (calibrated)", "brier_cal", ".6f"),
        _row("Recall@P90", "recall_at_p90", ".4f"),
        _row("KS Statistic", "ks_stat", ".4f"),
        _row("ECE", "ece", ".6f"),
        _row("F1 @ T_high", "cm_high", ".4f").replace("F1 @ T_high", "F1 @ T_high") if False else "",  # placeholder
        _row("Top-Decile Lift", "top_lift", ".1f"),
    ])

    # Build F1 rows manually since they're nested
    f1_high_row = f"| F1 @ T_high ({t_high}) |"
    f1_med_row = f"| F1 @ T_med ({t_med}) |"
    f1_high_vals = [r["cm_high"]["f1"] for r in results]
    f1_med_vals = [r["cm_med"]["f1"] for r in results]
    f1_high_best = max(f1_high_vals)
    f1_med_best = max(f1_med_vals)
    for r in results:
        fh = r["cm_high"]["f1"]
        fm = r["cm_med"]["f1"]
        fh_s = f"{fh:.4f}"
        fm_s = f"{fm:.4f}"
        if fh == f1_high_best and len(results) > 1:
            fh_s = f"**{fh_s}**"
        if fm == f1_med_best and len(results) > 1:
            fm_s = f"**{fm_s}**"
        f1_high_row += f" {fh_s} |"
        f1_med_row += f" {fm_s} |"

    # Rebuild comparison table properly
    comparison_table = "\n".join([
        header, sep,
        _row("PR-AUC", "pr_auc", ".4f"),
        _row("ROC-AUC", "roc_auc", ".4f"),
        _row("Brier (raw)", "brier_raw", ".6f"),
        _row("Brier (calibrated)", "brier_cal", ".6f"),
        _row("Recall@P90", "recall_at_p90", ".4f"),
        _row("KS Statistic", "ks_stat", ".4f"),
        _row("ECE", "ece", ".6f"),
        f1_high_row,
        f1_med_row,
        _row("Top-Decile Lift", "top_lift", ".1f"),
    ])

    # Per-model sections
    model_sections = ""
    for i, m in enumerate(results):
        model_file = "xgboost_fraud.json" if m["model_type"] == "xgboost" else "catboost_fraud.cbm"
        global_lines = "\n".join(f"| {n} | {v:.4f} |" for n, v in m["global_imp"][:15])
        reason_codes_section = _format_reason_codes(m)
        tag = " (Champion)" if i == 0 else ""

        model_sections += f"""
---

## {m['model_name']}{tag}

**Model file:** `{model_file}`
**Calibrator:** `outputs/models/calibrator_{m['model_type']}.joblib`

### Metrics (raw)

| Metric | Value |
|--------|--------|
| PR-AUC | {m['pr_auc']:.4f} |
| ROC-AUC | {m['roc_auc']:.4f} |
| Brier | {m['brier_raw']:.6f} |
| Recall@P90 | {m['recall_at_p90']:.4f} |

### Calibration ({m['calibration_method']})

- Brier before: {m['brier_raw']:.6f} → after: {m['brier_cal']:.6f}

### Global Feature Importance (SHAP)

| Feature | Mean |SHAP| |
|---------|------|
{global_lines}

### Confusion Matrix @ T_high ({t_high})

| | Predicted Fraud | Predicted Legit |
|---|---|---|
| **Actual Fraud** | {m['cm_high']['tp']} (TP) | {m['cm_high']['fn']} (FN) |
| **Actual Legit** | {m['cm_high']['fp']} (FP) | {m['cm_high']['tn']} (TN) |

P={m['cm_high']['precision']:.4f}, R={m['cm_high']['recall']:.4f}, F1={m['cm_high']['f1']:.4f}

### Confusion Matrix @ T_med ({t_med})

| | Predicted Fraud | Predicted Legit |
|---|---|---|
| **Actual Fraud** | {m['cm_med']['tp']} (TP) | {m['cm_med']['fn']} (FN) |
| **Actual Legit** | {m['cm_med']['fp']} (FP) | {m['cm_med']['tn']} (TN) |

P={m['cm_med']['precision']:.4f}, R={m['cm_med']['recall']:.4f}, F1={m['cm_med']['f1']:.4f}

### Threshold Counts

HIGH={m['n_high']}, MED={m['n_med']}, LOW={m['n_low']}
{reason_codes_section}
"""

    report = f"""# Validation Report — Fraud Early-Warning Model (Comparison)

**Val set:** val.parquet (labeled rows only, time-based 2018-2019)
**Champion:** {champion['model_name']} (PR-AUC={champion['pr_auc']:.4f})

## Head-to-Head Comparison

{comparison_table}

> **Bold** = best in column. Lower is better for Brier and ECE; higher is better for all others.
{model_sections}
"""
    (docs_dir / "VALIDATION_REPORT.md").write_text(report, encoding="utf-8")
    print("Wrote docs/VALIDATION_REPORT.md")

    # MODEL_CARD for champion
    _write_model_card(champion, feature_cols, cat_cols, docs_dir)

    # Log which model is champion (serving config in model_config.yaml points to named files)
    logger.info("champion_model", model_type=champion["model_type"],
                calibrator=f"calibrator_{champion['model_type']}.joblib",
                metadata=f"feature_metadata_{champion['model_type']}.json")


def _format_reason_codes(m: dict) -> str:
    """Format local reason codes section for a model result."""
    if not m.get("local_codes"):
        return ""
    lines = []
    high_indices = m.get("high_risk_idx", [])
    proba_cal = m.get("proba_cal")
    for i, (idx_orig, codes) in enumerate(m["local_codes"].items()):
        if i < len(high_indices) and proba_cal is not None:
            prob = float(proba_cal[high_indices[i]])
            lines.append(f"**Row {i + 1}** (calibrated proba={prob:.3f}):")
        else:
            lines.append(f"**Row {i + 1}**:")
        for feat, sh in codes:
            lines.append(f"  - {feat}: {sh:.4f}")
    return "\n### Sample Reason Codes (high-risk rows)\n\n" + "\n\n".join(lines) + "\n"


def _write_model_card(m: dict, feature_cols, cat_cols, docs_dir: Path):
    """Write MODEL_CARD.md for a single model."""
    model_file = "xgboost_fraud.json" if m["model_type"] == "xgboost" else "catboost_fraud.cbm"

    if m["model_type"] == "xgboost":
        load_instructions = (
            'xgb.XGBClassifier(); model.load_model("outputs/models/xgboost_fraud.json")\n'
            '  - Load encoders: `joblib.load("outputs/models/xgboost_target_encoders.joblib")`\n'
            '  - Apply target encoding before prediction: `_target_encode_transform(X, cat_cols, encoders)`'
        )
        cat_method = "Bayesian smoothed target encoding (m=100)"
    else:
        load_instructions = 'catboost.CatBoostClassifier(); model.load_model("outputs/models/catboost_fraud.cbm")'
        cat_method = "Native CatBoost handling"

    model_card = f"""# Model Card — Fraud Early-Warning Classifier

**Model type:** {m['model_name']} binary classifier
**Model file:** `{model_file}`
**Purpose:** Score each transaction with fraud probability; support alert levels (HIGH/MEDIUM/LOW) and reason codes.

## 1. Training data

- **Train:** time-based, date <= 2017-12-31 (~7.2M labeled rows)
- **Target:** is_fraud (0/1); only labeled rows (has_label=True)
- **Features:** {len(feature_cols)} total ({len(cat_cols)} categorical)
- **Categorical encoding:** {cat_method}

## 2. Metrics (validation, 2018-2019)

- **PR-AUC:** {m['pr_auc']:.4f} (primary)
- **ROC-AUC:** {m['roc_auc']:.4f}
- **Brier:** {m['brier_raw']:.6f} (raw), {m['brier_cal']:.6f} (calibrated)
- **Recall@P90:** {m['recall_at_p90']:.4f}
- **KS Statistic:** {m['ks_stat']:.4f}
- **ECE (calibrated):** {m['ece']:.6f}

## 3. Limitations

- Trained on 2010-2017; validated on 2018-2019. Performance may drift on future data.
- Trained on 3M stratified subsample (all fraud + sampled non-fraud) due to RAM constraints; full 7.2M labeled rows available.
- Class imbalance: fraud rate ~0.15%; PR-AUC and Recall@P90 are primary metrics.
- Behavioral features included: rolling velocity/monetary windows (10m/1h/24h/7d), recency, novelty, z-score. Top SHAP: time_since_last_tx_card, first_time_merchant_user.

## 4. Usage

- Load model: `{load_instructions}`
- Load calibrator: `joblib.load("outputs/models/calibrator_{m['model_type']}.joblib")`
- Apply calibrator for calibrated scores, compare to T_high/T_med for alert level
- SHAP reason codes: `get_global_importance()` / `get_local_reason_codes()` from `src.modeling.explain`
"""
    (docs_dir / "MODEL_CARD.md").write_text(model_card, encoding="utf-8")
    print("Wrote docs/MODEL_CARD.md")


def main():
    parser = argparse.ArgumentParser(description="Evaluate model: calibration, SHAP, reports")
    parser.add_argument("--model-type", type=str, default=None,
                        choices=["catboost", "xgboost"],
                        help="Model type (auto-detected if not specified)")
    parser.add_argument("--compare", action="store_true",
                        help="Evaluate both CatBoost and XGBoost and generate comparison report")
    parser.add_argument("--val", type=str, default=None)
    parser.add_argument("--train", type=str, default=None)
    parser.add_argument("--models-dir", type=str, default="outputs/models")
    parser.add_argument("--calibration", type=str, default="isotonic", choices=["isotonic", "platt"])
    parser.add_argument("--shap-sample", type=int, default=2000)
    args = parser.parse_args()

    setup_logging()

    root = Path(__file__).resolve().parents[1]
    paths = get_paths()
    processed = paths.get("data", {}).get("processed", {})

    models_dir = Path(args.models_dir)
    if not models_dir.is_absolute():
        models_dir = root / models_dir

    val_path = Path(args.val or processed.get("val", "data/processed/val.parquet"))
    train_path = Path(args.train or processed.get("train", "data/processed/train.parquet"))
    if not val_path.is_absolute():
        val_path = root / val_path
    if not train_path.is_absolute():
        train_path = root / train_path

    if not val_path.exists():
        logger.error("val_not_found", path=str(val_path))
        sys.exit(1)

    # Load feature columns from per-model metadata (avoids loading all of train.parquet ~10.7M rows).
    # Try catboost first, then xgboost. Falls back to train.parquet only if both are missing.
    feature_cols, cat_cols = None, None
    for model_prefix in ("catboost", "xgboost"):
        meta_path = models_dir / f"feature_metadata_{model_prefix}.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            feature_cols = meta["feature_cols"]
            cat_cols = meta["cat_cols"]
            logger.info("features_from_metadata", total=len(feature_cols),
                        categorical=len(cat_cols), source=meta_path.name)
            break
    if feature_cols is None:
        logger.info("loading_train_for_features", path=str(train_path))
        train_df = pd.read_parquet(train_path)
        feature_cols, cat_cols = get_feature_columns(train_df, target_col="is_fraud")
        del train_df
        logger.info("features", total=len(feature_cols), categorical=len(cat_cols))

    logger.info("loading_eval_data", path=str(val_path))
    val_df = pd.read_parquet(val_path)

    cfg = get_model_config()
    t_high = cfg.get("thresholds", {}).get("high", 0.7)
    t_med = cfg.get("thresholds", {}).get("medium", 0.3)

    docs_dir = root / "docs"
    docs_dir.mkdir(exist_ok=True)

    # MLflow setup
    mlflow_cfg = cfg.get("mlflow", {})
    tracker = MLflowTracker(
        experiment_name=mlflow_cfg.get("experiment_name", "fin-risk-engine"),
        tracking_uri=mlflow_cfg.get("tracking_uri", "mlruns"),
    )
    registry_name = mlflow_cfg.get("registry_model_name", "fraud-detection")

    if args.compare:
        # Evaluate all available models
        model_files = {
            "catboost": models_dir / "catboost_fraud.cbm",
            "xgboost": models_dir / "xgboost_fraud.json",
        }
        available = {k: v for k, v in model_files.items() if v.exists()}
        if not available:
            logger.error("no_models_found", models_dir=str(models_dir))
            sys.exit(1)

        print(f"Found {len(available)} model(s): {', '.join(available.keys())}")
        results = []
        for mtype in available:
            model, encoders = _load_model(mtype, models_dir)
            if model is None:
                continue
            result = _evaluate_single(
                model, mtype, encoders, val_df, feature_cols, cat_cols,
                models_dir, args.calibration, args.shap_sample,
            )
            if result:
                results.append(result)

        if len(results) >= 2:
            _write_comparison_report(results, feature_cols, cat_cols, docs_dir, t_high, t_med)
        elif len(results) == 1:
            _write_single_report(results[0], feature_cols, cat_cols, docs_dir, t_high, t_med)
        else:
            logger.error("no_successful_evaluations")
            sys.exit(1)

        # Log each model's calibrated metrics to MLflow
        champion = sorted(results, key=lambda r: r["pr_auc"], reverse=True)[0] if results else None
        for result in results:
            mtype = result["model_type"]
            with tracker.start_run(
                run_name=f"{mtype}-evaluate",
                tags={"model_type": mtype, "stage": "evaluation"},
            ):
                tracker.log_metrics({
                    "pr_auc": result["pr_auc"],
                    "roc_auc": result["roc_auc"],
                    "brier_raw": result["brier_raw"],
                    "brier_cal": result["brier_cal"],
                    "recall_at_p90": result["recall_at_p90"],
                    "ks_stat": result["ks_stat"],
                    "ece": result["ece"],
                    "f1_at_t_high": result["cm_high"]["f1"],
                    "f1_at_t_med": result["cm_med"]["f1"],
                    "top_decile_lift": result["top_lift"],
                })
                tracker.log_artifact(models_dir / f"calibrator_{mtype}.joblib")
                tracker.log_artifact(docs_dir / "VALIDATION_REPORT.md")
                if champion and mtype == champion["model_type"]:
                    tracker.set_tag("champion", "true")
                    tracker.log_artifact(docs_dir / "MODEL_CARD.md")

        # Promote champion / challenger aliases in the model registry
        if champion:
            tracker.promote_champion(registry_name, champion["model_type"])
            print(f"\nPromoted champion: {champion['model_type']} (PR-AUC={champion['pr_auc']:.4f})")

    else:
        # Single model evaluation (original behavior)
        model_type = args.model_type or _detect_model_type(models_dir)
        logger.info("evaluating", model_type=model_type)

        model, encoders = _load_model(model_type, models_dir)
        if model is None:
            sys.exit(1)

        result = _evaluate_single(
            model, model_type, encoders, val_df, feature_cols, cat_cols,
            models_dir, args.calibration, args.shap_sample,
        )
        if not result:
            sys.exit(1)

        _write_single_report(result, feature_cols, cat_cols, docs_dir, t_high, t_med)

        with tracker.start_run(
            run_name=f"{model_type}-evaluate",
            tags={"model_type": model_type, "stage": "evaluation", "champion": "true"},
        ):
            tracker.log_metrics({
                "pr_auc": result["pr_auc"],
                "roc_auc": result["roc_auc"],
                "brier_raw": result["brier_raw"],
                "brier_cal": result["brier_cal"],
                "recall_at_p90": result["recall_at_p90"],
                "ks_stat": result["ks_stat"],
                "ece": result["ece"],
                "f1_at_t_high": result["cm_high"]["f1"],
                "f1_at_t_med": result["cm_med"]["f1"],
                "top_decile_lift": result["top_lift"],
            })
            tracker.log_artifact(models_dir / f"calibrator_{model_type}.joblib")
            tracker.log_artifact(docs_dir / "VALIDATION_REPORT.md")
            tracker.log_artifact(docs_dir / "MODEL_CARD.md")

        # Single-model path: promote this model as champion
        tracker.promote_champion(registry_name, model_type)
        print(f"\nPromoted champion: {model_type} (PR-AUC={result['pr_auc']:.4f})")

    print("\nDone.")


if __name__ == "__main__":
    main()
