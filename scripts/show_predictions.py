#!/usr/bin/env python3
"""
Show predictions on a small sample: raw proba, calibrated proba, level (HIGH/MED/LOW), and top reason codes.
Usage: PYTHONPATH=. python scripts/show_predictions.py [--n 10]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd
import catboost as cb

from src.config import get_paths, get_model_config
from src.modeling.train import get_feature_columns
from src.modeling.calibrate import apply_calibrator
from src.modeling.explain import get_local_reason_codes


def _prepare_X(df: pd.DataFrame, feature_cols: list[str], cat_cols: list[str]):
    X = df.reindex(columns=feature_cols).copy()
    for c in feature_cols:
        if c in cat_cols and c in X.columns:
            X[c] = X[c].fillna("__missing__").astype(str).replace("nan", "__missing__")
        elif c in X.columns and pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].fillna(-999)
    return X


def main():
    parser = argparse.ArgumentParser(description="Show predictions sample")
    parser.add_argument("--n", type=int, default=10, help="Number of rows to show")
    parser.add_argument("--val", type=str, default=None)
    args = parser.parse_args()

    paths = get_paths()
    root = Path(__file__).resolve().parents[1]
    val_path = args.val or paths.get("data", {}).get("processed", {}).get("val", "data/processed/val.parquet")
    model_path = root / paths.get("outputs", {}).get("models", "outputs/models") / "catboost_fraud.cbm"
    cal_path = root / paths.get("outputs", {}).get("models", "outputs/models") / "calibrator.joblib"
    if not Path(val_path).is_absolute():
        val_path = root / val_path
    val_path = Path(val_path)

    if not model_path.exists() or not val_path.exists():
        print("Model or val not found. Run train_model.py and ensure val.parquet exists.")
        sys.exit(1)

    val = pd.read_parquet(val_path)
    val_labeled = val.loc[val["has_label"]] if "has_label" in val.columns else val
    val_labeled = val_labeled.dropna(subset=["is_fraud"]).head(args.n)
    if len(val_labeled) == 0:
        print("No labeled rows in val.")
        sys.exit(1)

    feature_cols, cat_cols = get_feature_columns(val, target_col="is_fraud")
    X = _prepare_X(val_labeled, feature_cols, cat_cols)

    model = cb.CatBoostClassifier()
    model.load_model(str(model_path))
    proba_raw = model.predict_proba(X)[:, 1]
    calibrator = joblib.load(cal_path) if cal_path.exists() else None
    proba_cal = apply_calibrator(proba_raw, calibrator) if calibrator is not None else proba_raw

    cfg = get_model_config()
    t_high = cfg.get("thresholds", {}).get("high", 0.7)
    t_med = cfg.get("thresholds", {}).get("medium", 0.3)

    def level(p):
        if p >= t_high:
            return "HIGH"
        if p >= t_med:
            return "MED"
        return "LOW"

    print("Predictions (first %d labeled val rows):" % len(val_labeled))
    print("-" * 60)
    for i in range(len(val_labeled)):
        y_true = int(val_labeled.iloc[i]["is_fraud"])
        p_raw = float(proba_raw[i])
        p_cal = float(proba_cal[i])
        lev = level(p_cal)
        codes = get_local_reason_codes(model, X, i, feature_cols, cat_cols=cat_cols, top_k=3)
        codes_str = "; ".join("%s=%.3f" % (f, v) for f, v in codes) if codes else "—"
        print("  row %d: is_fraud=%d  raw=%.4f  cal=%.4f  level=%s" % (i + 1, y_true, p_raw, p_cal, lev))
        print("    top-3: %s" % codes_str)
    print("-" * 60)
    print("Done. Run evaluate_model.py to refresh VALIDATION_REPORT.md (SHAP + reason codes).")


if __name__ == "__main__":
    main()
