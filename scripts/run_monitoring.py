#!/usr/bin/env python3
"""
Run monitoring: data drift (PSI, missing rate, new merchants) and performance drift
(PR-AUC by period, alert precision). Reference = train; current = val (or test).

Usage:
  PYTHONPATH=. python scripts/run_monitoring.py
  PYTHONPATH=. python scripts/run_monitoring.py --current data/processed/test.parquet
  PYTHONPATH=. python scripts/run_monitoring.py --period Q
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.config import get_paths
from src.modeling.train import get_feature_columns
from src.monitoring.drift import compute_drift
from src.monitoring.performance import performance_by_period


def main():
    parser = argparse.ArgumentParser(description="Run drift and performance monitoring")
    parser.add_argument("--reference", type=str, default=None, help="Reference (baseline) parquet (default: train)")
    parser.add_argument("--current", type=str, default=None, help="Current window parquet (default: val)")
    parser.add_argument("--period", type=str, default="M", help="Time period for PR-AUC: M=month, Q=quarter, W=week")
    parser.add_argument("--out", type=str, default=None, help="Optional: write report to this path")
    args = parser.parse_args()

    paths = get_paths()
    root = Path(__file__).resolve().parents[1]
    processed = paths.get("data", {}).get("processed", {})
    train_path = processed.get("train", "data/processed/train.parquet")
    val_path = processed.get("val", "data/processed/val.parquet")
    if not Path(train_path).is_absolute():
        train_path = root / train_path
        val_path = root / val_path
    ref_path = args.reference or train_path
    cur_path = args.current or val_path
    ref_path = Path(ref_path) if not Path(ref_path).is_absolute() else Path(ref_path)
    cur_path = Path(cur_path) if not Path(cur_path).is_absolute() else Path(cur_path)

    if not ref_path.exists():
        print("Reference not found:", ref_path)
        sys.exit(1)
    if not cur_path.exists():
        print("Current not found:", cur_path)
        sys.exit(1)

    print("Loading reference:", ref_path)
    ref_df = pd.read_parquet(ref_path)
    print("Loading current:", cur_path)
    cur_df = pd.read_parquet(cur_path)

    # Data drift
    print("\n--- Data drift (reference vs current) ---")
    drift = compute_drift(
        ref_df, cur_df,
        numeric_cols=["amount"],
        categorical_cols=["card_brand", "mcc_str"],
        new_category_cols=["merchant_id", "mcc_str"],
    )
    for col, psi in drift["psi"].items():
        print("  PSI %s: %.4f" % (col, psi))
    for col, rate in drift["missing_rate_current"].items():
        print("  Missing rate (current) %s: %.4f" % (col, rate))
    for col, v in drift["new_share"].items():
        print("  New share %s: %.4f (count_new=%d)" % (col, v["share"], v["count_new"]))

    # Performance by period (requires model + labeled current)
    model_path = paths.get("outputs", {}).get("models", "outputs/models")
    model_path = root / model_path / "catboost_fraud.cbm" if not Path(model_path).is_absolute() else Path(model_path) / "catboost_fraud.cbm"
    if model_path.exists():
        try:
            import catboost as cb
            model = cb.CatBoostClassifier()
            model.load_model(str(model_path))
            feature_cols, cat_cols = get_feature_columns(ref_df, target_col="is_fraud")
            print("\n--- Performance by period (%s) ---" % args.period)
            perf = performance_by_period(
                cur_df, model, feature_cols, cat_cols,
                time_col="date", target_col="is_fraud", has_label_col="has_label",
                period=args.period,
            )
            if len(perf) > 0:
                for _, row in perf.iterrows():
                    print("  %s: n=%d pr_auc=%.4f HIGH=%d prec_high=%.4f MED=%d prec_med=%.4f" % (
                        row["period_start"], row["n_rows"], row["pr_auc"],
                        row["n_high"], row["precision_high"] if pd.notna(row["precision_high"]) else 0,
                        row["n_med"], row["precision_med"] if pd.notna(row["precision_med"]) else 0))
            else:
                print("  No periods with enough labeled rows.")
        except Exception as e:
            print("  Performance by period skipped:", e)
    else:
        print("\n--- Performance by period: skipped (model not found) ---")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        lines = ["# Monitoring report\n", "## Data drift\n"]
        for col, psi in drift["psi"].items():
            lines.append("- PSI %s: %.4f\n" % (col, psi))
        for col, rate in drift["missing_rate_current"].items():
            lines.append("- Missing (current) %s: %.4f\n" % (col, rate))
        for col, v in drift["new_share"].items():
            lines.append("- New share %s: %.4f (count_new=%d)\n" % (col, v["share"], v["count_new"]))
        out_path.write_text("".join(lines), encoding="utf-8")
        print("Wrote", out_path)
    print("Done.")


if __name__ == "__main__":
    main()
