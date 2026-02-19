#!/usr/bin/env python3
"""
Run batch predictions on test (or val) data. Loads model + calibrator, scores each row,
writes risk_score, level, and optional reason codes to CSV.

Usage:
  PYTHONPATH=. python scripts/score_test_data.py
  PYTHONPATH=. python scripts/score_test_data.py --nrows 1000 --out outputs/predictions_test.csv
  PYTHONPATH=. python scripts/score_test_data.py --data data/processed/val.parquet
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.config import get_paths
from src.serving.app import load_artifacts, _score_one


def main():
    parser = argparse.ArgumentParser(description="Score test/val data with trained model")
    parser.add_argument("--data", type=str, default=None, help="Parquet path (default: test.parquet)")
    parser.add_argument("--nrows", type=int, default=None, help="Max rows to score (default: all)")
    parser.add_argument("--out", type=str, default=None, help="Output CSV (default: outputs/artifacts/predictions.csv)")
    parser.add_argument("--reason-codes", action="store_true", help="Include top-3 reason codes per row (wider CSV)")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    paths = get_paths()
    processed = paths.get("data", {}).get("processed", {})
    test_path = args.data or processed.get("test", "data/processed/test.parquet")
    test_path = root / test_path if not Path(test_path).is_absolute() else Path(test_path)
    if not test_path.exists():
        print("Data not found:", test_path)
        sys.exit(1)

    out_path = args.out or str(paths.get("outputs", {}).get("artifacts", "outputs/artifacts") + "/predictions.csv")
    out_path = root / out_path if not Path(out_path).is_absolute() else Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading model and calibrator...")
    model, calibrator, feature_cols, cat_cols, t_high, t_med, top_k = load_artifacts()
    print("  Features:", len(feature_cols), "Categorical:", len(cat_cols))

    print("Loading data:", test_path)
    df = pd.read_parquet(test_path)
    if args.nrows is not None:
        df = df.head(args.nrows)
    print("  Rows:", len(df))

    # If default test set is empty, try val.parquet so you get predictions without changing CLI
    if len(df) == 0 and args.data is None:
        val_path = root / processed.get("val", "data/processed/val.parquet")
        if val_path.exists():
            df = pd.read_parquet(val_path)
            if args.nrows is not None:
                df = df.head(args.nrows)
            print("Test set empty; using val.parquet instead. Rows:", len(df))
    if len(df) == 0:
        print("No rows to score. Wrote empty CSV.")
        pd.DataFrame(columns=["risk_score", "level"]).to_csv(out_path, index=False)
        print("Tip: use --data data/processed/val.parquet to score validation data.")
        print("Done.")
        return

    # Build feature dict per row (missing cols -> None so _score_one fills defaults)
    rows = []
    top_k_arg = 3 if args.reason_codes else 0
    for i in range(len(df)):
        row = df.iloc[i]
        features = {c: (row[c] if c in df.columns else None) for c in feature_cols}
        out = _score_one(model, calibrator, features, feature_cols, cat_cols, t_high, t_med, top_k_arg)
        rec = {"risk_score": out["risk_score"], "level": out["level"]}
        if args.reason_codes and out.get("reason_codes"):
            for j, rc in enumerate(out["reason_codes"][:3]):
                rec[f"reason_{j+1}_feature"] = rc["feature"]
                rec[f"reason_{j+1}_impact"] = rc["impact"]
        rows.append(rec)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False)
    print("Wrote", out_path)
    print("  HIGH:", (out_df["level"] == "HIGH").sum(), "MED:", (out_df["level"] == "MEDIUM").sum(), "LOW:", (out_df["level"] == "LOW").sum())
    print("Done.")


if __name__ == "__main__":
    main()
