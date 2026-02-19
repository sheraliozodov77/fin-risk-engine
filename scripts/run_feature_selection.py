"""
Run feature selection pipeline: correlation filter -> VIF -> permutation importance.
Usage:
    PYTHONPATH=. python scripts/run_feature_selection.py [--corr-threshold 0.95] [--max-vif 10.0]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.logging_config import setup_logging, get_logger
from src.features.selection import correlation_filter, vif_filter
from src.modeling.train import get_feature_columns

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Feature selection pipeline")
    parser.add_argument("--data-dir", default="data/gold", help="Directory with train.parquet")
    parser.add_argument("--corr-threshold", type=float, default=0.95)
    parser.add_argument("--max-vif", type=float, default=10.0)
    parser.add_argument("--sample-rows", type=int, default=50000, help="Sample rows for VIF (speed)")
    args = parser.parse_args()

    setup_logging()

    data_dir = Path(args.data_dir)
    train_path = data_dir / "train.parquet"
    if not train_path.exists():
        logger.error("train_parquet_not_found", path=str(train_path))
        sys.exit(1)

    logger.info("loading_train_data", path=str(train_path))
    df = pd.read_parquet(train_path)
    logger.info("loaded", rows=len(df), cols=len(df.columns))

    feature_cols, cat_cols = get_feature_columns(df)
    logger.info("initial_features", total=len(feature_cols), numeric=len(feature_cols) - len(cat_cols), categorical=len(cat_cols))

    # Step 1: Correlation filter (numeric only)
    numeric_features = [f for f in feature_cols if f not in cat_cols]
    surviving_numeric = correlation_filter(df, numeric_features, threshold=args.corr_threshold)
    dropped_corr = set(numeric_features) - set(surviving_numeric)

    # Step 2: VIF filter
    sample = df.sample(n=min(args.sample_rows, len(df)), random_state=42) if len(df) > args.sample_rows else df
    try:
        surviving_numeric = vif_filter(sample, surviving_numeric, max_vif=args.max_vif)
        dropped_vif = set(numeric_features) - set(surviving_numeric) - dropped_corr
    except ImportError:
        logger.warning("vif_skipped", reason="pip install statsmodels")
        dropped_vif = set()

    # Final surviving = surviving numeric + all categoricals
    final_features = surviving_numeric + cat_cols
    logger.info(
        "feature_selection_summary",
        original=len(feature_cols),
        dropped_corr=len(dropped_corr),
        dropped_vif=len(dropped_vif),
        final=len(final_features),
    )

    print("\n=== Feature Selection Results ===")
    print(f"Original features: {len(feature_cols)}")
    print(f"Dropped (correlation > {args.corr_threshold}): {len(dropped_corr)}")
    if dropped_corr:
        for f in sorted(dropped_corr):
            print(f"  - {f}")
    print(f"Dropped (VIF > {args.max_vif}): {len(dropped_vif)}")
    if dropped_vif:
        for f in sorted(dropped_vif):
            print(f"  - {f}")
    print(f"Final features: {len(final_features)}")
    print("\nSurviving features:")
    for f in final_features:
        tag = " (cat)" if f in cat_cols else ""
        print(f"  {f}{tag}")


if __name__ == "__main__":
    main()
