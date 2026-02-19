"""
Hyperparameter tuning with Optuna.
Usage:
    PYTHONPATH=. python scripts/run_tuning.py --model-type catboost --n-trials 50
    PYTHONPATH=. python scripts/run_tuning.py --model-type xgboost --n-trials 30
    PYTHONPATH=. python scripts/run_tuning.py --model-type catboost --n-trials 20 --subsample-train 3000000
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from src.logging_config import setup_logging, get_logger
from src.modeling.train import get_feature_columns
from src.modeling.tuning import run_tuning, OBJECTIVES

logger = get_logger(__name__)


def _load_train(train_path: Path, subsample: int | None) -> pd.DataFrame:
    """Load train parquet. If subsample set, filter to labeled only + stratified subsample."""
    if subsample is not None:
        # Filter pushdown: only labeled rows (7.2M vs 10.7M) -- saves ~3GB RAM
        logger.info("loading_train_labeled_only", path=str(train_path),
                    note="filter pushdown for has_label=True")
        df = pq.read_table(train_path, filters=[("has_label", "=", True)]).to_pandas()
        fraud = df[df["is_fraud"] == 1]
        non_fraud = df[df["is_fraud"] == 0]
        n_non_fraud = max(0, min(subsample - len(fraud), len(non_fraud)))
        non_fraud_sample = non_fraud.sample(n=n_non_fraud, random_state=42)
        n_fraud = len(fraud)
        df = pd.concat([fraud, non_fraud_sample]).sample(frac=1, random_state=42).reset_index(drop=True)
        del fraud, non_fraud, non_fraud_sample
        gc.collect()
        logger.info("subsampled_train", total_rows=len(df), fraud=n_fraud, non_fraud=n_non_fraud)
        print(f"  Train subsampled: {len(df):,} rows ({n_fraud:,} fraud + {n_non_fraud:,} non-fraud)")
        return df
    else:
        logger.info("loading_train", path=str(train_path))
        return pd.read_parquet(train_path)


def main():
    parser = argparse.ArgumentParser(description="Optuna hyperparameter tuning")
    parser.add_argument("--model-type", default="catboost", choices=list(OBJECTIVES.keys()))
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--output-dir", default="outputs/tuning")
    parser.add_argument("--subsample-train", type=int, default=None,
                        help="Max labeled training rows (keeps ALL fraud + samples non-fraud). "
                             "Use to avoid OOM on full 7.2M dataset (e.g. --subsample-train 3000000)")
    args = parser.parse_args()

    setup_logging()

    data_dir = Path(args.data_dir)
    train_path = data_dir / "train.parquet"
    val_path = data_dir / "val.parquet"

    for p in (train_path, val_path):
        if not p.exists():
            logger.error("file_not_found", path=str(p))
            sys.exit(1)

    logger.info("loading_data")
    train_df = _load_train(train_path, args.subsample_train)
    val_df = pd.read_parquet(val_path)
    logger.info("loaded", train_rows=len(train_df), val_rows=len(val_df))

    feature_cols, cat_features = get_feature_columns(train_df)
    logger.info("features", total=len(feature_cols), categorical=len(cat_features))

    logger.info("starting_tuning", model_type=args.model_type, n_trials=args.n_trials)
    result = run_tuning(
        model_type=args.model_type,
        train_df=train_df,
        val_df=val_df,
        feature_cols=feature_cols,
        cat_features=cat_features,
        n_trials=args.n_trials,
    )

    # Save best params
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    params_file = out_dir / f"{args.model_type}_best_params.json"
    with open(params_file, "w") as f:
        json.dump({
            "model_type": args.model_type,
            "best_pr_auc": result["best_pr_auc"],
            "best_params": result["best_params"],
            "n_trials": args.n_trials,
        }, f, indent=2)

    print(f"\n=== Tuning Results: {args.model_type} ===")
    print(f"Best PR-AUC: {result['best_pr_auc']:.6f}")
    print(f"Best params: {json.dumps(result['best_params'], indent=2)}")
    print(f"Saved to: {params_file}")


if __name__ == "__main__":
    main()
