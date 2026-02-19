"""
Run 3-model benchmark: CatBoost vs XGBoost vs LightGBM.
Usage:
    PYTHONPATH=. python scripts/run_benchmark.py --data-dir data/processed
    PYTHONPATH=. python scripts/run_benchmark.py --models xgboost --use-tuned
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from src.logging_config import setup_logging, get_logger
from src.modeling.train import get_feature_columns
from src.modeling.benchmark import run_benchmark, compare_models

logger = get_logger(__name__)


def _load_tuned_params(model_type: str, tuning_dir: str = "outputs/tuning") -> dict | None:
    """Load best params from Optuna tuning results."""
    path = Path(tuning_dir) / f"{model_type}_best_params.json"
    if path.exists():
        data = json.loads(path.read_text())
        logger.info("loaded_tuned_params", model=model_type, pr_auc=data.get("best_pr_auc"))
        return data.get("best_params", {})
    return None


def main():
    parser = argparse.ArgumentParser(description="3-model benchmark")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--output-dir", default="outputs/benchmark")
    parser.add_argument("--models", default="catboost,xgboost,lightgbm",
                        help="Comma-separated models to benchmark")
    parser.add_argument("--use-tuned", action="store_true",
                        help="Load best params from outputs/tuning/ if available")
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
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    logger.info("loaded", train_rows=len(train_df), val_rows=len(val_df))

    feature_cols, cat_features = get_feature_columns(train_df)
    logger.info("features", total=len(feature_cols), categorical=len(cat_features))

    enabled = set(args.models.split(","))

    # Load tuned params if requested
    catboost_params = _load_tuned_params("catboost") if args.use_tuned else None
    xgboost_params = _load_tuned_params("xgboost") if args.use_tuned else None
    lightgbm_params = _load_tuned_params("lightgbm") if args.use_tuned else None

    results = run_benchmark(
        train_df, val_df,
        feature_cols=feature_cols,
        cat_features=cat_features,
        catboost_params=catboost_params,
        xgboost_params=xgboost_params,
        lightgbm_params=lightgbm_params,
        skip_catboost="catboost" not in enabled,
        skip_xgboost="xgboost" not in enabled,
        skip_lightgbm="lightgbm" not in enabled,
    )

    if not results:
        logger.error("no_models_succeeded")
        sys.exit(1)

    comparison = compare_models(results)

    print("\n" + "=" * 70)
    print("MODEL BENCHMARK RESULTS")
    print("=" * 70)
    print(comparison.to_string(index=False))
    print("=" * 70)

    # Save results
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(out_dir / "benchmark_comparison.csv", index=False)
    logger.info("benchmark_saved", path=str(out_dir / "benchmark_comparison.csv"))


if __name__ == "__main__":
    main()
