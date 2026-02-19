#!/usr/bin/env python3
"""
Train CatBoost or XGBoost on train.parquet, evaluate on val.parquet (labeled rows only).
Saves model + metadata to outputs/models/.

Usage:
  # CatBoost (default)
  PYTHONPATH=. python scripts/train_model.py

  # XGBoost with tuned params from Optuna
  PYTHONPATH=. python scripts/train_model.py --model-type xgboost --use-tuned

  # XGBoost with custom params file
  PYTHONPATH=. python scripts/train_model.py --model-type xgboost --params-file outputs/tuning/xgboost_best_params.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gc

import joblib
import pandas as pd
import pyarrow.parquet as pq

from src.config import get_paths, get_model_config
from src.logging_config import setup_logging, get_logger
from src.modeling.train import get_feature_columns, evaluate_val

logger = get_logger(__name__)


def _resolve_data_paths(args):
    """Resolve train/val parquet paths from args or config."""
    paths = get_paths()
    processed = paths.get("data", {}).get("processed", {})
    train_path = args.train or processed.get("train", "data/processed/train.parquet")
    val_path = args.val or processed.get("val", "data/processed/val.parquet")
    root = Path(__file__).resolve().parents[1]
    if not Path(train_path).is_absolute():
        train_path = root / train_path
    if not Path(val_path).is_absolute():
        val_path = root / val_path
    return Path(train_path), Path(val_path)


def _load_params(args, model_type: str) -> dict | None:
    """Load model params from --params-file, --use-tuned, or config."""
    if args.params_file:
        data = json.loads(Path(args.params_file).read_text())
        params = data.get("best_params", data)
        logger.info("loaded_params_file", path=args.params_file, n_params=len(params))
        return params

    if args.use_tuned:
        tuned_path = Path(f"outputs/tuning/{model_type}_best_params.json")
        if tuned_path.exists():
            data = json.loads(tuned_path.read_text())
            params = data.get("best_params", {})
            logger.info("loaded_tuned_params", path=str(tuned_path), pr_auc=data.get("best_pr_auc"))
            return params
        else:
            logger.warning("tuned_params_not_found", path=str(tuned_path))
            return None

    if model_type == "catboost":
        cfg = get_model_config()
        return dict(cfg.get("catboost", {}))

    return None


def _train_catboost(train_df, val_df, feature_cols, cat_cols, params, out_dir):
    """Train CatBoost and save artifacts."""
    from src.modeling.train import train_catboost

    logger.info("training_catboost", params=params or "defaults")
    model = train_catboost(
        train_df,
        target_col="is_fraud",
        feature_cols=feature_cols,
        cat_features=cat_cols,
        params=params,
        use_labeled_only=True,
        val_df=val_df,
    )

    # Save model
    model_path = out_dir / "catboost_fraud.cbm"
    model.save_model(str(model_path))
    logger.info("saved_model", path=str(model_path))

    # Save per-model metadata
    meta = {"model_type": "catboost", "feature_cols": feature_cols, "cat_cols": cat_cols}
    model_meta_path = out_dir / "feature_metadata_catboost.json"
    with open(model_meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("saved_metadata", path=str(model_meta_path))

    return model


def _train_xgboost(train_df, val_df, feature_cols, cat_cols, params, out_dir):
    """Train XGBoost and save artifacts (model + target encoders)."""
    from src.modeling.train_xgboost import train_xgboost

    logger.info("training_xgboost", params=params or "defaults")
    model, encoders = train_xgboost(
        train_df,
        target_col="is_fraud",
        feature_cols=feature_cols,
        cat_features=cat_cols,
        params=params,
        use_labeled_only=True,
        val_df=val_df,
    )

    # Save model
    model_path = out_dir / "xgboost_fraud.json"
    model.save_model(str(model_path))
    logger.info("saved_model", path=str(model_path))

    # Save target encoders (needed for inference)
    encoders_path = out_dir / "xgboost_target_encoders.joblib"
    joblib.dump(encoders, str(encoders_path))
    logger.info("saved_encoders", path=str(encoders_path))

    # Save per-model metadata
    meta = {"model_type": "xgboost", "feature_cols": feature_cols, "cat_cols": cat_cols}
    model_meta_path = out_dir / "feature_metadata_xgboost.json"
    with open(model_meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("saved_metadata", path=str(model_meta_path))

    return model, encoders


def _evaluate(model, val_df, feature_cols, cat_cols, model_type, encoders=None):
    """Evaluate model on val set and print metrics."""
    if model_type == "xgboost":
        from src.modeling.train_xgboost import _prepare_val
        X_val, y_val = _prepare_val(val_df, feature_cols, cat_cols, "is_fraud", "has_label", encoders)
        from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, precision_recall_curve
        proba = model.predict_proba(X_val)[:, 1]
        pr_auc = average_precision_score(y_val, proba)
        roc_auc = roc_auc_score(y_val, proba)
        brier = brier_score_loss(y_val, proba)
        precision, recall, _ = precision_recall_curve(y_val, proba)
        mask = precision >= 0.9
        r_at_p90 = float(recall[mask].max()) if mask.any() else float("nan")
        metrics = {"pr_auc": pr_auc, "roc_auc": roc_auc, "brier": brier, "recall_at_precision_90": r_at_p90}
    else:
        metrics = evaluate_val(model, val_df, feature_cols, cat_cols, target_col="is_fraud", has_label_col="has_label")

    logger.info(
        "val_metrics",
        pr_auc=round(metrics["pr_auc"], 6),
        roc_auc=round(metrics["roc_auc"], 6),
        brier=round(metrics["brier"], 6),
        recall_at_p90=round(metrics.get("recall_at_precision_90", float("nan")), 4),
    )
    print("\n=== Validation Metrics ===")
    print(f"  PR-AUC:      {metrics['pr_auc']:.6f}")
    print(f"  ROC-AUC:     {metrics['roc_auc']:.6f}")
    print(f"  Brier:       {metrics['brier']:.6f}")
    print(f"  Recall@P90:  {metrics.get('recall_at_precision_90', float('nan')):.4f}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train fraud detection model")
    parser.add_argument("--model-type", default="catboost", choices=["catboost", "xgboost"],
                        help="Model type to train (default: catboost)")
    parser.add_argument("--train", type=str, default=None, help="Train parquet path")
    parser.add_argument("--val", type=str, default=None, help="Val parquet path")
    parser.add_argument("--out-dir", type=str, default="outputs/models", help="Output directory")
    parser.add_argument("--use-tuned", action="store_true",
                        help="Load best params from outputs/tuning/")
    parser.add_argument("--params-file", type=str, default=None,
                        help="JSON file with model params (overrides --use-tuned)")
    parser.add_argument("--subsample-train", type=int, default=None,
                        help="Max labeled training rows: keeps ALL fraud + samples non-fraud. "
                             "Use when full dataset causes OOM (e.g. --subsample-train 3000000)")
    args = parser.parse_args()

    setup_logging()

    train_path, val_path = _resolve_data_paths(args)

    if not train_path.exists():
        logger.error("train_not_found", path=str(train_path))
        print(f"Train file not found: {train_path}")
        print("Run: PYTHONPATH=. python scripts/build_features_and_splits.py --skip-behavioral")
        sys.exit(1)

    # Load data
    # When subsampling: use pyarrow filter pushdown to read only labeled rows (7.2M vs 10.7M total)
    # This avoids holding 10.7M + 3M + 1.4M (val) simultaneously (~12GB peak)
    if args.subsample_train is not None:
        logger.info("loading_train_labeled_only", path=str(train_path),
                    note="filter pushdown for has_label=True to save RAM")
        train = pq.read_table(train_path, filters=[("has_label", "=", True)]).to_pandas()
        n_labeled = len(train)
    else:
        logger.info("loading_train", path=str(train_path))
        train = pd.read_parquet(train_path)
        n_labeled = int(train["has_label"].sum()) if "has_label" in train.columns else len(train)
    logger.info("train_loaded", rows=len(train), labeled=n_labeled)

    # Stratified subsampling: keep ALL fraud + sample non-fraud (avoids OOM on full 7.2M rows)
    if args.subsample_train is not None:
        fraud = train[train["is_fraud"] == 1]
        non_fraud = train[train["is_fraud"] == 0]
        n_target = args.subsample_train
        n_non_fraud = max(0, min(n_target - len(fraud), len(non_fraud)))
        non_fraud_sample = non_fraud.sample(n=n_non_fraud, random_state=42)
        n_fraud_new = len(fraud)
        train = pd.concat([fraud, non_fraud_sample]).sample(
            frac=1, random_state=42
        ).reset_index(drop=True)
        del fraud, non_fraud, non_fraud_sample
        gc.collect()
        logger.info(
            "subsampled_train",
            total_rows=len(train),
            labeled=len(train),
            fraud=n_fraud_new,
            non_fraud=n_non_fraud,
        )
        print(f"  Subsampled train: {len(train):,} rows "
              f"({n_fraud_new:,} fraud + {n_non_fraud:,} non-fraud, "
              f"ratio 1:{n_non_fraud // max(n_fraud_new, 1)}")

    val = None
    if val_path.exists():
        logger.info("loading_val", path=str(val_path))
        val = pd.read_parquet(val_path)
        n_val_labeled = int(val["has_label"].sum()) if "has_label" in val.columns else len(val)
        logger.info("val_loaded", rows=len(val), labeled=n_val_labeled)
    else:
        logger.warning("val_not_found", path=str(val_path))

    # Feature columns
    feature_cols, cat_cols = get_feature_columns(train, target_col="is_fraud")
    logger.info("features", total=len(feature_cols), categorical=len(cat_cols))

    # Load params
    params = _load_params(args, args.model_type)

    # Output directory
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = Path(__file__).resolve().parents[1] / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Train
    encoders = None
    if args.model_type == "xgboost":
        model, encoders = _train_xgboost(train, val, feature_cols, cat_cols, params, out_dir)
    else:
        model = _train_catboost(train, val, feature_cols, cat_cols, params, out_dir)

    # Evaluate
    if val is not None:
        _evaluate(model, val, feature_cols, cat_cols, args.model_type, encoders)

    print(f"\nArtifacts saved to: {out_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()
