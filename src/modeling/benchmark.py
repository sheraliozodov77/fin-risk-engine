"""
Unified model benchmarking: train CatBoost, XGBoost, LightGBM and compare.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)

from src.logging_config import get_logger
from src.modeling.train import get_feature_columns, _prepare_X

logger = get_logger(__name__)

_PARTIAL_RESULTS_PATH = Path("outputs/benchmark/partial_results.json")


@dataclass
class ModelResult:
    """Stores benchmark results for a single model."""
    name: str
    model: Any
    pr_auc: float
    roc_auc: float
    brier: float
    recall_at_precision_90: float
    train_time_sec: float
    params: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)


def _evaluate(
    model: Any,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> dict[str, float]:
    """Compute metrics from model predictions."""
    proba = model.predict_proba(X_val)[:, 1]
    pr_auc = average_precision_score(y_val, proba)
    roc_auc = roc_auc_score(y_val, proba)
    brier = brier_score_loss(y_val, proba)
    precision, recall, _ = precision_recall_curve(y_val, proba)
    mask = precision >= 0.9
    r_at_p90 = float(recall[mask].max()) if mask.any() else float("nan")
    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "brier": brier,
        "recall_at_precision_90": r_at_p90,
    }


def _save_partial(results: list[ModelResult]) -> None:
    """Save results after each model completes so nothing is lost on crash."""
    _PARTIAL_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in results:
        rows.append({
            "model": r.name,
            "pr_auc": r.pr_auc,
            "roc_auc": r.roc_auc,
            "brier": r.brier,
            "recall_at_precision_90": r.recall_at_precision_90,
            "train_time_sec": r.train_time_sec,
        })
    _PARTIAL_RESULTS_PATH.write_text(json.dumps(rows, indent=2))
    logger.info("partial_results_saved", path=str(_PARTIAL_RESULTS_PATH), n_models=len(rows))


def compare_models(results: list[ModelResult]) -> pd.DataFrame:
    """
    Create a comparison DataFrame sorted by PR-AUC (desc).
    """
    rows = []
    for r in results:
        rows.append({
            "Model": r.name,
            "PR-AUC": round(r.pr_auc, 6),
            "ROC-AUC": round(r.roc_auc, 6),
            "Brier": round(r.brier, 6),
            "Recall@P90": round(r.recall_at_precision_90, 4),
            "Train Time (s)": round(r.train_time_sec, 1),
        })
    df = pd.DataFrame(rows).sort_values("PR-AUC", ascending=False).reset_index(drop=True)
    return df


def run_benchmark(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    cat_features: list[str] | None = None,
    target_col: str = "is_fraud",
    has_label_col: str = "has_label",
    catboost_params: dict | None = None,
    xgboost_params: dict | None = None,
    lightgbm_params: dict | None = None,
    skip_catboost: bool = False,
    skip_xgboost: bool = False,
    skip_lightgbm: bool = False,
) -> list[ModelResult]:
    """
    Train selected models and return list of ModelResult.
    Saves partial results after each model in case of crash.
    """
    if feature_cols is None or cat_features is None:
        feature_cols, cat_features = get_feature_columns(train_df, target_col=target_col)

    results = []

    # --- CatBoost ---
    if not skip_catboost:
        try:
            from src.modeling.train import train_catboost
            logger.info("benchmark_training", model="CatBoost")
            t0 = time.time()
            cb_model = train_catboost(
                train_df, target_col=target_col,
                feature_cols=feature_cols, cat_features=cat_features,
                params=catboost_params, val_df=val_df,
            )
            train_time = time.time() - t0

            # Prepare val for evaluation
            vdf = val_df.loc[val_df[has_label_col]].copy() if has_label_col in val_df.columns else val_df.copy()
            vdf = vdf.dropna(subset=[target_col])
            X_val = _prepare_X(vdf, feature_cols, cat_features)
            y_val = vdf[target_col].astype(int)

            metrics = _evaluate(cb_model, X_val, y_val)
            results.append(ModelResult(
                name="CatBoost", model=cb_model, train_time_sec=train_time,
                params=catboost_params or {}, **metrics,
            ))
            logger.info("benchmark_done", model="CatBoost", pr_auc=round(metrics["pr_auc"], 6))
            _save_partial(results)
        except Exception as e:
            logger.warning("benchmark_failed", model="CatBoost", error=str(e))

    # --- XGBoost ---
    if not skip_xgboost:
        try:
            from src.modeling.train_xgboost import train_xgboost
            logger.info("benchmark_training", model="XGBoost")
            t0 = time.time()
            xgb_model, encoders = train_xgboost(
                train_df, target_col=target_col,
                feature_cols=feature_cols, cat_features=cat_features,
                params=xgboost_params, val_df=val_df,
            )
            train_time = time.time() - t0

            from src.modeling.train_xgboost import _prepare_val
            X_val, y_val = _prepare_val(val_df, feature_cols, cat_features, target_col, has_label_col, encoders)

            metrics = _evaluate(xgb_model, X_val, y_val)
            results.append(ModelResult(
                name="XGBoost", model=xgb_model, train_time_sec=train_time,
                params=xgboost_params or {}, extra={"encoders": encoders}, **metrics,
            ))
            logger.info("benchmark_done", model="XGBoost", pr_auc=round(metrics["pr_auc"], 6))
            _save_partial(results)
        except Exception as e:
            logger.warning("benchmark_failed", model="XGBoost", error=str(e))

    # --- LightGBM ---
    if not skip_lightgbm:
        try:
            from src.modeling.train_lightgbm import train_lightgbm
            logger.info("benchmark_training", model="LightGBM")
            t0 = time.time()
            lgb_model = train_lightgbm(
                train_df, target_col=target_col,
                feature_cols=feature_cols, cat_features=cat_features,
                params=lightgbm_params, val_df=val_df,
            )
            train_time = time.time() - t0

            from src.modeling.train_lightgbm import _prepare_val as lgb_prep_val
            X_val, y_val = lgb_prep_val(val_df, feature_cols, cat_features, target_col, has_label_col)

            metrics = _evaluate(lgb_model, X_val, y_val)
            results.append(ModelResult(
                name="LightGBM", model=lgb_model, train_time_sec=train_time,
                params=lightgbm_params or {}, **metrics,
            ))
            logger.info("benchmark_done", model="LightGBM", pr_auc=round(metrics["pr_auc"], 6))
            _save_partial(results)
        except Exception as e:
            logger.warning("benchmark_failed", model="LightGBM", error=str(e))

    return results
