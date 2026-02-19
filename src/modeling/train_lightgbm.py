"""
Train LightGBM fraud classifier.
Same interface as train_catboost for benchmarking.
LightGBM supports native categorical features.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from src.logging_config import get_logger
from src.modeling.train import get_feature_columns, _fill_cat

logger = get_logger(__name__)


def _prep_lgb_cats(X: pd.DataFrame, cat_features: list[str]) -> pd.DataFrame:
    """Convert categoricals to LightGBM-native category dtype."""
    for c in cat_features:
        if c in X.columns:
            X[c] = _fill_cat(X[c]).astype("category")
    return X


def train_lightgbm(
    train_df: pd.DataFrame,
    target_col: str = "is_fraud",
    feature_cols: list[str] | None = None,
    cat_features: list[str] | None = None,
    params: dict[str, Any] | None = None,
    use_labeled_only: bool = True,
    has_label_col: str = "has_label",
    val_df: pd.DataFrame | None = None,
) -> Any:
    """
    Train LightGBM binary classifier on labeled rows.
    Uses native categorical support. Returns fitted model.
    """
    import lightgbm as lgb

    if use_labeled_only and has_label_col in train_df.columns:
        train_df = train_df.loc[train_df[has_label_col]].copy()
    if feature_cols is None or cat_features is None:
        feature_cols, cat_features = get_feature_columns(train_df, target_col=target_col)

    train_df = train_df.dropna(subset=[target_col])
    X = train_df[feature_cols].copy()
    y = train_df[target_col].astype(int)

    X = _prep_lgb_cats(X, cat_features)

    # Fill numeric NaN
    for c in feature_cols:
        if c not in cat_features and c in X.columns and pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].fillna(-999)

    # scale_pos_weight: default 1.0 (no reweighting).
    # PR-AUC eval_metric already handles class imbalance correctly.
    # Use Optuna to search SPW in [1, 50] range if needed.
    cfg = dict(params or {})

    model = lgb.LGBMClassifier(
        n_estimators=cfg.get("n_estimators", cfg.get("iterations", 1000)),
        learning_rate=cfg.get("learning_rate", 0.05),
        max_depth=cfg.get("max_depth", cfg.get("depth", 6)),
        scale_pos_weight=cfg.get("scale_pos_weight", 1.0),
        num_leaves=cfg.get("num_leaves", 63),
        metric="average_precision",
        verbosity=cfg.get("verbosity", -1),
        random_state=42,
        reg_lambda=cfg.get("reg_lambda", 1.0),
        subsample=cfg.get("subsample", 0.8),
        colsample_bytree=cfg.get("colsample_bytree", 0.8),
    )

    fit_kwargs = {
        "categorical_feature": cat_features if cat_features else "auto",
    }
    callbacks = [lgb.log_evaluation(cfg.get("verbose", 100))]

    if val_df is not None:
        X_val, y_val = _prepare_val(val_df, feature_cols, cat_features, target_col, has_label_col)
        fit_kwargs["eval_set"] = [(X_val, y_val)]
        callbacks.append(lgb.early_stopping(cfg.get("early_stopping_rounds", 50), verbose=False))

    fit_kwargs["callbacks"] = callbacks
    model.fit(X, y, **fit_kwargs)
    logger.info("lightgbm_trained", n_rows=len(X), n_features=len(feature_cols))
    return model


def _prepare_val(
    val_df: pd.DataFrame,
    feature_cols: list[str],
    cat_features: list[str],
    target_col: str,
    has_label_col: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare val set with same categorical encoding."""
    if has_label_col in val_df.columns:
        val_df = val_df.loc[val_df[has_label_col]].copy()
    val_df = val_df.dropna(subset=[target_col])
    X = val_df[feature_cols].copy()
    y = val_df[target_col].astype(int)
    X = _prep_lgb_cats(X, cat_features)
    for c in feature_cols:
        if c not in cat_features and c in X.columns and pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].fillna(-999)
    return X, y
