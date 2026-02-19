"""
Train XGBoost fraud classifier.
Same interface as train_catboost for benchmarking.

Uses smoothed target encoding for categoricals instead of LabelEncoder.
Target encoding converts each category to its Bayesian-smoothed fraud rate,
giving XGBoost meaningful signal instead of arbitrary integers.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.logging_config import get_logger
from src.modeling.train import get_feature_columns, _fill_cat

logger = get_logger(__name__)

# Minimum samples per category before full trust in category mean.
# Below this, shrink toward global mean (Bayesian smoothing).
_SMOOTHING_WEIGHT = 100


def _target_encode_fit(
    X: pd.DataFrame,
    y: pd.Series,
    cat_features: list[str],
    smoothing: int = _SMOOTHING_WEIGHT,
) -> dict:
    """
    Smoothed target encoding: encode each category as its Bayesian-smoothed
    target rate from training data.

    Formula: encoded = (n * cat_mean + m * global_mean) / (n + m)
      - n = number of samples in category
      - m = smoothing weight (prior strength)
      - Rare categories shrink toward global mean (prevents overfitting)
      - Unseen categories at inference → global mean

    Returns dict of {col: {'mapping': {cat_value: encoded_float}, 'global_mean': float}}.
    """
    global_mean = float(y.mean())
    encoders = {}

    for c in cat_features:
        if c not in X.columns:
            continue

        X[c] = _fill_cat(X[c])

        # Group stats
        stats = pd.DataFrame({"target": y, "cat": X[c]}).groupby("cat")["target"]
        cat_mean = stats.mean()
        cat_count = stats.count()

        # Bayesian smoothing
        smoothed = (cat_count * cat_mean + smoothing * global_mean) / (cat_count + smoothing)

        mapping = smoothed.to_dict()
        encoders[c] = {"mapping": mapping, "global_mean": global_mean}
        X[c] = X[c].map(mapping).astype(np.float64)

    return encoders


def _target_encode_transform(
    X: pd.DataFrame,
    cat_features: list[str],
    encoders: dict,
) -> pd.DataFrame:
    """Apply fitted target encoders. Unseen categories → global mean."""
    for c in cat_features:
        if c not in X.columns or c not in encoders:
            continue
        enc = encoders[c]
        X[c] = _fill_cat(X[c])
        X[c] = X[c].map(enc["mapping"]).fillna(enc["global_mean"]).astype(np.float64)
    return X


def train_xgboost(
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
    Train XGBoost binary classifier on labeled rows.
    Categoricals are target-encoded. Returns (model, target_encoders).
    """
    import xgboost as xgb

    if use_labeled_only and has_label_col in train_df.columns:
        train_df = train_df.loc[train_df[has_label_col]].copy()
    if feature_cols is None or cat_features is None:
        feature_cols, cat_features = get_feature_columns(train_df, target_col=target_col)

    train_df = train_df.dropna(subset=[target_col])
    X = train_df[feature_cols].copy()
    y = train_df[target_col].astype(int)

    # Target-encode categoricals (modifies X in-place, returns encoders)
    encoders = _target_encode_fit(X, y, cat_features)

    # Fill numeric NaN
    for c in feature_cols:
        if c not in cat_features and c in X.columns and pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].fillna(-999)

    cfg = dict(params or {})

    model = xgb.XGBClassifier(
        n_estimators=cfg.get("n_estimators", cfg.get("iterations", 1000)),
        learning_rate=cfg.get("learning_rate", 0.05),
        max_depth=cfg.get("max_depth", cfg.get("depth", 6)),
        scale_pos_weight=cfg.get("scale_pos_weight", 1.0),
        eval_metric="aucpr",
        early_stopping_rounds=cfg.get("early_stopping_rounds", 50),
        verbosity=cfg.get("verbosity", 1),
        random_state=42,
        tree_method=cfg.get("tree_method", "hist"),
        reg_lambda=cfg.get("reg_lambda", 1.0),
        subsample=cfg.get("subsample", 0.8),
        colsample_bytree=cfg.get("colsample_bytree", 0.8),
        min_child_weight=cfg.get("min_child_weight", 5),
        gamma=cfg.get("gamma", 0.1),
    )

    fit_kwargs = {}
    if val_df is not None:
        X_val, y_val = _prepare_val(val_df, feature_cols, cat_features, target_col, has_label_col, encoders)
        fit_kwargs["eval_set"] = [(X_val, y_val)]
        fit_kwargs["verbose"] = cfg.get("verbose", 100)

    model.fit(X, y, **fit_kwargs)
    logger.info("xgboost_trained", n_rows=len(X), n_features=len(feature_cols))
    return model, encoders


def _prepare_val(
    val_df: pd.DataFrame,
    feature_cols: list[str],
    cat_features: list[str],
    target_col: str,
    has_label_col: str,
    encoders: dict,
) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare val set with same target encoding as train."""
    if has_label_col in val_df.columns:
        val_df = val_df.loc[val_df[has_label_col]].copy()
    val_df = val_df.dropna(subset=[target_col])
    X = val_df[feature_cols].copy()
    y = val_df[target_col].astype(int)
    X = _target_encode_transform(X, cat_features, encoders)
    for c in feature_cols:
        if c not in cat_features and c in X.columns and pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].fillna(-999)
    return X, y
