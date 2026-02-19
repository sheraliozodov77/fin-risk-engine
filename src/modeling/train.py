"""
Train CatBoost fraud classifier.
Uses time-based splits; expects train/val with is_fraud and has_label.
"""
from pathlib import Path
from typing import Any

import pandas as pd

# Columns to exclude from features (IDs, target, PII, metadata)
EXCLUDE_FEATURES = {
    "id",
    "date",
    "transaction_id_str",
    "has_label",
    "is_fraud",
    "card_number",
    "address",
    "mcc_description",  # long text, use mcc_str only
}


def _fill_cat(series: pd.Series) -> pd.Series:
    """Robust NaN→__missing__ for categorical columns (handles pandas 3.0 StringDtype)."""
    return series.fillna("__missing__").astype(str).replace("nan", "__missing__")


def _is_categorical_column(series: pd.Series) -> bool:
    """True if column should be treated as categorical for CatBoost (string, object, category, bool)."""
    dtype = series.dtype
    if pd.api.types.is_categorical_dtype(dtype):
        return True
    if pd.api.types.is_bool_dtype(dtype):
        return True
    if pd.api.types.is_string_dtype(dtype):
        return True
    if dtype.name == "object":
        return True
    if dtype.kind == "O":
        return True
    return False


def get_feature_columns(
    df: pd.DataFrame,
    target_col: str = "is_fraud",
    exclude: set[str] | None = None,
) -> tuple[list[str], list[str]]:
    """
    Return (feature_cols, cat_feature_cols) for modeling.
    Excludes IDs, target, PII. Categorical = string/object/category/bool (robust to parquet dtypes).
    """
    exclude = exclude or set(EXCLUDE_FEATURES)
    exclude = exclude | {target_col}
    candidates = [c for c in df.columns if c not in exclude and c in df.columns]
    feature_cols = []
    cat_cols = []
    for c in candidates:
        try:
            s = df[c]
        except Exception:
            continue
        if _is_categorical_column(s):
            feature_cols.append(c)
            cat_cols.append(c)
        elif pd.api.types.is_numeric_dtype(s):
            feature_cols.append(c)
    return feature_cols, cat_cols


def _prepare_X(
    df: pd.DataFrame,
    feature_cols: list[str],
    cat_features: list[str],
) -> pd.DataFrame:
    """Prepare feature matrix: fill NaN for categorical and numeric columns."""
    X = df[feature_cols].copy()
    for c in feature_cols:
        if c in cat_features and c in X.columns:
            X[c] = _fill_cat(X[c])
        elif c in X.columns and pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].fillna(-999)
    return X


def train_catboost(
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
    Train CatBoost binary classifier on labeled rows only.
    Pass val_df for early stopping (critical to prevent overfitting).
    Returns fitted model (or dict with feature_cols if CatBoost not installed).
    """
    if use_labeled_only and has_label_col in train_df.columns:
        train_df = train_df.loc[train_df[has_label_col]].copy()
    if feature_cols is None or cat_features is None:
        feature_cols, cat_features = get_feature_columns(train_df, target_col=target_col)
    # Drop rows with missing target
    train_df = train_df.dropna(subset=[target_col])
    X = _prepare_X(train_df, feature_cols, cat_features)
    y = train_df[target_col].astype(int)

    try:
        import catboost as cb
    except ImportError:
        return {
            "feature_cols": feature_cols,
            "cat_features": cat_features,
            "placeholder": True,
            "message": "pip install catboost to train",
        }

    cfg = dict(params or {})

    # scale_pos_weight: default 1.0 (no reweighting).
    # PR-AUC eval_metric already handles class imbalance correctly.
    # Use Optuna to search SPW in [1, 50] range if needed.
    model = cb.CatBoostClassifier(
        iterations=cfg.get("iterations", 1000),
        learning_rate=cfg.get("learning_rate", 0.05),
        depth=cfg.get("depth", 6),
        l2_leaf_reg=cfg.get("l2_leaf_reg", 3.0),
        border_count=cfg.get("border_count", 254),
        loss_function=cfg.get("loss_function", "Logloss"),
        eval_metric=cfg.get("eval_metric", "PRAUC"),
        early_stopping_rounds=cfg.get("early_stopping_rounds", 50),
        verbose=cfg.get("verbose", 100),
        scale_pos_weight=cfg.get("scale_pos_weight", 1.0),
        random_seed=42,
        train_dir="outputs/catboost_info",
    )

    train_pool = cb.Pool(X, y, cat_features=cat_features)

    fit_kwargs: dict[str, Any] = {}
    if val_df is not None:
        vdf = val_df.copy()
        if has_label_col in vdf.columns:
            vdf = vdf.loc[vdf[has_label_col]].copy()
        vdf = vdf.dropna(subset=[target_col])
        X_val = _prepare_X(vdf, feature_cols, cat_features)
        y_val = vdf[target_col].astype(int)
        eval_pool = cb.Pool(X_val, y_val, cat_features=cat_features)
        fit_kwargs["eval_set"] = eval_pool

    model.fit(train_pool, **fit_kwargs)
    return model


def evaluate_val(
    model: Any,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    cat_features: list[str],
    target_col: str = "is_fraud",
    has_label_col: str = "has_label",
) -> dict[str, float]:
    """Compute PR-AUC, ROC-AUC, Brier on labeled val rows."""
    if has_label_col in val_df.columns:
        val_df = val_df.loc[val_df[has_label_col]].copy()
    val_df = val_df.dropna(subset=[target_col])
    if len(val_df) == 0:
        return {
            "pr_auc": float("nan"),
            "roc_auc": float("nan"),
            "brier": float("nan"),
            "recall_at_precision_90": float("nan"),
        }
    X = _prepare_X(val_df, feature_cols, cat_features)
    y = val_df[target_col].astype(int)

    try:
        import numpy as np
        from sklearn.metrics import (
            average_precision_score,
            brier_score_loss,
            precision_recall_curve,
            roc_auc_score,
        )
    except ImportError:
        return {
            "pr_auc": float("nan"),
            "roc_auc": float("nan"),
            "brier": float("nan"),
            "recall_at_precision_90": float("nan"),
        }

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
    else:
        return {
            "pr_auc": float("nan"),
            "roc_auc": float("nan"),
            "brier": float("nan"),
            "recall_at_precision_90": float("nan"),
        }

    pr_auc = average_precision_score(y, proba)
    roc_auc = roc_auc_score(y, proba)
    brier = brier_score_loss(y, proba)
    precision, recall, _ = precision_recall_curve(y, proba)
    # Recall at 90% precision: max recall where precision >= 0.9
    mask = precision >= 0.9
    recall_at_precision_90 = float(recall[mask].max()) if mask.any() else float("nan")

    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "brier": brier,
        "recall_at_precision_90": recall_at_precision_90,
    }
