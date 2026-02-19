"""
Explainability: global feature importance (SHAP) and local top-k reason codes.
Supports CatBoost (built-in ShapValues via Pool) and XGBoost (shap.TreeExplainer).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.logging_config import get_logger

logger = get_logger(__name__)


def _to_pool(X: pd.DataFrame, feature_cols: list[str], cat_cols: list[str] | None = None):
    """Build CatBoost Pool from X (same column order as feature_cols)."""
    try:
        from catboost import Pool
    except ImportError:
        return None
    X_ord = X.reindex(columns=feature_cols).fillna(-999)
    for c in (cat_cols or []):
        if c in X_ord.columns:
            X_ord[c] = X_ord[c].fillna("__missing__").astype(str).replace("nan", "__missing__")
    cat_idx = [i for i, c in enumerate(feature_cols) if c in (cat_cols or [])]
    return Pool(data=X_ord, cat_features=cat_idx if cat_idx else None)


def _get_shap_matrix(
    model: Any,
    X: pd.DataFrame,
    feature_cols: list[str],
    cat_cols: list[str] | None = None,
) -> np.ndarray | None:
    """
    Compute SHAP values matrix (n_rows, n_features).
    Supports CatBoost (built-in) and XGBoost/sklearn (shap.TreeExplainer).
    """
    # CatBoost: use built-in ShapValues
    if hasattr(model, "get_feature_importance"):
        pool = _to_pool(X, feature_cols, cat_cols)
        if pool is None:
            return None
        try:
            shap_matrix = model.get_feature_importance(type="ShapValues", data=pool)
            shap_matrix = np.asarray(shap_matrix)
            if shap_matrix.ndim == 1:
                shap_matrix = shap_matrix.reshape(1, -1)
            # CatBoost returns (n_rows, n_features+1) where last col is base value
            n_f = min(len(feature_cols), shap_matrix.shape[1] - 1)
            return shap_matrix[:, :n_f]
        except Exception as e:
            logger.warning("catboost_shap_failed", error=str(e))
            return None

    # XGBoost / sklearn tree models: use shap library
    try:
        import shap
        X_ord = X.reindex(columns=feature_cols).copy()
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_ord)
        return np.asarray(shap_values)
    except Exception as e:
        logger.warning("shap_tree_explainer_failed", error=str(e))
        return None


def get_global_importance(
    model: Any,
    X: pd.DataFrame,
    feature_cols: list[str],
    cat_cols: list[str] | None = None,
    top_n: int = 20,
) -> list[tuple[str, float]]:
    """
    Return list of (feature_name, mean_abs_shap) sorted by importance (desc).
    Works with CatBoost, XGBoost, and any sklearn tree model.
    """
    shap_matrix = _get_shap_matrix(model, X, feature_cols, cat_cols)
    if shap_matrix is None:
        return []

    n_f = min(len(feature_cols), shap_matrix.shape[1])
    mean_abs = np.abs(shap_matrix[:, :n_f]).mean(axis=0)
    order = np.argsort(-mean_abs)[:top_n]
    return [(feature_cols[i], float(mean_abs[i])) for i in order]


def get_local_reason_codes(
    model: Any,
    X: pd.DataFrame,
    row_idx: int,
    feature_cols: list[str],
    cat_cols: list[str] | None = None,
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """
    Return top-k (feature_name, shap_value) for one row (reason codes for alert).
    Positive shap = pushes toward fraud; negative = pushes toward non-fraud.
    """
    row = X.iloc[[row_idx]]
    codes_batch = get_local_reason_codes_batch(model, row, feature_cols, cat_cols=cat_cols, top_k=top_k, max_rows=1)
    return codes_batch.get(0, [])


def get_local_reason_codes_batch(
    model: Any,
    X: pd.DataFrame,
    feature_cols: list[str],
    cat_cols: list[str] | None = None,
    top_k: int = 5,
    max_rows: int = 1000,
) -> dict[int, list[tuple[str, float]]]:
    """
    Return for each row index (up to max_rows) the top-k reason codes.
    Works with CatBoost, XGBoost, and any sklearn tree model.
    """
    X_sub = X.head(max_rows).copy()
    shap_matrix = _get_shap_matrix(model, X_sub, feature_cols, cat_cols)
    if shap_matrix is None:
        return {}

    n_f = min(len(feature_cols), shap_matrix.shape[1])
    out = {}
    for i in range(shap_matrix.shape[0]):
        row_shap = shap_matrix[i, :n_f]
        order = np.argsort(-np.abs(row_shap))[:top_k]
        out[i] = [(feature_cols[j], float(row_shap[j])) for j in order]
    return out
