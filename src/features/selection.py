"""
Feature selection: correlation filtering, VIF, and permutation importance.
All methods operate on numeric features only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.logging_config import get_logger

logger = get_logger(__name__)


def correlation_filter(
    df: pd.DataFrame,
    features: list[str],
    threshold: float = 0.95,
) -> list[str]:
    """
    Drop one of each pair of features with |correlation| > threshold.
    Keeps the first feature in each correlated pair (stable ordering).
    Returns surviving feature names.
    """
    numeric_feats = [f for f in features if f in df.columns and pd.api.types.is_numeric_dtype(df[f])]
    if len(numeric_feats) < 2:
        return features

    corr = df[numeric_feats].corr().abs()
    to_drop = set()
    for i in range(len(numeric_feats)):
        if numeric_feats[i] in to_drop:
            continue
        for j in range(i + 1, len(numeric_feats)):
            if numeric_feats[j] in to_drop:
                continue
            if corr.iloc[i, j] > threshold:
                to_drop.add(numeric_feats[j])
                logger.info(
                    "correlation_drop",
                    dropped=numeric_feats[j],
                    correlated_with=numeric_feats[i],
                    corr=round(float(corr.iloc[i, j]), 4),
                )

    surviving = [f for f in features if f not in to_drop]
    logger.info("correlation_filter_done", original=len(features), surviving=len(surviving), dropped=len(to_drop))
    return surviving


def vif_filter(
    df: pd.DataFrame,
    features: list[str],
    max_vif: float = 10.0,
) -> list[str]:
    """
    Iteratively remove the feature with highest VIF until all VIF <= max_vif.
    Only considers numeric features. Returns surviving feature names.
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    numeric_feats = [f for f in features if f in df.columns and pd.api.types.is_numeric_dtype(df[f])]
    if len(numeric_feats) < 2:
        return features

    # Work on a clean copy (drop NaN rows for VIF computation)
    X = df[numeric_feats].dropna()
    if len(X) < 10:
        logger.warning("vif_filter_skipped", reason="too_few_rows", n_rows=len(X))
        return features

    remaining = list(numeric_feats)
    while len(remaining) > 1:
        X_curr = X[remaining].values.astype(float)
        vifs = []
        for i in range(len(remaining)):
            try:
                vifs.append(variance_inflation_factor(X_curr, i))
            except Exception:
                vifs.append(0.0)
        max_idx = int(np.argmax(vifs))
        if vifs[max_idx] <= max_vif:
            break
        dropped = remaining.pop(max_idx)
        logger.info("vif_drop", dropped=dropped, vif=round(vifs[max_idx], 2))

    # Rebuild full feature list preserving non-numeric features
    dropped_set = set(numeric_feats) - set(remaining)
    surviving = [f for f in features if f not in dropped_set]
    logger.info("vif_filter_done", original=len(features), surviving=len(surviving), dropped=len(dropped_set))
    return surviving


def permutation_importance_filter(
    model,
    X: pd.DataFrame,
    y: np.ndarray | pd.Series,
    features: list[str],
    min_importance: float = 0.0001,
    n_repeats: int = 5,
    scoring: str = "average_precision",
) -> list[str]:
    """
    Drop features whose permutation importance is below min_importance.
    Uses sklearn permutation_importance. Returns surviving feature names.
    """
    from sklearn.inspection import permutation_importance

    X_eval = X.reindex(columns=features).fillna(-999)
    result = permutation_importance(
        model, X_eval, y,
        n_repeats=n_repeats,
        scoring=scoring,
        random_state=42,
        n_jobs=-1,
    )
    importances = result.importances_mean
    to_drop = set()
    for i, feat in enumerate(features):
        imp = importances[i]
        if imp < min_importance:
            to_drop.add(feat)
            logger.info("permutation_drop", feature=feat, importance=round(float(imp), 6))

    surviving = [f for f in features if f not in to_drop]
    logger.info(
        "permutation_filter_done",
        original=len(features),
        surviving=len(surviving),
        dropped=len(to_drop),
    )
    return surviving


def run_all_filters(
    df: pd.DataFrame,
    features: list[str],
    corr_threshold: float = 0.95,
    max_vif: float = 10.0,
    model=None,
    y=None,
    min_importance: float = 0.0001,
) -> list[str]:
    """Run correlation -> VIF -> permutation importance filters in sequence."""
    logger.info("feature_selection_start", n_features=len(features))

    surviving = correlation_filter(df, features, threshold=corr_threshold)

    try:
        surviving = vif_filter(df, surviving, max_vif=max_vif)
    except ImportError:
        logger.warning("vif_skipped", reason="statsmodels_not_installed")

    if model is not None and y is not None:
        surviving = permutation_importance_filter(
            model, df, y, surviving, min_importance=min_importance,
        )

    logger.info("feature_selection_done", original=len(features), final=len(surviving))
    return surviving
