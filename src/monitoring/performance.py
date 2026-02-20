"""
Performance drift: PR-AUC by time period, alert precision over time.
Uses labeled val (or test) and model predictions grouped by period (e.g. month).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _prepare_X(
    df: pd.DataFrame,
    feature_cols: list[str],
    cat_cols: list[str],
) -> pd.DataFrame:
    X = df.reindex(columns=feature_cols).copy()
    for c in feature_cols:
        if c in cat_cols and c in X.columns:
            X[c] = X[c].fillna("__missing__").astype(str).replace("nan", "__missing__")
        elif c in X.columns and pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].fillna(-999)
    return X


def performance_by_period(
    val_df: pd.DataFrame,
    model: Any,
    feature_cols: list[str],
    cat_cols: list[str],
    time_col: str = "date",
    target_col: str = "is_fraud",
    has_label_col: str = "has_label",
    period: str = "M",
    t_high: float = 0.7,
    t_med: float = 0.3,
) -> pd.DataFrame:
    """
    Group labeled val by time period; for each period compute PR-AUC, alert count, alert precision.
    period: 'M' month, 'Q' quarter, 'W' week (pandas Grouper freq).
    Returns DataFrame with columns: period_start, n_rows, pr_auc, n_high, n_med, precision_high, precision_med.
    """
    from sklearn.metrics import average_precision_score

    df = val_df.loc[val_df[has_label_col]].copy() if has_label_col in val_df.columns else val_df.copy()
    df = df.dropna(subset=[target_col, time_col])
    if len(df) == 0:
        return pd.DataFrame()

    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col])
    df = df.sort_values(time_col)
    X = _prepare_X(df, feature_cols, cat_cols)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
    else:
        return pd.DataFrame()
    df = df.copy()
    df["_proba"] = proba
    df["_period"] = pd.Grouper(key=time_col, freq=period)

    rows = []
    for period_start, g in df.groupby("_period"):
        if len(g) < 10:
            continue
        y_g = g[target_col].astype(int).values
        p_g = g["_proba"].values
        pr_auc = average_precision_score(y_g, p_g)
        n_high = (p_g >= t_high).sum()
        n_med = ((p_g >= t_med) & (p_g < t_high)).sum()
        fraud_high = ((p_g >= t_high) & (y_g == 1)).sum()
        fraud_med = ((p_g >= t_med) & (p_g < t_high) & (y_g == 1)).sum()
        precision_high = fraud_high / n_high if n_high > 0 else np.nan
        precision_med = fraud_med / n_med if n_med > 0 else np.nan
        rows.append({
            "period_start": period_start,
            "n_rows": len(g),
            "pr_auc": pr_auc,
            "n_high": int(n_high),
            "n_med": int(n_med),
            "precision_high": precision_high,
            "precision_med": precision_med,
        })
    return pd.DataFrame(rows)
