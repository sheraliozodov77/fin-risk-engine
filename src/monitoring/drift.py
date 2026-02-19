"""
Data drift monitoring: PSI, missing rate, new categorical values (e.g. new merchants).
Reference = baseline (e.g. train); current = recent window (e.g. val or live).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def psi_numeric(
    reference: pd.Series,
    current: pd.Series,
    n_bins: int = 10,
) -> float:
    """
    Population Stability Index for a numeric series.
    Bins by reference quantiles; computes PSI = sum((curr_pct - ref_pct) * ln(curr_pct / ref_pct)).
    """
    ref = reference.dropna()
    cur = current.dropna()
    if len(ref) == 0 or len(cur) == 0:
        return np.nan
    try:
        edges = np.percentile(ref, np.linspace(0, 100, n_bins + 1))
        edges = np.unique(edges)
        if len(edges) < 2:
            return 0.0
        ref_bins = np.clip(np.searchsorted(edges, ref, side="right") - 1, 0, len(edges) - 2)
        cur_bins = np.clip(np.searchsorted(edges, cur, side="right") - 1, 0, len(edges) - 2)
        ref_counts = np.bincount(ref_bins, minlength=len(edges) - 1)
        cur_counts = np.bincount(cur_bins, minlength=len(edges) - 1)
        n_bins = len(ref_counts)
        ref_pct = (ref_counts + 1e-6) / (ref_counts.sum() + 1e-6 * n_bins)
        cur_pct = (cur_counts + 1e-6) / (cur_counts.sum() + 1e-6 * n_bins)
        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
        return float(psi)
    except Exception:
        return np.nan  # degrade gracefully on malformed data


def psi_categorical(reference: pd.Series, current: pd.Series) -> float:
    """
    PSI for categorical: treat each category as a bin; same formula.
    """
    ref = reference.fillna("__missing__").astype(str).replace("nan", "__missing__")
    cur = current.fillna("__missing__").astype(str).replace("nan", "__missing__")
    all_cats = pd.Index(ref.unique()).union(pd.Index(cur.unique()))
    ref_counts = ref.value_counts().reindex(all_cats, fill_value=0)
    cur_counts = cur.value_counts().reindex(all_cats, fill_value=0)
    ref_pct = (ref_counts + 1e-6) / (ref_counts.sum() + 1e-6 * len(all_cats))
    cur_pct = (cur_counts + 1e-6) / (cur_counts.sum() + 1e-6 * len(all_cats))
    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def missing_rate(df: pd.DataFrame, columns: list[str] | None = None) -> dict[str, float]:
    """Fraction missing per column. If columns is None, use all columns."""
    cols = columns or df.columns.tolist()
    cols = [c for c in cols if c in df.columns]
    return {c: float(df[c].isna().mean()) for c in cols}


def new_categories_share(
    reference: pd.Series,
    current: pd.Series,
) -> tuple[float, int]:
    """
    Share of current values that are new (not seen in reference).
    Returns (share_new, count_new).
    """
    ref_set = set(reference.dropna().astype(str).unique())
    cur = current.dropna().astype(str)
    n_cur = len(cur)
    if n_cur == 0:
        return 0.0, 0
    new_count = cur.isin(ref_set).eq(False).sum()
    return float(new_count / n_cur), int(new_count)


def compute_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    numeric_cols: list[str] | None = None,
    categorical_cols: list[str] | None = None,
    missing_cols: list[str] | None = None,
    new_category_cols: list[str] | None = None,
) -> dict:
    """
    Compute drift metrics between reference and current.
    Returns dict: psi (per column), missing_rate (current), new_share (per categorical).
    """
    numeric_cols = numeric_cols or ["amount"]
    categorical_cols = categorical_cols or ["card_brand", "mcc_str"]
    missing_cols = missing_cols or list(set(numeric_cols + categorical_cols + ["merchant_id"]))
    new_category_cols = new_category_cols or ["merchant_id", "mcc_str"]

    out = {
        "psi": {},
        "missing_rate_current": {},
        "new_share": {},
    }
    for c in numeric_cols:
        if c in reference_df.columns and c in current_df.columns:
            out["psi"][c] = psi_numeric(reference_df[c], current_df[c])
    for c in categorical_cols:
        if c in reference_df.columns and c in current_df.columns:
            out["psi"][c] = psi_categorical(reference_df[c], current_df[c])
    for c in missing_cols:
        if c in current_df.columns:
            out["missing_rate_current"][c] = float(current_df[c].isna().mean())
    for c in new_category_cols:
        if c in reference_df.columns and c in current_df.columns:
            share, count = new_categories_share(reference_df[c], current_df[c])
            out["new_share"][c] = {"share": share, "count_new": count}
    return out
