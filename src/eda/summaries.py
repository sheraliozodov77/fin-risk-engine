"""
Numeric and categorical summary statistics for EDA.
"""
from __future__ import annotations

import pandas as pd


def numeric_summary(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Summary stats for numeric columns: count, mean, std, min, p25, p50, p75, max."""
    if columns is None:
        columns = df.select_dtypes(include=["number"]).columns.tolist()
    if not columns:
        return pd.DataFrame()
    return df[columns].describe(percentiles=[0.25, 0.5, 0.75]).T


def categorical_summary(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    top_n: int = 15,
) -> dict[str, pd.DataFrame]:
    """Value counts and proportions for object/category columns. Returns dict col -> value_counts df."""
    if columns is None:
        columns = df.select_dtypes(include=["object", "category"]).columns.tolist()
    out = {}
    for col in columns:
        vc = df[col].value_counts(dropna=False).head(top_n)
        prop = (vc / len(df) * 100).round(2)
        out[col] = pd.DataFrame({"count": vc, "pct": prop})
    return out
