"""
Schema audit and data quality checks for EDA.
Industry-standard: dtype, null %, cardinality, range (numeric), sample (categorical).
"""
from __future__ import annotations

import pandas as pd


def schema_audit_df(df: pd.DataFrame, name: str = "df") -> pd.DataFrame:
    """
    Build a schema audit table: column, dtype, null_count, null_pct, cardinality, range/sample.
    """
    rows = []
    for col in df.columns:
        s = df[col]
        dtype = str(s.dtype)
        null_count = s.isna().sum()
        null_pct = round(100 * null_count / len(df), 2) if len(df) > 0 else 0
        card = s.nunique()

        if pd.api.types.is_numeric_dtype(s) and s.notna().any():
            range_str = f"[{s.min():.2f}, {s.max():.2f}]"
        elif pd.api.types.is_datetime64_any_dtype(s) and s.notna().any():
            range_str = f"[{s.min()}, {s.max()}]"
        else:
            sample = s.dropna().head(3).tolist()
            range_str = str(sample)[:80] + ("..." if len(str(sample)) > 80 else "")

        rows.append({
            "column": col,
            "dtype": dtype,
            "null_count": null_count,
            "null_pct": null_pct,
            "cardinality": card,
            "range_or_sample": range_str,
        })
    audit = pd.DataFrame(rows)
    audit.attrs["table_name"] = name
    audit.attrs["row_count"] = len(df)
    return audit


def data_quality_checks(
    df: pd.DataFrame,
    id_col: str | None = None,
    date_col: str | None = None,
    amount_col: str | None = None,
) -> dict[str, any]:
    """
    Run standard data quality checks. Returns dict of check name -> result.
    """
    results = {}
    if id_col and id_col in df.columns:
        n = len(df)
        unique = df[id_col].nunique()
        results["duplicate_ids"] = n - unique
        results["unique_id_ratio"] = round(unique / n, 4) if n > 0 else 0
    if date_col and date_col in df.columns:
        s = pd.to_datetime(df[date_col], errors="coerce")
        results["null_dates"] = s.isna().sum()
        results["future_dates"] = (s > pd.Timestamp.now()).sum() if s.notna().any() else 0
        if s.notna().any():
            results["date_min"] = s.min()
            results["date_max"] = s.max()
    if amount_col and amount_col in df.columns:
        a = pd.to_numeric(df[amount_col].astype(str).str.replace(r"[$,\s]", "", regex=True), errors="coerce")
        results["null_amounts"] = a.isna().sum()
        results["negative_amounts"] = (a < 0).sum() if a.notna().any() else 0
        results["zero_amounts"] = (a == 0).sum() if a.notna().any() else 0
    return results
