"""
Build static and (placeholder) behavioral features.
Behavioral windows require time-ordered transactions and are implemented
in the full pipeline (see notebooks/02_feature_engineering.ipynb).
"""
import pandas as pd


def _is_string_like_dtype(series: pd.Series) -> bool:
    """Check if series has string-like dtype (object or StringDtype in pandas 3.0+)."""
    return pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)


def parse_currency(series: pd.Series) -> pd.Series:
    """Parse $ strings to float. Handles $-77.00, $14.57."""
    return series.astype(str).str.replace(r"[$,\s]", "", regex=True).astype(float)


def parse_card_bools(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize has_chip, card_on_dark_web to 0/1."""
    out = df.copy()
    for col in ("has_chip", "card_on_dark_web"):
        if col not in out.columns:
            continue
        s = out[col].astype(str).str.strip().str.upper()
        out[col] = (s.isin(("YES", "TRUE", "1", "Y"))).astype(int)
    return out


def add_time_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Add hour_of_day, day_of_week, is_weekend, month, is_night (00-05)."""
    out = df.copy()
    dt = pd.to_datetime(out[date_col], errors="coerce")
    out["hour_of_day"] = dt.dt.hour
    out["day_of_week"] = dt.dt.dayofweek
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    out["month"] = dt.dt.month
    out["is_night"] = ((dt.dt.hour >= 0) & (dt.dt.hour < 5)).astype(int)
    return out


def prepare_static_transaction(df: pd.DataFrame) -> pd.DataFrame:
    """Parse amount and add time features. Idempotent on already-parsed cols."""
    out = df.copy()
    if "amount" in out.columns and _is_string_like_dtype(out["amount"]):
        out["amount"] = parse_currency(out["amount"])
    return add_time_features(out, "date")


def prepare_cards(df: pd.DataFrame) -> pd.DataFrame:
    """Parse credit_limit, acct_open_date, booleans. Replace impossible numerics with NaN."""
    out = df.copy()
    if "credit_limit" in out.columns and _is_string_like_dtype(out["credit_limit"]):
        out["credit_limit"] = parse_currency(out["credit_limit"])
    if "credit_limit" in out.columns and pd.api.types.is_numeric_dtype(out["credit_limit"]):
        out.loc[out["credit_limit"] < 0, "credit_limit"] = pd.NA
    if "acct_open_date" in out.columns:
        out["acct_open_date"] = pd.to_datetime(out["acct_open_date"], format="%m/%Y", errors="coerce")
    return parse_card_bools(out)


def prepare_users(df: pd.DataFrame) -> pd.DataFrame:
    """Parse income/debt $ columns. Replace impossible (negative) values with NaN."""
    out = df.copy()
    for col in ("per_capita_income", "yearly_income", "total_debt"):
        if col in out.columns and _is_string_like_dtype(out[col]):
            out[col] = parse_currency(out[col])
        if col in out.columns and pd.api.types.is_numeric_dtype(out[col]):
            out.loc[out[col] < 0, col] = pd.NA
    return out


def add_acct_age_days(
    df: pd.DataFrame,
    date_col: str = "date",
    open_col: str = "acct_open_date",
    out_col: str = "acct_age_days",
) -> pd.DataFrame:
    """Derive acct_age_days = (date - acct_open_date).days. Non-negative; NaN if open date missing."""
    out = df.copy()
    if open_col not in out.columns or date_col not in out.columns:
        return out
    d = pd.to_datetime(out[date_col], errors="coerce")
    o = pd.to_datetime(out[open_col], errors="coerce")
    delta = (d - o).dt.days
    out[out_col] = delta.clip(lower=0).where(o.notna(), pd.NA)
    return out
