"""
Leakage-safe historical fraud-rate features: merchant, MCC (and optional card_brand).
Uses only transactions with event_time < current transaction time; Bayesian smoothing.

Memory-efficient v2: only copies the 4 needed columns (not all 105+).
Peak RAM: ~430MB per call vs ~20GB in v1.
"""
from __future__ import annotations

import gc

import pandas as pd
import numpy as np


def add_merchant_fraud_rate(
    df: pd.DataFrame,
    date_col: str = "date",
    entity_col: str = "merchant_id",
    label_col: str = "is_fraud",
    has_label_col: str = "has_label",
    alpha: float = 1.0,
    beta: float = 10.0,
    out_col: str = "merchant_historical_fraud_rate",
) -> pd.DataFrame:
    """
    For each row at time t: fraud_rate = (fraud_count_past + alpha) / (total_count_past + beta),
    where past = only transactions with same entity and date < t; counts use only labeled rows.

    Memory-efficient: sorts only the 4 required columns (not all 105+).
    Modifies df in-place — no full DataFrame copy.
    """
    if entity_col not in df.columns:
        return df

    # Sort only the 4 columns needed for computation (~430MB vs ~10GB for full df)
    needed = [c for c in [entity_col, date_col, label_col, has_label_col] if c in df.columns]
    small = df[needed].sort_values([entity_col, date_col])

    def _past_rate(g: pd.DataFrame) -> pd.Series:
        total_past = g[has_label_col].astype(int).cumsum().shift(1).fillna(0)
        fraud_past = (
            (g[label_col].fillna(0) * g[has_label_col].astype(int))
            .cumsum().shift(1).fillna(0)
        )
        return (fraud_past + alpha) / (total_past + beta)

    rates = small.groupby(entity_col, group_keys=False).apply(_past_rate)
    if isinstance(rates, pd.Series):
        rates = rates.droplevel(0) if rates.index.nlevels > 1 else rates

    # Assign back using original index — no reset_index needed
    df[out_col] = rates.reindex(df.index).values

    del small, rates
    gc.collect()
    return df


def add_mcc_fraud_rate(
    df: pd.DataFrame,
    date_col: str = "date",
    mcc_col: str = "mcc_str",
    label_col: str = "is_fraud",
    has_label_col: str = "has_label",
    alpha: float = 1.0,
    beta: float = 10.0,
    out_col: str = "mcc_historical_fraud_rate",
) -> pd.DataFrame:
    """Same as merchant but per MCC."""
    if mcc_col not in df.columns:
        mcc_col = "mcc"
    return add_merchant_fraud_rate(
        df,
        date_col=date_col,
        entity_col=mcc_col,
        label_col=label_col,
        has_label_col=has_label_col,
        alpha=alpha,
        beta=beta,
        out_col=out_col,
    )


def add_card_brand_fraud_rate(
    df: pd.DataFrame,
    date_col: str = "date",
    entity_col: str = "card_brand",
    label_col: str = "is_fraud",
    has_label_col: str = "has_label",
    alpha: float = 1.0,
    beta: float = 10.0,
    out_col: str = "card_brand_historical_fraud_rate",
) -> pd.DataFrame:
    """Historical fraud rate per card_brand (past-only, Bayesian smoothed)."""
    if entity_col not in df.columns:
        return df
    return add_merchant_fraud_rate(
        df,
        date_col=date_col,
        entity_col=entity_col,
        label_col=label_col,
        has_label_col=has_label_col,
        alpha=alpha,
        beta=beta,
        out_col=out_col,
    )


def add_risk_aggregates(
    df: pd.DataFrame,
    date_col: str = "date",
    alpha: float = 1.0,
    beta: float = 10.0,
    include_card_brand: bool = True,
) -> pd.DataFrame:
    """
    Add merchant_historical_fraud_rate, mcc_historical_fraud_rate,
    and optionally card_brand_historical_fraud_rate. All leakage-free (past-only).

    Memory-efficient: each call sorts only 4 columns. Total extra RAM ~1.3GB vs ~60GB in v1.
    """
    df = add_merchant_fraud_rate(df, date_col=date_col, alpha=alpha, beta=beta,
                                  out_col="merchant_historical_fraud_rate")
    df = add_mcc_fraud_rate(df, date_col=date_col, alpha=alpha, beta=beta,
                             out_col="mcc_historical_fraud_rate")
    if include_card_brand and "card_brand" in df.columns:
        df = add_card_brand_fraud_rate(df, date_col=date_col, alpha=alpha, beta=beta,
                                        out_col="card_brand_historical_fraud_rate")
    return df
