"""
Time-based train/validation/test splits. No random shuffle.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def get_split_boundaries(config: dict | None = None) -> tuple[str, str, str]:
    """
    Return (train_end, val_end, test_start) from config.
    Train: date <= train_end; Val: train_end < date <= val_end; Test: date >= test_start.
    """
    if config is None:
        try:
            from src.config import get_model_config
            config = get_model_config()
        except Exception:
            config = {}  # fallback to defaults if config unavailable
    splits = config.get("splits", {})
    time_col = splits.get("time_column", "date")
    train_end = splits.get("train_end", "2017-12-31")
    val_end = splits.get("val_end", "2019-12-31")
    test_start = splits.get("test_start", "2020-01-01")
    return train_end, val_end, test_start


def time_based_splits(
    df: pd.DataFrame,
    date_col: str = "date",
    train_end: str = "2017-12-31",
    val_end: str = "2019-12-31",
    test_start: str = "2020-01-01",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split df by date. Returns (train, val, test).
    Train: date <= train_end; Val: train_end < date <= val_end; Test: date >= test_start.
    """
    d = pd.to_datetime(df[date_col], errors="coerce")
    te = pd.Timestamp(train_end)
    ve = pd.Timestamp(val_end)
    ts = pd.Timestamp(test_start)

    train = df[d <= te]
    val = df[(d > te) & (d <= ve)]
    test = df[d >= ts]
    return train, val, test


def split_fraud_rates(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    label_col: str = "is_fraud",
    has_label_col: str = "has_label",
) -> dict[str, dict]:
    """
    Return fraud rate and row counts per split (labeled rows only for rate).
    """
    def _stats(_df: pd.DataFrame) -> dict:
        labeled = _df[_df[has_label_col]] if has_label_col in _df.columns else _df
        n = len(labeled)
        n_fraud = labeled[label_col].sum() if n else 0
        rate = n_fraud / n if n else 0.0
        return {"rows": len(_df), "labeled_rows": n, "fraud_count": int(n_fraud), "fraud_rate": float(rate)}

    return {
        "train": _stats(train),
        "val": _stats(val),
        "test": _stats(test),
    }


def save_splits(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    out_dir: str | Path,
    *,
    train_name: str = "train.parquet",
    val_name: str = "val.parquet",
    test_name: str = "test.parquet",
) -> tuple[Path, Path, Path]:
    """Write train/val/test to parquet; create out_dir if needed. Returns paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p_train = out / train_name
    p_val = out / val_name
    p_test = out / test_name
    train.to_parquet(p_train, index=False)
    val.to_parquet(p_val, index=False)
    test.to_parquet(p_test, index=False)
    return p_train, p_val, p_test
