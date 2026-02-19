"""
Load raw datasets with minimal parsing.
Schema validation and cleaning are separate steps.
"""
from pathlib import Path
import json
import pandas as pd

from src.config import get_paths


def _resolve_path(key: str) -> Path:
    paths = get_paths()
    raw = paths.get("data", {}).get("raw", {})
    p = raw.get(key)
    if not p:
        raise FileNotFoundError(f"config/data/raw/{key} not set")
    path = Path(p)
    if not path.is_absolute():
        root = Path(__file__).resolve().parents[2]
        path = root / p
    return Path(path)


def load_transactions(
    path: str | Path | None = None,
    nrows: int | None = None,
    parse_dates: bool = True,
) -> pd.DataFrame:
    """Load transactions_data.csv. Optionally limit rows for EDA."""
    p = path or _resolve_path("transactions")
    df = pd.read_csv(p, nrows=nrows, low_memory=False)
    if parse_dates and "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def load_cards(path: str | Path | None = None) -> pd.DataFrame:
    """Load cards_data.csv."""
    p = path or _resolve_path("cards")
    return pd.read_csv(p, low_memory=False)


def load_users(path: str | Path | None = None) -> pd.DataFrame:
    """Load users_data.csv."""
    p = path or _resolve_path("users")
    return pd.read_csv(p, low_memory=False)


def load_mcc_codes(path: str | Path | None = None) -> dict[str, str]:
    """Load mcc_codes.json as {mcc_str: description}."""
    p = path or _resolve_path("mcc_codes")
    with open(p) as f:
        data = json.load(f)
    # Handle both flat dict and nested {"root": {...}}
    if "root" in data and isinstance(data["root"], dict):
        return data["root"]
    if isinstance(data, dict) and not any(k in data for k in ("root", "target")):
        return data
    return data


def load_fraud_labels(path: str | Path | None = None) -> dict[str, int]:
    """
    Load train_fraud_labels.json.
    Returns dict[transaction_id_str, 0|1] with Yes->1, No->0.
    """
    p = path or _resolve_path("fraud_labels")
    with open(p) as f:
        data = json.load(f)
    target = data.get("target", data)
    return {
        str(k): 1 if str(v).strip().lower() == "yes" else 0
        for k, v in target.items()
    }
