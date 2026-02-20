"""Basic tests for ingestion and config."""
from pathlib import Path

import pytest

_RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
_requires_raw = pytest.mark.skipif(
    not _RAW_DIR.exists(),
    reason="data/raw/ not present (large files not committed to repo)",
)


def test_get_paths():
    from src.config import get_paths
    base = Path(__file__).resolve().parents[1]
    paths = get_paths(base_dir=base)
    assert "data" in paths
    assert "raw" in paths["data"]
    assert "transactions" in paths["data"]["raw"]


@_requires_raw
def test_load_cards():
    from src.ingestion import load_cards
    df = load_cards()
    assert "id" in df.columns
    assert "client_id" in df.columns
    assert len(df) > 0


@_requires_raw
def test_load_users():
    from src.ingestion import load_users
    df = load_users()
    assert "id" in df.columns
    assert len(df) > 0


@_requires_raw
def test_load_mcc():
    from src.ingestion import load_mcc_codes
    mcc = load_mcc_codes()
    assert isinstance(mcc, dict)
    assert len(mcc) > 0


@_requires_raw
def test_load_transactions_small():
    from src.ingestion import load_transactions
    df = load_transactions(nrows=100)
    assert "id" in df.columns
    assert "date" in df.columns
    assert "card_id" in df.columns
    assert len(df) <= 100


def test_parse_currency():
    from src.features.build import parse_currency
    import pandas as pd
    s = pd.Series(["$100.00", "$-50.50", " $1,234.56 "])
    out = parse_currency(s)
    assert out.iloc[0] == 100.0
    assert out.iloc[1] == -50.5
    assert out.iloc[2] == 1234.56
