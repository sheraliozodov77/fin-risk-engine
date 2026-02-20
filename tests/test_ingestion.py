"""Basic tests for ingestion and config."""
from pathlib import Path


def test_get_paths():
    from src.config import get_paths
    base = Path(__file__).resolve().parents[1]
    paths = get_paths(base_dir=base)
    assert "data" in paths
    assert "raw" in paths["data"]
    assert "transactions" in paths["data"]["raw"]


def test_parse_currency():
    from src.features.build import parse_currency
    import pandas as pd
    s = pd.Series(["$100.00", "$-50.50", " $1,234.56 "])
    out = parse_currency(s)
    assert out.iloc[0] == 100.0
    assert out.iloc[1] == -50.5
    assert out.iloc[2] == 1234.56
