"""Shared test fixtures: synthetic DataFrames, mock models."""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def tiny_transactions():
    """10-row synthetic transaction DataFrame mimicking gold table structure."""
    np.random.seed(42)
    n = 10
    dates = pd.date_range("2016-01-01", periods=n, freq="30D")
    return pd.DataFrame({
        "id": range(1, n + 1),
        "date": dates,
        "card_id": [1, 2, 1, 3, 2, 1, 3, 2, 1, 3],
        "client_id": [10, 20, 10, 30, 20, 10, 30, 20, 10, 30],
        "amount": np.random.uniform(5.0, 500.0, n).round(2),
        "merchant_id": np.random.choice(["M1", "M2", "M3"], n),
        "mcc_str": np.random.choice(["grocery", "gas", "online"], n),
        "card_brand": np.random.choice(["Visa", "MC"], n),
        "has_chip": np.random.choice([0, 1], n),
        "card_on_dark_web": np.zeros(n, dtype=int),
        "credit_limit": np.random.uniform(1000, 10000, n).round(2),
        "is_fraud": [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
        "has_label": [True] * n,
    })


@pytest.fixture
def tiny_transactions_dollar():
    """Transactions with dollar-string amounts (pre-parse)."""
    return pd.DataFrame({
        "id": [1, 2, 3],
        "date": pd.to_datetime(["2017-06-15 14:30", "2018-03-01 02:15", "2020-07-20 22:00"]),
        "amount": ["$100.50", "$-25.00", "$1,234.56"],
        "card_id": [1, 2, 3],
    })


class MockModel:
    """Minimal model mock with predict_proba for testing evaluate_val."""

    def __init__(self, fraud_prob=0.5):
        self.fraud_prob = fraud_prob

    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([
            np.full(n, 1 - self.fraud_prob),
            np.full(n, self.fraud_prob),
        ])


@pytest.fixture
def mock_model():
    return MockModel(fraud_prob=0.8)
