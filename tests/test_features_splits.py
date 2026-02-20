"""Tests for src/features/splits.py -- time-based splitting and fraud rate stats."""
import pandas as pd
import pytest

from src.features.splits import time_based_splits, split_fraud_rates


@pytest.fixture
def split_df():
    """DataFrame spanning 2016-2021 for split testing."""
    dates = pd.to_datetime([
        "2016-06-01", "2017-06-01", "2017-12-31",  # train
        "2018-06-01", "2019-06-01",                 # val
        "2020-06-01", "2021-06-01",                 # test
    ])
    return pd.DataFrame({
        "date": dates,
        "amount": [100, 200, 300, 400, 500, 600, 700],
        "is_fraud": [0, 0, 1, 0, 1, 0, 0],
        "has_label": [True] * 7,
    })


class TestTimeBasedSplits:
    def test_split_sizes(self, split_df):
        train, val, test = time_based_splits(split_df)
        assert len(train) == 3
        assert len(val) == 2
        assert len(test) == 2

    def test_no_overlap(self, split_df):
        train, val, test = time_based_splits(split_df)
        train_max = pd.to_datetime(train["date"]).max()
        val_min = pd.to_datetime(val["date"]).min()
        test_min = pd.to_datetime(test["date"]).min()
        assert train_max <= pd.Timestamp("2017-12-31")
        assert val_min > pd.Timestamp("2017-12-31")
        assert test_min >= pd.Timestamp("2020-01-01")

    def test_empty_val(self):
        df = pd.DataFrame({
            "date": pd.to_datetime(["2015-01-01", "2022-01-01"]),
            "amount": [1, 2],
        })
        train, val, test = time_based_splits(df)
        assert len(val) == 0


class TestSplitFraudRates:
    def test_rates(self, split_df):
        train, val, test = time_based_splits(split_df)
        rates = split_fraud_rates(train, val, test)
        assert rates["train"]["rows"] == 3
        assert rates["train"]["fraud_count"] == 1
        assert rates["val"]["fraud_count"] == 1
        assert rates["test"]["fraud_count"] == 0
        assert 0 <= rates["train"]["fraud_rate"] <= 1
