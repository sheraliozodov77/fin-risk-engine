"""Tests for src/features/build.py -- parsing, time features, static prep."""
import pandas as pd
import numpy as np

from src.features.build import (
    parse_currency,
    parse_card_bools,
    add_time_features,
    prepare_static_transaction,
    prepare_cards,
    prepare_users,
    add_acct_age_days,
)


class TestParseCurrency:
    def test_basic_values(self):
        s = pd.Series(["$100.00", "$-50.50", " $1,234.56 "])
        out = parse_currency(s)
        assert out.iloc[0] == 100.0
        assert out.iloc[1] == -50.5
        assert out.iloc[2] == 1234.56

    def test_zero(self):
        out = parse_currency(pd.Series(["$0.00"]))
        assert out.iloc[0] == 0.0


class TestParseCardBools:
    def test_yes_no(self):
        df = pd.DataFrame({"has_chip": ["YES", "NO"], "card_on_dark_web": ["TRUE", "FALSE"]})
        out = parse_card_bools(df)
        assert out["has_chip"].tolist() == [1, 0]
        assert out["card_on_dark_web"].tolist() == [1, 0]

    def test_missing_columns(self):
        df = pd.DataFrame({"other_col": [1, 2]})
        out = parse_card_bools(df)
        assert "other_col" in out.columns


class TestAddTimeFeatures:
    def test_features_added(self, tiny_transactions):
        out = add_time_features(tiny_transactions)
        for col in ("hour_of_day", "day_of_week", "is_weekend", "month", "is_night"):
            assert col in out.columns

    def test_is_night(self):
        df = pd.DataFrame({"date": pd.to_datetime(["2020-01-01 03:00", "2020-01-01 12:00"])})
        out = add_time_features(df)
        assert out["is_night"].tolist() == [1, 0]

    def test_is_weekend(self):
        # 2020-01-04 is Saturday, 2020-01-06 is Monday
        df = pd.DataFrame({"date": pd.to_datetime(["2020-01-04", "2020-01-06"])})
        out = add_time_features(df)
        assert out["is_weekend"].tolist() == [1, 0]


class TestPrepareStaticTransaction:
    def test_parses_and_adds_time(self):
        # Use np.array with object dtype to match CSV-loaded data
        df = pd.DataFrame({
            "date": pd.to_datetime(["2020-01-01 14:30"]),
            "amount": np.array(["$100.50"], dtype=object),
            "card_id": [1],
        })
        out = prepare_static_transaction(df)
        assert pd.api.types.is_float_dtype(out["amount"])
        assert "hour_of_day" in out.columns

    def test_idempotent_on_numeric_amount(self, tiny_transactions):
        out = prepare_static_transaction(tiny_transactions)
        assert pd.api.types.is_float_dtype(out["amount"])


class TestPrepareCards:
    def test_credit_limit_parsing(self):
        df = pd.DataFrame({
            "credit_limit": np.array(["$5,000.00", "$10,000.00"], dtype=object),
            "has_chip": np.array(["YES", "NO"], dtype=object),
            "card_on_dark_web": np.array(["FALSE", "TRUE"], dtype=object),
            "acct_open_date": np.array(["01/2015", "06/2020"], dtype=object),
        })
        out = prepare_cards(df)
        assert out["credit_limit"].iloc[0] == 5000.0
        assert out["has_chip"].iloc[0] == 1
        assert pd.api.types.is_datetime64_any_dtype(out["acct_open_date"])

    def test_negative_credit_limit_replaced(self):
        df = pd.DataFrame({"credit_limit": [-100.0, 5000.0]})
        out = prepare_cards(df)
        assert pd.isna(out["credit_limit"].iloc[0])
        assert out["credit_limit"].iloc[1] == 5000.0


class TestPrepareUsers:
    def test_dollar_parsing(self):
        df = pd.DataFrame({
            "per_capita_income": np.array(["$50,000.00"], dtype=object),
            "yearly_income": np.array(["$80,000.00"], dtype=object),
            "total_debt": np.array(["$10,000.00"], dtype=object),
        })
        out = prepare_users(df)
        assert out["yearly_income"].iloc[0] == 80000.0

    def test_negative_replaced(self):
        df = pd.DataFrame({"yearly_income": [-1000.0, 50000.0]})
        out = prepare_users(df)
        assert pd.isna(out["yearly_income"].iloc[0])


class TestAddAcctAgeDays:
    def test_basic(self):
        df = pd.DataFrame({
            "date": pd.to_datetime(["2020-06-15"]),
            "acct_open_date": pd.to_datetime(["2020-01-01"]),
        })
        out = add_acct_age_days(df)
        assert out["acct_age_days"].iloc[0] == 166

    def test_missing_open_date(self):
        df = pd.DataFrame({
            "date": pd.to_datetime(["2020-06-15"]),
            "acct_open_date": [pd.NaT],
        })
        out = add_acct_age_days(df)
        assert pd.isna(out["acct_age_days"].iloc[0])
