"""Tests for src/monitoring/drift.py -- PSI, missing rates, new categories."""
import pandas as pd
import numpy as np
import pytest

from src.monitoring.drift import (
    psi_numeric,
    psi_categorical,
    missing_rate,
    new_categories_share,
)


class TestPsiNumeric:
    def test_identical_distributions(self):
        ref = pd.Series(np.random.normal(0, 1, 1000))
        psi = psi_numeric(ref, ref)
        assert psi < 0.01  # identical -> near zero

    def test_shifted_distribution(self):
        ref = pd.Series(np.random.normal(0, 1, 1000))
        cur = pd.Series(np.random.normal(3, 1, 1000))
        psi = psi_numeric(ref, cur)
        assert psi > 0.1  # clearly shifted

    def test_empty_returns_nan(self):
        ref = pd.Series(dtype=float)
        cur = pd.Series([1.0, 2.0])
        assert np.isnan(psi_numeric(ref, cur))


class TestPsiCategorical:
    def test_identical(self):
        ref = pd.Series(["A", "B", "C"] * 100)
        psi = psi_categorical(ref, ref)
        assert psi < 0.01

    def test_different_distribution(self):
        ref = pd.Series(["A"] * 100 + ["B"] * 100)
        cur = pd.Series(["A"] * 10 + ["B"] * 190)
        psi = psi_categorical(ref, cur)
        assert psi > 0.05


class TestMissingRate:
    def test_no_missing(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        rates = missing_rate(df)
        assert rates["a"] == 0.0

    def test_some_missing(self):
        df = pd.DataFrame({"a": [1, np.nan, 3], "b": [np.nan, np.nan, 6]})
        rates = missing_rate(df)
        assert abs(rates["a"] - 1 / 3) < 1e-6
        assert abs(rates["b"] - 2 / 3) < 1e-6


class TestNewCategoriesShare:
    def test_no_new(self):
        ref = pd.Series(["A", "B", "C"])
        cur = pd.Series(["A", "B"])
        share, count = new_categories_share(ref, cur)
        assert share == 0.0
        assert count == 0

    def test_all_new(self):
        ref = pd.Series(["A", "B"])
        cur = pd.Series(["X", "Y", "Z"])
        share, count = new_categories_share(ref, cur)
        assert share == 1.0
        assert count == 3

    def test_partial_new(self):
        ref = pd.Series(["A", "B"])
        cur = pd.Series(["A", "C"])
        share, count = new_categories_share(ref, cur)
        assert share == 0.5
        assert count == 1
