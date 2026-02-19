"""Tests for src/modeling/train.py -- feature column detection and evaluate_val."""
import pandas as pd
import numpy as np
import pytest

from src.modeling.train import get_feature_columns, EXCLUDE_FEATURES, evaluate_val


class TestGetFeatureColumns:
    def test_excludes_ids_and_target(self, tiny_transactions):
        feature_cols, cat_cols = get_feature_columns(tiny_transactions)
        for excl in ("id", "date", "is_fraud", "has_label"):
            assert excl not in feature_cols

    def test_numeric_included(self, tiny_transactions):
        feature_cols, _ = get_feature_columns(tiny_transactions)
        assert "amount" in feature_cols
        assert "credit_limit" in feature_cols

    def test_categorical_detected(self, tiny_transactions):
        _, cat_cols = get_feature_columns(tiny_transactions)
        assert "merchant_id" in cat_cols
        assert "mcc_str" in cat_cols

    def test_custom_exclude(self, tiny_transactions):
        feature_cols, _ = get_feature_columns(
            tiny_transactions, exclude={"id", "date", "is_fraud", "has_label", "amount"}
        )
        assert "amount" not in feature_cols


class TestEvaluateVal:
    def test_returns_metrics(self, tiny_transactions, mock_model):
        feature_cols = ["amount", "credit_limit", "has_chip"]
        cat_cols = []
        metrics = evaluate_val(
            mock_model, tiny_transactions, feature_cols, cat_cols
        )
        assert "pr_auc" in metrics
        assert "roc_auc" in metrics
        assert "brier" in metrics
        assert not np.isnan(metrics["pr_auc"])

    def test_empty_df(self, mock_model):
        empty = pd.DataFrame({
            "amount": pd.Series(dtype=float),
            "is_fraud": pd.Series(dtype=int),
            "has_label": pd.Series(dtype=bool),
        })
        metrics = evaluate_val(mock_model, empty, ["amount"], [])
        assert np.isnan(metrics["pr_auc"])
