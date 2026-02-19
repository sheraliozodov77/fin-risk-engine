"""
Probability calibration (Platt / isotonic) for fraud risk scores.
Fit on val (proba, y_true); apply to new proba; report Brier before/after.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def fit_calibrator(
    proba: np.ndarray,
    y_true: np.ndarray,
    method: str = "isotonic",
) -> Any:
    """
    Fit a calibrator on (proba, y_true). proba = raw model probabilities (fraud class).
    method: 'isotonic' (default) or 'platt'.
    Returns a fitted calibrator with .predict(proba) or .transform(proba).
    """
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression

    proba = np.asarray(proba).ravel()
    y_true = np.asarray(y_true).ravel()
    if method == "isotonic":
        cal = IsotonicRegression(out_of_bounds="clip")
        cal.fit(proba, y_true)
        return cal
    if method == "platt":
        cal = LogisticRegression(C=1.0, max_iter=1000)
        cal.fit(proba.reshape(-1, 1), y_true)
        return cal
    raise ValueError("method must be 'isotonic' or 'platt'")


def apply_calibrator(proba: np.ndarray, calibrator: Any) -> np.ndarray:
    """Return calibrated probabilities (same shape as proba)."""
    proba = np.asarray(proba).ravel()
    if hasattr(calibrator, "predict"):
        out = calibrator.predict(proba)
    else:
        out = calibrator.transform(proba.reshape(-1, 1)).ravel()
    return np.clip(out, 1e-6, 1 - 1e-6)


def brier_score(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Brier score (lower = better calibrated)."""
    from sklearn.metrics import brier_score_loss
    return float(brier_score_loss(y_true, proba))
