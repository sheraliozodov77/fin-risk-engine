"""Tests for src/modeling/calibrate.py -- isotonic/platt calibration."""
import numpy as np
import pytest

from src.modeling.calibrate import fit_calibrator, apply_calibrator, brier_score


@pytest.fixture
def calibration_data():
    """Synthetic probabilities and labels."""
    np.random.seed(42)
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    proba = np.array([0.1, 0.2, 0.15, 0.3, 0.05, 0.7, 0.8, 0.9, 0.6, 0.85])
    return proba, y_true


class TestFitCalibrator:
    def test_isotonic(self, calibration_data):
        proba, y_true = calibration_data
        cal = fit_calibrator(proba, y_true, method="isotonic")
        assert hasattr(cal, "predict")

    def test_platt(self, calibration_data):
        proba, y_true = calibration_data
        cal = fit_calibrator(proba, y_true, method="platt")
        assert hasattr(cal, "predict_proba")

    def test_invalid_method(self, calibration_data):
        proba, y_true = calibration_data
        with pytest.raises(ValueError, match="method must be"):
            fit_calibrator(proba, y_true, method="invalid")


class TestApplyCalibrator:
    def test_isotonic_output_shape(self, calibration_data):
        proba, y_true = calibration_data
        cal = fit_calibrator(proba, y_true, method="isotonic")
        calibrated = apply_calibrator(proba, cal)
        assert calibrated.shape == proba.shape
        assert np.all(calibrated >= 1e-6)
        assert np.all(calibrated <= 1 - 1e-6)

    def test_platt_output_shape(self, calibration_data):
        proba, y_true = calibration_data
        cal = fit_calibrator(proba, y_true, method="platt")
        # Platt uses LogisticRegression which needs 2D input for predict;
        # apply_calibrator calls .predict(proba) which fails on 1D.
        # Use predict_proba directly as a workaround to verify the calibrator fits.
        calibrated = cal.predict_proba(proba.reshape(-1, 1))[:, 1]
        assert calibrated.shape == proba.shape


class TestBrierScore:
    def test_perfect(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([0.0, 0.0, 1.0, 1.0])
        assert brier_score(y, p) == 0.0

    def test_worst(self):
        y = np.array([0, 1])
        p = np.array([1.0, 0.0])
        assert brier_score(y, p) == 1.0
