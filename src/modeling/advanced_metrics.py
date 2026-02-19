"""
Advanced evaluation metrics for fraud detection models:
confusion matrix at threshold, KS statistic, lift/gain charts,
calibration curve, expected calibration error.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as scipy_stats


def confusion_matrix_at_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Confusion matrix metrics at a given probability threshold.
    Returns TP, FP, TN, FN, precision, recall, F1, specificity.
    """
    y_true = np.asarray(y_true).ravel()
    pred = (np.asarray(proba).ravel() >= threshold).astype(int)

    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "threshold": threshold,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
    }


def ks_statistic(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    """
    Kolmogorov-Smirnov statistic measuring separation between
    fraud and non-fraud score distributions.
    Returns KS statistic, p-value, and threshold at max separation.
    """
    y_true = np.asarray(y_true).ravel()
    proba = np.asarray(proba).ravel()

    fraud_scores = proba[y_true == 1]
    legit_scores = proba[y_true == 0]

    if len(fraud_scores) == 0 or len(legit_scores) == 0:
        return {"ks_stat": 0.0, "p_value": 1.0, "ks_threshold": 0.5}

    ks_stat, p_value = scipy_stats.ks_2samp(fraud_scores, legit_scores)

    # Find threshold at max KS
    thresholds = np.linspace(0, 1, 200)
    ks_values = []
    for t in thresholds:
        tpr = (fraud_scores >= t).mean()
        fpr = (legit_scores >= t).mean()
        ks_values.append(tpr - fpr)
    best_idx = int(np.argmax(ks_values))

    return {
        "ks_stat": float(ks_stat),
        "p_value": float(p_value),
        "ks_threshold": float(thresholds[best_idx]),
    }


def lift_chart_data(
    y_true: np.ndarray,
    proba: np.ndarray,
    n_bins: int = 10,
) -> list[dict[str, float]]:
    """
    Decile lift analysis. Returns list of dicts with decile, n, n_fraud,
    fraud_rate, lift (vs overall fraud rate).
    """
    y_true = np.asarray(y_true).ravel()
    proba = np.asarray(proba).ravel()
    overall_rate = y_true.mean()
    if overall_rate == 0:
        return []

    order = np.argsort(-proba)
    y_sorted = y_true[order]
    bin_size = len(y_sorted) // n_bins

    results = []
    for i in range(n_bins):
        start = i * bin_size
        end = start + bin_size if i < n_bins - 1 else len(y_sorted)
        y_bin = y_sorted[start:end]
        rate = y_bin.mean()
        results.append({
            "decile": i + 1,
            "n": int(end - start),
            "n_fraud": int(y_bin.sum()),
            "fraud_rate": float(rate),
            "lift": float(rate / overall_rate),
        })
    return results


def gain_chart_data(
    y_true: np.ndarray,
    proba: np.ndarray,
    n_bins: int = 10,
) -> list[dict[str, float]]:
    """
    Cumulative gain chart. Returns list of dicts with
    pct_population, pct_fraud_captured (cumulative).
    """
    y_true = np.asarray(y_true).ravel()
    proba = np.asarray(proba).ravel()
    total_fraud = y_true.sum()
    if total_fraud == 0:
        return []

    order = np.argsort(-proba)
    y_sorted = y_true[order]

    results = []
    cum_fraud = 0
    bin_size = len(y_sorted) // n_bins
    for i in range(n_bins):
        start = i * bin_size
        end = start + bin_size if i < n_bins - 1 else len(y_sorted)
        cum_fraud += y_sorted[start:end].sum()
        results.append({
            "pct_population": float((end) / len(y_sorted)),
            "pct_fraud_captured": float(cum_fraud / total_fraud),
        })
    return results


def calibration_curve_data(
    y_true: np.ndarray,
    proba: np.ndarray,
    n_bins: int = 10,
) -> list[dict[str, float]]:
    """
    Reliability diagram data. For each bin: mean predicted probability
    vs actual fraction of positives.
    """
    y_true = np.asarray(y_true).ravel()
    proba = np.asarray(proba).ravel()

    edges = np.linspace(0, 1, n_bins + 1)
    results = []
    for i in range(n_bins):
        mask = (proba >= edges[i]) & (proba < edges[i + 1])
        if i == n_bins - 1:
            mask = (proba >= edges[i]) & (proba <= edges[i + 1])
        if mask.sum() == 0:
            continue
        results.append({
            "bin_lower": float(edges[i]),
            "bin_upper": float(edges[i + 1]),
            "mean_predicted": float(proba[mask].mean()),
            "fraction_positive": float(y_true[mask].mean()),
            "count": int(mask.sum()),
        })
    return results


def expected_calibration_error(
    y_true: np.ndarray,
    proba: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error (ECE).
    Weighted average of |fraction_positive - mean_predicted| across bins.
    Lower is better.
    """
    bins = calibration_curve_data(y_true, proba, n_bins)
    if not bins:
        return 0.0
    total = sum(b["count"] for b in bins)
    ece = sum(
        b["count"] * abs(b["fraction_positive"] - b["mean_predicted"])
        for b in bins
    ) / total
    return float(ece)
