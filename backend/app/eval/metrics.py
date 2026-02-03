import numpy as np
from typing import Dict, List, Optional

def directional_accuracy(y_true: np.ndarray, y_pred_median: np.ndarray) -> float:
    """
    Computes directional accuracy.
    True if sign(y_true) == sign(y_pred_median).
    Handles 0 as neutral (or false).
    """
    # Use sign.
    s_true = np.sign(y_true)
    s_pred = np.sign(y_pred_median)

    # If pred is exactly 0, count as incorrect? Or ignore?
    # Usually swing trades are directional.
    matches = (s_true == s_pred) & (s_pred != 0)
    return float(np.mean(matches))

def coverage_probability(y_true: np.ndarray, q_lower: np.ndarray, q_upper: np.ndarray) -> float:
    """
    Fraction of times y_true falls within [q_lower, q_upper].
    Target 0.90 for 5%-95% interval.
    """
    in_bound = (y_true >= q_lower) & (y_true <= q_upper)
    return float(np.mean(in_bound))

def interval_width(q_lower: np.ndarray, q_upper: np.ndarray) -> float:
    """Average width of interval."""
    return float(np.mean(q_upper - q_lower))

def compute_metrics(y_true: np.ndarray, quantiles: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    y_true: (N,)
    quantiles: {"0.05": (N,), "0.50": (N,), "0.95": (N,)}
    """
    metrics = {}

    # Direction
    if "0.50" in quantiles:
        metrics["accuracy"] = directional_accuracy(y_true, quantiles["0.50"])

    # Calibration
    if "0.05" in quantiles and "0.95" in quantiles:
        metrics["coverage_90"] = coverage_probability(y_true, quantiles["0.05"], quantiles["0.95"])
        metrics["width_90"] = interval_width(quantiles["0.05"], quantiles["0.95"])

    return metrics
