"""
Evaluation metrics — signal quality and calibration.

Includes:
  - rank_ic: Rank IC (Spearman) between scores and forward returns
  - directional_accuracy: sign-match between actual and predicted
  - coverage_probability: fraction within quantile interval
  - interval_width: average quantile interval width
  - compute_metrics: aggregate quantile metrics dict
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Ranking signal quality
# ---------------------------------------------------------------------------

def rank_ic(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    return frame["score"].corr(frame["rank"], method="spearman")


# ---------------------------------------------------------------------------
# Quantile / directional metrics (from deprecated eval/metrics.py)
# ---------------------------------------------------------------------------

def directional_accuracy(y_true: np.ndarray, y_pred_median: np.ndarray) -> float:
    """Fraction of samples where ``sign(y_true) == sign(y_pred)`` and prediction is not zero."""
    s_true = np.sign(y_true)
    s_pred = np.sign(y_pred_median)
    matches = (s_true == s_pred) & (s_pred != 0)
    return float(np.mean(matches))


def coverage_probability(y_true: np.ndarray, q_lower: np.ndarray, q_upper: np.ndarray) -> float:
    """Fraction of true values within ``[q_lower, q_upper]`` (target 0.90 for 5–95% interval)."""
    in_bound = (y_true >= q_lower) & (y_true <= q_upper)
    return float(np.mean(in_bound))


def interval_width(q_lower: np.ndarray, q_upper: np.ndarray) -> float:
    """Average width of the prediction interval."""
    return float(np.mean(q_upper - q_lower))


def compute_metrics(y_true: np.ndarray, quantiles: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Compute aggregate quantile metrics.

    Parameters
    ----------
    y_true : ``(N,)`` true values
    quantiles : ``{"0.05": (N,), "0.50": (N,), "0.95": (N,)}``
    """
    metrics: Dict[str, float] = {}
    if "0.50" in quantiles:
        metrics["accuracy"] = directional_accuracy(y_true, quantiles["0.50"])
    if "0.05" in quantiles and "0.95" in quantiles:
        metrics["coverage_90"] = coverage_probability(y_true, quantiles["0.05"], quantiles["0.95"])
        metrics["width_90"] = interval_width(quantiles["0.05"], quantiles["0.95"])
    return metrics

