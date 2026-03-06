"""
StatArb Beta-Neutral Weight Optimizer.

Replaces the naive dollar-neutral stub with a scipy-based convex optimizer
that enforces BOTH dollar-neutrality (sum w = 0) AND beta-neutrality
(sum w*beta = 0).

Without this, holding $10k XLK (beta ~1.2) long and $10k XLU (beta ~0.5)
short exposes the portfolio to a 1.4% loss on a 2% market drawdown,
even though it appears "market-neutral" in dollar terms.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Default sector ETF betas (estimated rolling 252d vs SPY)
# Updated 2026-03-04. In production, these should be fetched live.
DEFAULT_ETF_BETAS = {
    "XLF": 1.04, "XLK": 1.18, "XLE": 0.92, "XLI": 1.06, "XLY": 1.12,
    "XLP": 0.66, "XLV": 0.82, "XLU": 0.52, "XLB": 1.00, "XLRE": 0.78,
    "XLC": 1.05, "SPY": 1.00, "QQQ": 1.15,
}


def beta_neutralize(
    weights: dict[str, float],
    betas: Optional[dict[str, float]] = None,
) -> dict[str, float]:
    """Optimize weights to be dollar-neutral AND beta-neutral.

    Parameters
    ----------
    weights : dict
        Raw target weights (can be dollar-neutral or not).
    betas : dict, optional
        Per-symbol beta to SPY. Falls back to ``DEFAULT_ETF_BETAS``.

    Returns
    -------
    dict
        Optimized weights with sum(w)=0 and sum(w*beta)=0.
    """
    if not weights:
        return {}

    if betas is None:
        betas = DEFAULT_ETF_BETAS

    symbols = list(weights.keys())
    n = len(symbols)
    if n < 2:
        return weights

    alpha = np.array([weights[s] for s in symbols])
    beta_arr = np.array([betas.get(s, 1.0) for s in symbols])

    try:
        from scipy.optimize import minimize

        def objective(w):
            return -np.dot(w, alpha) + 0.5 * np.sum(w ** 2)

        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w)},            # Dollar neutral
            {"type": "eq", "fun": lambda w: np.dot(w, beta_arr)},  # Beta neutral
        ]
        bounds = [(-1.0, 1.0) for _ in range(n)]

        res = minimize(objective, np.zeros(n), bounds=bounds, constraints=constraints)

        if res.success and np.sum(np.abs(res.x)) > 1e-8:
            opt_w = res.x / np.sum(np.abs(res.x))  # Scale gross exposure to 1.0
            return {s: float(round(opt_w[i], 6)) for i, s in enumerate(symbols)}

        logger.warning("Beta-neutral optimizer did not converge; falling back to dollar-neutral.")
    except ImportError:
        logger.warning("scipy not available; falling back to dollar-neutral.")

    # Fallback: naive dollar-neutral
    net = sum(weights.values())
    adj = net / max(n, 1)
    return {s: w - adj for s, w in weights.items()}
