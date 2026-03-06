"""
Cost model for portfolio rebalancing.

Applies linear commission/spread costs and optional market-impact costs
at each rebalance.  Turnover is defined as 1-way (sum of buys / notional).

The primary API is ``compute_turnover_and_cost`` which is a *pure* function
of portfolio weights and cost config — it never sees realised returns.
This ensures costs are impossible to accidentally couple to returns.

A thin backward-compatible wrapper ``apply_costs`` is kept for legacy callers.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


@dataclass
class CostConfig:
    """Cost model parameters."""

    cost_bps: float = 10.0     # all-in linear cost per 1-way notional
    impact_bps: float = 0.0    # extra cost (e.g. sqrt-impact), additive
    min_cost_bps: float = 0.0  # floor on total cost bps


def compute_turnover_and_cost(
    prev_w: Dict[str, float],
    new_w: Dict[str, float],
    cfg: CostConfig,
) -> Tuple[float, float]:
    """Compute turnover and transaction cost — *pure*, no returns arg.

    This is the production API.  Cost depends *only* on weight deltas
    and cost configuration, never on realised returns.

    Parameters
    ----------
    prev_w : dict
        symbol → weight at start of period (before rebalance).
    new_w : dict
        symbol → weight after rebalance.
    cfg : CostConfig
        Cost model parameters.

    Returns
    -------
    turnover_1way : float
        1-way turnover = 0.5 * sum(|Δw|).
    cost : float
        Transaction cost as a fraction of portfolio notional.
    """
    all_syms = set(prev_w) | set(new_w)

    # Sum of absolute weight changes
    sum_abs_delta = sum(
        abs(new_w.get(s, 0.0) - prev_w.get(s, 0.0))
        for s in all_syms
    )

    # 1-way turnover
    turnover_1way = 0.5 * sum_abs_delta

    # Total cost bps (floor-bounded)
    total_bps = max(cfg.cost_bps + cfg.impact_bps, cfg.min_cost_bps)

    # Cost as fraction of portfolio notional
    cost = turnover_1way * total_bps / 10_000.0

    return turnover_1way, cost


# ── Backward-compatible wrapper ─────────────────────────────────────────

def apply_costs(
    prev_w: Dict[str, float],
    new_w: Dict[str, float],
    gross_return: float,
    cfg: CostConfig,
) -> Tuple[float, float, float]:
    """Backward-compatible wrapper around ``compute_turnover_and_cost``.

    .. deprecated::
        Prefer ``compute_turnover_and_cost`` which is a pure function
        that does not accept (or depend on) realised returns.

    Returns
    -------
    net_return : float
        ``gross_return - cost``.
    cost : float
        Transaction cost as a fraction of portfolio notional.
    turnover_1way : float
        1-way turnover = 0.5 * sum(|Δw|).
    """
    turnover_1way, cost = compute_turnover_and_cost(prev_w, new_w, cfg)
    net_return = gross_return - cost
    return net_return, cost, turnover_1way
