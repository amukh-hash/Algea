"""
Gamma danger-zone guard — two-signal short-leg proximity detector.

Detects when the underlying is dangerously close to the short strike
of a put credit spread using both delta-based and sigma-distance checks.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date as _date
from typing import List, Optional

import numpy as np

from algaie.data.options.greeks_engine import bs_delta
from algaie.execution.options.config import VRPConfig
from algaie.execution.options.structures import DerivativesPosition


# ═══════════════════════════════════════════════════════════════════════════
# Result
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class DangerZoneResult:
    """Result of a danger-zone check for a single position."""
    position_id: str
    in_danger: bool
    short_delta_current: float
    z_score: float               # sigma-distance to short strike
    delta_triggered: bool
    z_triggered: bool
    action: str                  # "none" | "tighten_stop" | "close"


# ═══════════════════════════════════════════════════════════════════════════
# Guard
# ═══════════════════════════════════════════════════════════════════════════

def check_danger_zone(
    pos: DerivativesPosition,
    underlying_price: float,
    rv_est_daily: float,
    as_of_date: _date,
    config: VRPConfig,
    risk_free_rate: float = 0.05,
    dividend_yield: float = 0.0,
) -> DangerZoneResult:
    """Check if a position's short leg is in the danger zone.

    Two independent signals:
    1. Delta-based: |current delta of short leg| > threshold
    2. Sigma-distance: spot is within ``z_threshold`` std devs of short strike

    Parameters
    ----------
    pos : the derivatives position to check
    underlying_price : current spot price
    rv_est_daily : estimated daily realised vol (NOT annualised)
    as_of_date : current date for DTE computation
    config : VRPConfig with danger_zone thresholds

    Returns
    -------
    DangerZoneResult with action recommendation.
    """
    # Find the short put leg
    short_leg = None
    for leg in pos.legs:
        if leg.qty < 0 and leg.option_type == "put":
            short_leg = leg
            break

    if short_leg is None:
        return DangerZoneResult(
            position_id=pos.position_id,
            in_danger=False,
            short_delta_current=0.0,
            z_score=0.0,
            delta_triggered=False,
            z_triggered=False,
            action="none",
        )

    # DTE
    dte = max((pos.expiry - as_of_date).days, 1)
    T = dte / 365.0

    # Current delta of short leg
    iv = short_leg.entry_iv if short_leg.entry_iv > 0 else 0.20  # fallback
    current_delta = float(np.abs(bs_delta(
        underlying_price, short_leg.strike, T, risk_free_rate, iv, "put", dividend_yield,
    )))

    # Sigma-distance: how many daily vol moves away is the short strike?
    daily_dollar_vol = underlying_price * max(rv_est_daily, 1e-6)
    distance = underlying_price - short_leg.strike
    z_score = distance / (daily_dollar_vol * np.sqrt(T * 252)) if daily_dollar_vol > 0 else 10.0

    delta_triggered = current_delta > config.danger_zone_delta_threshold
    z_triggered = z_score < config.danger_zone_z_threshold

    if config.danger_zone_close_if_both:
        in_danger = delta_triggered and z_triggered
    else:
        in_danger = delta_triggered or z_triggered

    action = "none"
    if in_danger:
        action = "close"
    elif delta_triggered or z_triggered:
        action = "tighten_stop"

    return DangerZoneResult(
        position_id=pos.position_id,
        in_danger=in_danger,
        short_delta_current=current_delta,
        z_score=z_score,
        delta_triggered=delta_triggered,
        z_triggered=z_triggered,
        action=action,
    )


def check_all_danger_zones(
    positions: List[DerivativesPosition],
    underlying_prices: dict[str, float],
    rv_estimates: dict[str, float],
    as_of_date: _date,
    config: VRPConfig,
) -> List[DangerZoneResult]:
    """Check danger zone for all open positions."""
    results = []
    for pos in positions:
        if not pos.is_open:
            continue
        price = underlying_prices.get(pos.underlying, 0.0)
        rv = rv_estimates.get(pos.underlying, 0.01)
        results.append(check_danger_zone(pos, price, rv, as_of_date, config))
    return results
