"""
Derivatives risk management — scenario repricing, limits enforcement,
sleeve vol targeting, convexity controls, margin heuristics, concentration.

v2: per-leg IV repricing, conservative bid/ask close marks, dynamic shocks,
    RiskScaler dataclass, margin heuristic, concentration checks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date as _date
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from algea.data.options.greeks_engine import bs_price
from algea.execution.options.config import VRPConfig
from algea.execution.options.structures import (
    DerivativesPosition,
    DerivativesPositionFrame,
)


# ═══════════════════════════════════════════════════════════════════════════
# Per-position risk
# ═══════════════════════════════════════════════════════════════════════════

def compute_max_loss(pos: DerivativesPosition) -> float:
    """Max loss in dollars for a defined-risk spread."""
    return pos.max_loss * pos.multiplier


def compute_capital_at_risk(positions: DerivativesPositionFrame) -> float:
    """Aggregate capital-at-risk across all open positions."""
    return sum(compute_max_loss(p) for p in positions.open_positions)


# ═══════════════════════════════════════════════════════════════════════════
# Margin heuristic
# ═══════════════════════════════════════════════════════════════════════════

def estimate_margin(pos: DerivativesPosition, config: VRPConfig) -> float:
    """Conservative margin heuristic for a defined-risk spread.

    estimated_margin = max_loss_dollars * margin_buffer_multiplier
    """
    return compute_max_loss(pos) * config.margin_buffer_multiplier


def compute_budget_basis(pos: DerivativesPosition, config: VRPConfig) -> float:
    """Return the binding budget quantity: risk or margin."""
    if config.budget_basis == "margin":
        return max(compute_max_loss(pos), estimate_margin(pos, config))
    return compute_max_loss(pos)


# ═══════════════════════════════════════════════════════════════════════════
# Scenario engine (v2 — per-leg IV + conservative close marks)
# ═══════════════════════════════════════════════════════════════════════════

_DEFAULT_SPOT_SHOCKS = [-0.20, -0.10, -0.05, 0.0, 0.05, 0.10]
_DEFAULT_VOL_SHOCKS = [0.0, 0.25, 0.50]
CRASH_COMBO = {"spot_shock": -0.10, "vol_shock": 0.50}

# Fallback IV when leg has entry_iv = 0 (v1 migration)
_FALLBACK_IV = 0.20


def _reprice_leg(
    spot: float,
    strike: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str,
    q: float,
    spot_shock: float,
    vol_shock: float,
) -> float:
    """Reprice a single leg under scenario shocks using BS engine."""
    new_spot = spot * (1.0 + spot_shock)
    new_sigma = sigma * (1.0 + vol_shock)
    new_sigma = max(new_sigma, 0.01)  # floor
    price = float(bs_price(new_spot, strike, T, r, new_sigma, option_type, q))
    return price


def compute_position_scenario_pnl(
    pos: DerivativesPosition,
    underlying_price: float,
    as_of_date: _date,
    risk_free_rate: float = 0.05,
    dividend_yield: float = 0.0,
    spot_shock: float = 0.0,
    vol_shock: float = 0.0,
    liquidity_widen_factor: float = 1.0,
) -> float:
    """Compute PnL for a single position under a scenario shock.

    Uses per-leg entry_iv (with fallback) and conservative close marks:
    - Short legs: buyback at ask (widened)
    - Long legs: sell at bid (widened)
    """
    dte = max((pos.expiry - as_of_date).days, 1)
    T = dte / 365.0

    total_pnl = 0.0
    for leg in pos.legs:
        # Use per-leg IV; fallback for v1 positions
        sigma = leg.entry_iv if leg.entry_iv > 0 else _FALLBACK_IV

        # Underlying price at entry (for accurate repricing base)
        spot = leg.entry_underlying if leg.entry_underlying > 0 else underlying_price

        new_price = _reprice_leg(
            spot, leg.strike, T, risk_free_rate,
            sigma, leg.option_type, dividend_yield,
            spot_shock, vol_shock,
        )

        # Conservative close marks with liquidity widening
        entry_price = leg.entry_price_mid
        if leg.qty < 0:
            # Short leg: must BUY BACK at ask → increase close price
            close_price = new_price * (1.0 + 0.005 * liquidity_widen_factor)
        else:
            # Long leg: must SELL at bid → decrease close price
            close_price = new_price * (1.0 - 0.005 * liquidity_widen_factor)

        pnl = (close_price - entry_price) * leg.qty * pos.multiplier
        total_pnl += pnl

    return total_pnl


def compute_scenario_grid(
    positions: DerivativesPositionFrame,
    underlying_prices: Dict[str, float],
    as_of_date: Optional[_date] = None,
    spot_shocks: Optional[List[float]] = None,
    vol_shocks: Optional[List[float]] = None,
    risk_free_rate: float = 0.05,
    liquidity_widen_factor: float = 1.0,
) -> pd.DataFrame:
    """Compute scenario grid: PnL under each (spot_shock, vol_shock) combo."""
    if as_of_date is None:
        as_of_date = _date.today()
    if spot_shocks is None:
        spot_shocks = _DEFAULT_SPOT_SHOCKS
    if vol_shocks is None:
        vol_shocks = _DEFAULT_VOL_SHOCKS

    rows = []
    for ss in spot_shocks:
        for vs in vol_shocks:
            total = 0.0
            per_und: Dict[str, float] = {}
            for pos in positions.open_positions:
                price = underlying_prices.get(pos.underlying, 0.0)
                pnl = compute_position_scenario_pnl(
                    pos, price, as_of_date, risk_free_rate, 0.0,
                    ss, vs, liquidity_widen_factor,
                )
                total += pnl
                per_und[pos.underlying] = per_und.get(pos.underlying, 0.0) + pnl
            rows.append({
                "spot_shock": ss,
                "vol_shock": vs,
                "total_pnl": total,
                **{f"pnl_{k}": v for k, v in per_und.items()},
            })
    return pd.DataFrame(rows)


def compute_scenario_with_contributors(
    positions: DerivativesPositionFrame,
    underlying_prices: Dict[str, float],
    as_of_date: Optional[_date] = None,
    spot_shocks: Optional[List[float]] = None,
    vol_shocks: Optional[List[float]] = None,
    risk_free_rate: float = 0.05,
    liquidity_widen_factor: float = 1.0,
) -> Tuple[float, Dict[str, float]]:
    """Compute worst-case scenario loss and per-position contributions.

    Returns
    -------
    (worst_case_total, {position_id: worst_case_pnl})
    """
    if as_of_date is None:
        as_of_date = _date.today()
    if spot_shocks is None:
        spot_shocks = _DEFAULT_SPOT_SHOCKS
    if vol_shocks is None:
        vol_shocks = _DEFAULT_VOL_SHOCKS

    # For each position, find its worst-case PnL across all scenarios
    per_pos_worst: Dict[str, float] = {}
    for pos in positions.open_positions:
        price = underlying_prices.get(pos.underlying, 0.0)
        worst = 0.0
        for ss in spot_shocks:
            for vs in vol_shocks:
                pnl = compute_position_scenario_pnl(
                    pos, price, as_of_date, risk_free_rate, 0.0,
                    ss, vs, liquidity_widen_factor,
                )
                worst = min(worst, pnl)
        per_pos_worst[pos.position_id] = worst

    total_worst = sum(per_pos_worst.values())
    return total_worst, per_pos_worst


# ═══════════════════════════════════════════════════════════════════════════
# Dynamic shock grid (forecast-calibrated)
# ═══════════════════════════════════════════════════════════════════════════

def build_dynamic_shock_grid(
    rv10_pred_p95: Optional[float] = None,
    rv10_pred_p99: Optional[float] = None,
    config: Optional[VRPConfig] = None,
) -> Tuple[List[float], List[float]]:
    """Build spot shocks calibrated by Lag-Llama forecast output.

    Fixed shocks are always included as floors.  Dynamic shocks are added
    if forecasts are available and ``use_dynamic_shocks`` is enabled.

    Returns (spot_shocks, vol_shocks).
    """
    cfg = config or VRPConfig()
    spot_shocks = list(_DEFAULT_SPOT_SHOCKS)
    vol_shocks = list(_DEFAULT_VOL_SHOCKS)

    if cfg.use_dynamic_shocks and rv10_pred_p95 is not None:
        # Convert annualised RV to daily, then to 10-day shock
        rv_daily = rv10_pred_p95 / np.sqrt(252)
        shock_95 = min(max(0.10, cfg.dynamic_shock_k * rv_daily * np.sqrt(10)), 0.40)
        if -shock_95 not in spot_shocks:
            spot_shocks.append(-shock_95)
            spot_shocks.sort()

    if cfg.use_dynamic_shocks and rv10_pred_p99 is not None:
        rv_daily = rv10_pred_p99 / np.sqrt(252)
        shock_99 = min(max(0.20, cfg.dynamic_shock_k * rv_daily * np.sqrt(10)), 0.50)
        if -shock_99 not in spot_shocks:
            spot_shocks.append(-shock_99)
            spot_shocks.sort()

    return spot_shocks, vol_shocks


# ═══════════════════════════════════════════════════════════════════════════
# Limits enforcement
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class LimitCheckResult:
    passed: bool
    violations: List[str]


def check_risk_limits(
    positions: DerivativesPositionFrame,
    nav: float,
    config: VRPConfig,
) -> LimitCheckResult:
    """Check all risk limits.  Returns detailed violation report."""
    violations: List[str] = []

    # Per-structure budget
    for pos in positions.open_positions:
        budget = compute_budget_basis(pos, config) / nav
        if budget > config.max_risk_per_structure_pct_nav:
            violations.append(
                f"Position {pos.position_id} ({pos.underlying}): "
                f"budget {budget:.2%} > limit {config.max_risk_per_structure_pct_nav:.2%}"
            )

    # Aggregate budget
    total_budget = sum(
        compute_budget_basis(p, config) for p in positions.open_positions
    ) / nav if nav > 0 else 0
    if total_budget > config.max_total_vrp_risk_pct_nav:
        violations.append(
            f"Total VRP budget {total_budget:.2%} > limit {config.max_total_vrp_risk_pct_nav:.2%}"
        )

    # Per-underlying position count
    by_und = positions.positions_by_underlying()
    for und, plist in by_und.items():
        if len(plist) > config.max_positions_per_underlying:
            violations.append(
                f"{und}: {len(plist)} positions > limit {config.max_positions_per_underlying}"
            )

    # Greeks limits
    for und, plist in by_und.items():
        und_vega = sum(abs(p.vega) * p.multiplier for p in plist)
        und_gamma = sum(abs(p.gamma) * p.multiplier for p in plist)
        if und_vega > config.max_vega_per_underlying:
            violations.append(f"{und}: vega {und_vega:.1f} > limit {config.max_vega_per_underlying}")
        if und_gamma > config.max_gamma_per_underlying:
            violations.append(f"{und}: gamma {und_gamma:.1f} > limit {config.max_gamma_per_underlying}")

    return LimitCheckResult(
        passed=len(violations) == 0,
        violations=violations,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Concentration checks
# ═══════════════════════════════════════════════════════════════════════════

def check_expiry_concentration(
    positions: DerivativesPositionFrame,
    config: VRPConfig,
) -> LimitCheckResult:
    """Ensure no single expiry holds more than max_loss_pct_single_expiry of total."""
    violations: List[str] = []
    total_ml = positions.total_max_loss()
    if total_ml <= 0:
        return LimitCheckResult(passed=True, violations=[])

    by_exp = positions.positions_by_expiry()
    for exp, plist in by_exp.items():
        exp_ml = sum(p.max_loss * p.multiplier for p in plist)
        pct = exp_ml / total_ml
        if pct > config.max_loss_pct_single_expiry:
            violations.append(
                f"Expiry {exp}: {pct:.0%} of max_loss > limit {config.max_loss_pct_single_expiry:.0%}"
            )

    return LimitCheckResult(passed=len(violations) == 0, violations=violations)


def check_strike_spacing(
    positions: DerivativesPositionFrame,
    underlying_prices: Dict[str, float],
    config: VRPConfig,
) -> LimitCheckResult:
    """Ensure short strikes are sufficiently spaced."""
    violations: List[str] = []
    by_und = positions.positions_by_underlying()

    for und, plist in by_und.items():
        price = underlying_prices.get(und, 0.0)
        if price <= 0:
            continue
        short_strikes = []
        for pos in plist:
            for leg in pos.legs:
                if leg.qty < 0:
                    short_strikes.append(leg.strike)

        short_strikes.sort()
        for i in range(1, len(short_strikes)):
            spacing = abs(short_strikes[i] - short_strikes[i - 1]) / price
            if spacing < config.min_short_strike_spacing_pct:
                violations.append(
                    f"{und}: strikes {short_strikes[i-1]}/{short_strikes[i]} "
                    f"spacing {spacing:.2%} < min {config.min_short_strike_spacing_pct:.2%}"
                )

    return LimitCheckResult(passed=len(violations) == 0, violations=violations)


# ═══════════════════════════════════════════════════════════════════════════
# Sleeve vol targeting (v2 — RiskScaler)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class RiskScaler:
    """Result of the sizing scaler computation."""
    final_scaler: float
    pnl_scaler: float
    proxy_scaler: float
    realized_vol_ann: float
    target_vol: float


def compute_risk_scaler(
    recent_pnl: pd.Series,
    target_vol: float,
    lookback: int = 20,
    aggregate_vega: float = 0.0,
    aggregate_gamma: float = 0.0,
    forecast_rv_p50: Optional[float] = None,
) -> RiskScaler:
    """Compute sizing scaler = min(pnl_scaler, proxy_scaler), bounded.

    Parameters
    ----------
    recent_pnl : recent daily PnL series
    target_vol : target annualised vol
    lookback : rolling window
    aggregate_vega/gamma : current greeks exposure
    forecast_rv_p50 : predicted 10-day RV (median, annualised)
    """
    # PnL-based scaler
    pnl_scaler = 1.0
    ann_vol = 0.0
    if len(recent_pnl) >= max(lookback // 2, 5):
        tail = recent_pnl.iloc[-lookback:]
        daily_vol = tail.std()
        if daily_vol > 0 and not np.isnan(daily_vol):
            ann_vol = daily_vol * np.sqrt(252)
            pnl_scaler = target_vol / ann_vol
    pnl_scaler = float(np.clip(pnl_scaler, 0.25, 2.0))

    # Greeks-proxy scaler (higher exposure → lower scaler)
    proxy_scaler = 1.0
    exposure = abs(aggregate_vega) * 0.01 + abs(aggregate_gamma) * 10.0
    if exposure > 0:
        # Crude normalisation: 100 vega + 10 gamma ≈ neutral zone
        proxy_scaler = max(0.25, min(2.0, 200.0 / max(exposure, 1.0)))

    # Forecast adjustment
    if forecast_rv_p50 is not None and forecast_rv_p50 > 0:
        forecast_scaler = target_vol / forecast_rv_p50
        forecast_scaler = float(np.clip(forecast_scaler, 0.25, 2.0))
        proxy_scaler = min(proxy_scaler, forecast_scaler)

    final = min(pnl_scaler, proxy_scaler)

    return RiskScaler(
        final_scaler=float(np.clip(final, 0.25, 2.0)),
        pnl_scaler=pnl_scaler,
        proxy_scaler=proxy_scaler,
        realized_vol_ann=ann_vol,
        target_vol=target_vol,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Scenario headroom (Phase 3)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class HeadroomResult:
    """Scenario budget headroom analysis."""
    headroom_pct: float       # remaining budget as % of NAV
    block_new_entries: bool   # True if headroom < min
    scale_factor: float       # [0, 1] multiplier for RiskScaler

def compute_scenario_headroom(
    scenario_worst_loss_pct: float,
    config: VRPConfig,
) -> HeadroomResult:
    """Compute scenario headroom and entry/scaler adjustments.

    Parameters
    ----------
    scenario_worst_loss_pct : current worst-case scenario loss as fraction of NAV
    config : VRPConfig with budget and headroom thresholds

    Returns
    -------
    HeadroomResult with headroom, block flag, and scaler factor.
    """
    budget = config.max_worst_case_scenario_loss_pct_nav
    headroom = budget - scenario_worst_loss_pct
    min_hr = config.min_headroom_for_new_entries

    if headroom < min_hr:
        # Proportional reduction; floor at 0
        scale = max(0.0, headroom / min_hr) if min_hr > 0 else 0.0
        return HeadroomResult(
            headroom_pct=headroom,
            block_new_entries=True,
            scale_factor=scale,
        )

    return HeadroomResult(
        headroom_pct=headroom,
        block_new_entries=False,
        scale_factor=1.0,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Convexity controls
# ═══════════════════════════════════════════════════════════════════════════

def short_convexity_score(positions: DerivativesPositionFrame) -> float:
    """Compute aggregate short convexity score.

    score ≈ |aggregate gamma| + |aggregate vega| * sensitivity
    Higher score → more dangerous short-convexity clustering.
    """
    greeks = positions.aggregate_greeks()
    gamma_component = abs(min(greeks["gamma"], 0.0))
    vega_component = abs(min(greeks["vega"], 0.0)) * 0.1
    return gamma_component + vega_component


# ═══════════════════════════════════════════════════════════════════════════
# Early assignment guard
# ═══════════════════════════════════════════════════════════════════════════

def check_early_assignment_risk(
    pos: DerivativesPosition,
    underlying_price: float,
    as_of_date: _date,
    config: VRPConfig,
    risk_free_rate: float = 0.05,
    dividend_yield: float = 0.0,
) -> bool:
    """Return True if position is at risk of early assignment.

    A short put is at risk if it is ITM and extrinsic value is below threshold.
    """
    for leg in pos.legs:
        if leg.qty >= 0:
            continue  # only check short legs
        if leg.option_type != "put":
            continue

        dte = max((pos.expiry - as_of_date).days, 1)
        T = dte / 365.0
        iv = leg.entry_iv if leg.entry_iv > 0 else _FALLBACK_IV
        bs_val = float(bs_price(
            underlying_price, leg.strike, T, risk_free_rate, iv, "put", dividend_yield,
        ))
        intrinsic = max(leg.strike - underlying_price, 0.0)
        extrinsic = max(bs_val - intrinsic, 0.0)

        if intrinsic > 0 and extrinsic < config.early_assignment_extrinsic_threshold:
            return True

    return False
