"""
Options backtest adapter — realistic fill simulation, MTM, assignment risk,
and full PnL tracking for the VRP sleeve.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from algae.execution.options.config import VRPConfig
from algae.execution.options.structures import (
    DerivativesPosition,
    DerivativesPositionFrame,
    OptionLeg,
)


# ═══════════════════════════════════════════════════════════════════════════
# Fill model
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class OptionsFillResult:
    """Result of a single spread entry/exit fill."""
    credit_or_debit: float     # positive = credit, negative = debit
    commission: float
    exchange_fees: float
    fill_date: date
    slippage_cost: float


def simulate_entry_fill(
    pos: DerivativesPosition,
    config: VRPConfig,
    fill_date: date,
) -> OptionsFillResult:
    """Simulate entry fill using bid/ask-based pricing.

    Credit = sell_leg at bid - buy_leg at ask
    (conservative: we sell at bid, buy at ask)
    """
    credit = 0.0
    for leg in pos.legs:
        if leg.side == "sell":
            # Sell at bid (worst case for us)
            leg_price = leg.entry_price_mid * (1.0 - config.slippage_bps / 10_000)
            credit += leg_price
        else:
            # Buy at ask (worst case for us)
            leg_price = leg.entry_price_mid * (1.0 + config.slippage_bps / 10_000)
            credit -= leg_price

    n_contracts = sum(abs(leg.qty) for leg in pos.legs)
    commission = n_contracts * config.commission_per_contract
    exchange_fees = n_contracts * config.exchange_fee_per_contract
    slippage_cost = abs(pos.premium_collected - credit) * pos.multiplier

    return OptionsFillResult(
        credit_or_debit=credit * pos.multiplier,
        commission=commission,
        exchange_fees=exchange_fees,
        fill_date=fill_date,
        slippage_cost=slippage_cost,
    )


def simulate_exit_fill(
    pos: DerivativesPosition,
    current_marks: Dict[str, float],
    config: VRPConfig,
    fill_date: date,
) -> OptionsFillResult:
    """Simulate exit fill.

    Debit = buy_back_short at ask - sell_long at bid
    """
    mark = current_marks.get(pos.position_id, pos.premium_collected * 0.5)

    # Apply slippage: we pay more to close
    debit = mark * (1.0 + config.slippage_bps / 10_000)

    n_contracts = sum(abs(leg.qty) for leg in pos.legs)
    commission = n_contracts * config.commission_per_contract
    exchange_fees = n_contracts * config.exchange_fee_per_contract

    return OptionsFillResult(
        credit_or_debit=-debit * pos.multiplier,
        commission=commission,
        exchange_fees=exchange_fees,
        fill_date=fill_date,
        slippage_cost=abs(mark - debit) * pos.multiplier,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Backtest adapter
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DailyPnLRecord:
    date: date
    gross_pnl: float
    costs: float
    net_pnl: float
    open_positions: int
    total_max_loss: float
    total_premium: float


@dataclass
class BacktestResult:
    """Container for full backtest output."""
    daily_pnl: List[DailyPnLRecord] = field(default_factory=list)
    trade_log: List[Dict[str, Any]] = field(default_factory=list)

    def to_pnl_df(self) -> pd.DataFrame:
        rows = [
            {
                "date": r.date,
                "gross_pnl": r.gross_pnl,
                "costs": r.costs,
                "net_pnl": r.net_pnl,
                "open_positions": r.open_positions,
                "total_max_loss": r.total_max_loss,
            }
            for r in self.daily_pnl
        ]
        return pd.DataFrame(rows)

    def to_trade_log_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.trade_log)

    def compute_metrics(self) -> Dict[str, float]:
        """Compute summary metrics from daily PnL."""
        df = self.to_pnl_df()
        if df.empty:
            return {}
        returns = df["net_pnl"]
        cum = returns.cumsum()

        sharpe = _sharpe(returns)
        sortino = _sortino(returns)
        max_dd = _max_drawdown(cum)
        calmar = _calmar(returns, max_dd)
        es95 = _expected_shortfall(returns, 0.05)
        es99 = _expected_shortfall(returns, 0.01)
        skew = float(returns.skew()) if len(returns) > 3 else 0.0
        kurtosis = float(returns.kurtosis()) if len(returns) > 4 else 0.0

        wins = returns[returns > 0]
        losses = returns[returns < 0]
        hit_rate = len(wins) / len(returns) if len(returns) > 0 else 0.0
        avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
        avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0

        return {
            "total_pnl": float(cum.iloc[-1]) if not cum.empty else 0.0,
            "sharpe": sharpe,
            "sortino": sortino,
            "calmar": calmar,
            "max_drawdown": max_dd,
            "es95": es95,
            "es99": es99,
            "skew": skew,
            "kurtosis": kurtosis,
            "hit_rate": hit_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "total_costs": float(df["costs"].sum()),
            "num_trades": len(self.trade_log),
            "trading_days": len(df),
        }


# ═══════════════════════════════════════════════════════════════════════════
# Metric helpers
# ═══════════════════════════════════════════════════════════════════════════

def _sharpe(returns: pd.Series, risk_free: float = 0.0) -> float:
    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free / 252
    std = excess.std()
    if std <= 0 or np.isnan(std):
        return 0.0
    return float(excess.mean() / std * np.sqrt(252))


def _sortino(returns: pd.Series, risk_free: float = 0.0) -> float:
    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free / 252
    downside = excess[excess < 0]
    if len(downside) == 0:
        return float("inf") if excess.mean() > 0 else 0.0
    down_std = downside.std()
    if down_std <= 0 or np.isnan(down_std):
        return 0.0
    return float(excess.mean() / down_std * np.sqrt(252))


def _max_drawdown(cum_pnl: pd.Series) -> float:
    if cum_pnl.empty:
        return 0.0
    peak = cum_pnl.cummax()
    dd = cum_pnl - peak
    return float(dd.min())


def _calmar(returns: pd.Series, max_dd: float) -> float:
    if max_dd >= 0 or len(returns) < 2:
        return 0.0
    ann_return = float(returns.mean() * 252)
    return ann_return / abs(max_dd)


def _expected_shortfall(returns: pd.Series, alpha: float = 0.05) -> float:
    """Historical expected shortfall (CVaR)."""
    if len(returns) < 10:
        return 0.0
    cutoff = returns.quantile(alpha)
    tail = returns[returns <= cutoff]
    return float(tail.mean()) if len(tail) > 0 else 0.0
