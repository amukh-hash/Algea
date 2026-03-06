from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MetricsResult:
    equity: Dict[str, float]
    trades: Dict[str, float]


def compute_equity_metrics(equity_curve: pd.DataFrame, freq: int = 252) -> Dict[str, float]:
    if equity_curve.empty:
        return {}
    equity = equity_curve["equity"].astype(float)
    returns = equity.pct_change().dropna()
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    years = max(len(equity) / freq, 1e-6)
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1
    vol = returns.std(ddof=0) * np.sqrt(freq)
    downside = returns[returns < 0].std(ddof=0) * np.sqrt(freq)
    ret_std = returns.std(ddof=0)
    sharpe = returns.mean() / ret_std * np.sqrt(freq) if ret_std else 0.0
    sortino = returns.mean() / downside * np.sqrt(freq) if downside else 0.0
    drawdown = equity / equity.cummax() - 1
    max_drawdown = drawdown.min()
    drawdown_duration = _max_run_length(drawdown < 0)
    calmar = cagr / abs(max_drawdown) if max_drawdown else 0.0
    ulcer_index = np.sqrt(np.mean(drawdown**2))
    skew = returns.skew() if not returns.empty else 0.0
    kurtosis = returns.kurtosis() if not returns.empty else 0.0
    return {
        "total_return": total_return,
        "cagr": cagr,
        "annualized_vol": vol,
        "downside_vol": downside,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "max_drawdown_duration": drawdown_duration,
        "calmar": calmar,
        "ulcer_index": float(ulcer_index),
        "skew": skew,
        "kurtosis": kurtosis,
    }


def compute_trade_metrics(trades: pd.DataFrame) -> Dict[str, float]:
    if trades.empty:
        return {}
    pnl = trades["pnl"].astype(float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    win_rate = len(wins) / max(len(trades), 1)
    profit_factor = wins.sum() / abs(losses.sum()) if losses.sum() else 0.0
    avg_win = wins.mean() if not wins.empty else 0.0
    avg_loss = losses.mean() if not losses.empty else 0.0
    payoff = avg_win / abs(avg_loss) if avg_loss else 0.0
    expectancy = pnl.mean()
    trade_days = trades["exit_date"].nunique()
    trades_per_day = len(trades) / trade_days if trade_days else 0.0
    max_wins = _max_run_length(pnl > 0)
    max_losses = _max_run_length(pnl <= 0)
    return {
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "payoff_ratio": payoff,
        "expectancy": expectancy,
        "median_trade_return": trades["ret"].median(),
        "trades": len(trades),
        "trades_per_day": trades_per_day,
        "max_consecutive_wins": max_wins,
        "max_consecutive_losses": max_losses,
    }


def compute_metrics(equity_curve: pd.DataFrame, trades: pd.DataFrame) -> MetricsResult:
    return MetricsResult(
        equity=compute_equity_metrics(equity_curve),
        trades=compute_trade_metrics(trades),
    )


def _max_run_length(mask: pd.Series) -> int:
    """Compute the maximum consecutive True run length in a boolean Series."""
    if mask.empty or not mask.any():
        return 0
    groups = (~mask).cumsum()
    return mask.groupby(groups).sum().max()
