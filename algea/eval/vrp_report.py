"""
VRP risk report — unified reporting for the VRP sleeve.

Produces Sharpe, Sortino, Calmar, drawdown, ES, skew/kurtosis,
exposure decomposition, and scenario-based metrics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from algea.execution.options.backtest_adapter import BacktestResult
from algea.execution.options.structures import DerivativesPositionFrame
from algea.trading.derivatives_risk import (
    compute_capital_at_risk,
    compute_scenario_grid,
)


@dataclass
class VRPRiskReport:
    """Unified risk report for the VRP sleeve."""

    as_of_date: date
    # Return metrics
    sharpe: float = 0.0
    sortino: float = 0.0
    calmar: float = 0.0
    max_drawdown: float = 0.0
    total_pnl: float = 0.0

    # Tail-risk metrics
    es95: float = 0.0
    es99: float = 0.0
    worst_scenario_loss: float = 0.0

    # Distribution metrics
    skew: float = 0.0
    kurtosis: float = 0.0

    # Turnover / costs
    total_costs: float = 0.0
    num_trades: int = 0

    # Exposure decomposition
    aggregate_delta: float = 0.0
    aggregate_gamma: float = 0.0
    aggregate_vega: float = 0.0
    aggregate_theta: float = 0.0
    total_max_loss: float = 0.0

    # Scenario grid
    scenario_losses: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "as_of_date": self.as_of_date.isoformat(),
            "sharpe": self.sharpe,
            "sortino": self.sortino,
            "calmar": self.calmar,
            "max_drawdown": self.max_drawdown,
            "total_pnl": self.total_pnl,
            "es95": self.es95,
            "es99": self.es99,
            "worst_scenario_loss": self.worst_scenario_loss,
            "skew": self.skew,
            "kurtosis": self.kurtosis,
            "total_costs": self.total_costs,
            "num_trades": self.num_trades,
            "aggregate_delta": self.aggregate_delta,
            "aggregate_gamma": self.aggregate_gamma,
            "aggregate_vega": self.aggregate_vega,
            "aggregate_theta": self.aggregate_theta,
            "total_max_loss": self.total_max_loss,
            "scenario_losses": self.scenario_losses,
        }


def build_vrp_report(
    as_of_date: date,
    backtest: Optional[BacktestResult] = None,
    positions: Optional[DerivativesPositionFrame] = None,
    underlying_prices: Optional[Dict[str, float]] = None,
) -> VRPRiskReport:
    """Build a VRP risk report from backtest results and/or current positions."""
    report = VRPRiskReport(as_of_date=as_of_date)

    # From backtest metrics
    if backtest is not None:
        metrics = backtest.compute_metrics()
        report.sharpe = metrics.get("sharpe", 0.0)
        report.sortino = metrics.get("sortino", 0.0)
        report.calmar = metrics.get("calmar", 0.0)
        report.max_drawdown = metrics.get("max_drawdown", 0.0)
        report.total_pnl = metrics.get("total_pnl", 0.0)
        report.es95 = metrics.get("es95", 0.0)
        report.es99 = metrics.get("es99", 0.0)
        report.skew = metrics.get("skew", 0.0)
        report.kurtosis = metrics.get("kurtosis", 0.0)
        report.total_costs = metrics.get("total_costs", 0.0)
        report.num_trades = metrics.get("num_trades", 0)

    # From current positions
    if positions is not None:
        greeks = positions.aggregate_greeks()
        report.aggregate_delta = greeks["delta"]
        report.aggregate_gamma = greeks["gamma"]
        report.aggregate_vega = greeks["vega"]
        report.aggregate_theta = greeks["theta"]
        report.total_max_loss = compute_capital_at_risk(positions)

        # Scenario grid
        if underlying_prices:
            grid = compute_scenario_grid(positions, underlying_prices)
            if not grid.empty:
                report.worst_scenario_loss = float(grid["total_pnl"].min())
                report.scenario_losses = {
                    f"spot={row['spot_shock']:.0%}_vol={row['vol_shock']:.0%}": row["total_pnl"]
                    for _, row in grid.iterrows()
                }

    return report
