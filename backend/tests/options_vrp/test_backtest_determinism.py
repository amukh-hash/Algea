"""Tests for backtest determinism — same inputs → same outputs, costs deducted."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from algea.execution.options.backtest_adapter import (
    BacktestResult,
    DailyPnLRecord,
    OptionsFillResult,
    simulate_entry_fill,
)
from algea.execution.options.config import VRPConfig
from algea.execution.options.structures import (
    DerivativesPosition,
    OptionLeg,
    StructureType,
)


def _make_position() -> DerivativesPosition:
    return DerivativesPosition(
        underlying="SPY",
        structure_type=StructureType.PUT_CREDIT_SPREAD,
        expiry=date(2024, 7, 15),
        legs=[
            OptionLeg("put", 395.0, -1, "sell", 3.00),
            OptionLeg("put", 390.0, 1, "buy", 1.50),
        ],
        premium_collected=1.50,
        max_loss=3.50,
        multiplier=100,
    )


class TestEntryFill:
    def test_fill_produces_credit(self):
        config = VRPConfig(slippage_bps=5.0, commission_per_contract=0.65)
        pos = _make_position()
        fill = simulate_entry_fill(pos, config, date(2024, 6, 10))
        # Credit should be positive (we receive money)
        assert fill.credit_or_debit > 0, "Entry fill should be a net credit"

    def test_commission_calculated(self):
        config = VRPConfig(commission_per_contract=0.65, exchange_fee_per_contract=0.20)
        pos = _make_position()
        fill = simulate_entry_fill(pos, config, date(2024, 6, 10))
        # 2 legs, each abs(qty)=1 → 2 contracts
        expected_commission = 2 * 0.65
        expected_fees = 2 * 0.20
        assert fill.commission == expected_commission
        assert fill.exchange_fees == expected_fees


class TestBacktestDeterminism:
    def test_same_inputs_same_output(self):
        """Metrics computation must be deterministic."""
        result1 = BacktestResult()
        result2 = BacktestResult()

        np.random.seed(42)
        pnls = np.random.normal(10, 50, 100)
        for i, pnl in enumerate(pnls):
            rec = DailyPnLRecord(
                date=date(2024, 1, 1),
                gross_pnl=float(pnl),
                costs=1.0,
                net_pnl=float(pnl) - 1.0,
                open_positions=1,
                total_max_loss=500.0,
                total_premium=150.0,
            )
            result1.daily_pnl.append(rec)
            result2.daily_pnl.append(rec)

        m1 = result1.compute_metrics()
        m2 = result2.compute_metrics()

        assert m1["sharpe"] == m2["sharpe"]
        assert m1["sortino"] == m2["sortino"]
        assert m1["max_drawdown"] == m2["max_drawdown"]

    def test_costs_reduce_total_pnl(self):
        """Net PnL should be less than gross PnL due to costs."""
        result = BacktestResult()
        for i in range(50):
            result.daily_pnl.append(DailyPnLRecord(
                date=date(2024, 1, 1),
                gross_pnl=10.0,
                costs=2.0,
                net_pnl=8.0,
                open_positions=1,
                total_max_loss=500.0,
                total_premium=150.0,
            ))

        df = result.to_pnl_df()
        gross_total = df["gross_pnl"].sum()
        net_total = df["net_pnl"].sum()
        assert net_total < gross_total

    def test_metrics_keys_present(self):
        """All expected metric keys should be present."""
        result = BacktestResult()
        for i in range(30):
            result.daily_pnl.append(DailyPnLRecord(
                date=date(2024, 1, 1),
                gross_pnl=float(np.random.normal(5, 20)),
                costs=1.0,
                net_pnl=float(np.random.normal(4, 20)),
                open_positions=1,
                total_max_loss=500.0,
                total_premium=150.0,
            ))
        metrics = result.compute_metrics()
        expected_keys = {"sharpe", "sortino", "calmar", "max_drawdown",
                         "es95", "es99", "skew", "kurtosis", "hit_rate",
                         "avg_win", "avg_loss", "total_costs", "num_trades"}
        assert expected_keys.issubset(metrics.keys())
