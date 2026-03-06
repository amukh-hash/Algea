"""Runner integration tests for the IBKR paper-trading cycle.

Tests cover noop mode, guard enforcement, regime handling, and
reconciliation output — all without an IBKR gateway.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import pytest

from algae.trading.orders import OrderIntent
from algae.trading.paper_guards_futures import (
    GuardResult,
    PaperGuardConfig,
    apply_paper_guards,
)
from sleeves.cooc_reversal_futures.signal_mode import (
    SignalMode,
    heuristic_predictions,
    resolve_signal_mode,
)
from backend.app.execution.reconcile_futures import reconcile_day


# ---------------------------------------------------------------------------
# Signal mode tests
# ---------------------------------------------------------------------------


class TestSignalMode:
    def test_no_pack_returns_heuristic(self) -> None:
        assert resolve_signal_mode(None) == SignalMode.HEURISTIC

    def test_missing_dir_returns_heuristic(self) -> None:
        assert resolve_signal_mode("/nonexistent/path") == SignalMode.HEURISTIC

    def test_heuristic_predictions_retco(self) -> None:
        df = pd.DataFrame({
            "root": ["ES", "ES", "NQ", "NQ"],
            "trading_day": pd.date_range("2026-02-10", periods=4, freq="D"),
            "ret_co": [0.01, -0.005, 0.02, -0.01],
        })
        preds = heuristic_predictions(df, ["ES", "NQ"])
        # Heuristic = ret_co of last row (higher = short)
        assert preds["ES"] == pytest.approx(-0.005)
        assert preds["NQ"] == pytest.approx(-0.01)

    def test_heuristic_missing_root_returns_zero(self) -> None:
        df = pd.DataFrame({"root": ["ES"], "ret_co": [0.01]})
        preds = heuristic_predictions(df, ["ES", "YM"])
        assert preds["YM"] == 0.0


# ---------------------------------------------------------------------------
# Paper guard tests
# ---------------------------------------------------------------------------


class TestPaperGuards:
    def _intents(self, n: int = 3, qty: int = 1) -> List[OrderIntent]:
        roots = ["ESH26", "NQH26", "RTYH26"]
        return [
            OrderIntent(
                asof=date(2026, 2, 14),
                ticker=roots[i % len(roots)],
                quantity=qty,
                side="buy",
                reason="test",
            )
            for i in range(n)
        ]

    def test_all_pass(self) -> None:
        config = PaperGuardConfig(max_orders_per_day=10, max_contracts_per_order=5)
        result = apply_paper_guards(self._intents(3), config)
        assert result.passed is True
        assert len(result.filtered_intents) == 3
        assert len(result.violations) == 0

    def test_crash_risk_blocks_all(self) -> None:
        config = PaperGuardConfig()
        result = apply_paper_guards(self._intents(3), config, regime="CRASH_RISK")
        assert result.passed is False
        assert len(result.filtered_intents) == 0
        assert any("CRASH_RISK" in v for v in result.violations)

    def test_per_order_qty_cap(self) -> None:
        config = PaperGuardConfig(max_contracts_per_order=2)
        intents = self._intents(1, qty=5)
        result = apply_paper_guards(intents, config)
        assert result.passed is False
        assert len(result.filtered_intents) == 0

    def test_per_instrument_cap(self) -> None:
        config = PaperGuardConfig(max_contracts_per_instrument=2)
        intents = [
            OrderIntent(asof=date(2026, 2, 14), ticker="ESH26", quantity=1, side="buy", reason="t"),
            OrderIntent(asof=date(2026, 2, 14), ticker="ESH26", quantity=1, side="sell", reason="t"),
            OrderIntent(asof=date(2026, 2, 14), ticker="ESH26", quantity=1, side="buy", reason="t"),
        ]
        result = apply_paper_guards(intents, config)
        assert result.passed is False
        assert any("max_contracts_per_instrument" in v for v in result.violations)

    def test_flatten_bypass(self) -> None:
        config = PaperGuardConfig(max_orders_per_day=1)
        intents = self._intents(5)
        result = apply_paper_guards(intents, config, is_flatten=True)
        assert result.passed is True
        assert len(result.filtered_intents) == 5

    def test_roll_window_blocks(self) -> None:
        config = PaperGuardConfig(roll_window_block=True)
        result = apply_paper_guards(self._intents(1), config, is_roll_window=True)
        assert result.passed is False
        assert any("Roll window" in v for v in result.violations)

    def test_gross_notional_cap(self) -> None:
        config = PaperGuardConfig(max_gross_notional=100_000.0)
        intents = [
            OrderIntent(asof=date(2026, 2, 14), ticker="ESH26", quantity=5, side="buy", reason="t"),
        ]
        multipliers = {"ES": 50.0}
        prices = {"ES": 5000.0}
        result = apply_paper_guards(
            intents, config,
            multipliers=multipliers,
            reference_prices=prices,
        )
        # 5 * 50 * 5000 = 1,250,000 > 100,000
        assert result.passed is False
        assert any("Gross notional" in v for v in result.violations)

    def test_caution_regime_does_not_block(self) -> None:
        config = PaperGuardConfig()
        result = apply_paper_guards(self._intents(1), config, regime="CAUTION")
        assert result.passed is True


# ---------------------------------------------------------------------------
# Reconciliation tests
# ---------------------------------------------------------------------------


class TestReconciliation:
    def test_clean_reconciliation(self) -> None:
        report = reconcile_day(
            asof=date(2026, 2, 14),
            open_intents=[{"ticker": "ESH26", "quantity": 1, "side": "buy"}],
            close_intents=[{"ticker": "ESH26", "quantity": 1, "side": "sell"}],
            open_orders=None,
            close_orders=None,
            fills=[
                {"ticker": "ESH26", "quantity": 1, "price": 5000.0, "side": "buy", "commission": 2.25},
                {"ticker": "ESH26", "quantity": 1, "price": 5010.0, "side": "sell", "commission": 2.25},
            ],
            positions=[],
        )
        assert report["status"] == "CLEAN"
        assert report["summary"]["open_fill_coverage"] == 1.0
        assert report["summary"]["close_fill_coverage"] == 1.0
        assert report["summary"]["total_commission"] == 4.5

    def test_partial_fill_detected(self) -> None:
        report = reconcile_day(
            asof=date(2026, 2, 14),
            open_intents=[{"ticker": "ESH26", "quantity": 3, "side": "buy"}],
            close_intents=[],
            open_orders=None,
            close_orders=None,
            fills=[
                {"ticker": "ESH26", "quantity": 1, "price": 5000.0, "side": "buy", "commission": 2.25},
            ],
            positions=[],
        )
        assert report["status"] == "ISSUES_FOUND"
        assert report["summary"]["partial_fills"] == 1
        assert report["details"]["partial_fills"][0]["shortfall"] == 2

    def test_residual_position_flagged(self) -> None:
        report = reconcile_day(
            asof=date(2026, 2, 14),
            open_intents=[],
            close_intents=[],
            open_orders=None,
            close_orders=None,
            fills=[],
            positions=[{"ticker": "ESH26", "quantity": 1, "avg_cost": 5000.0}],
        )
        assert report["status"] == "ISSUES_FOUND"
        assert report["summary"]["residual_positions"] == 1

    def test_empty_day_is_clean(self) -> None:
        report = reconcile_day(
            asof=date(2026, 2, 14),
            open_intents=[],
            close_intents=[],
            open_orders=None,
            close_orders=None,
            fills=[],
            positions=[],
        )
        assert report["status"] == "CLEAN"
