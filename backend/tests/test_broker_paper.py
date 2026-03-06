"""Unit tests for the persistent paper broker cash & position accounting.

Tests equity and futures fill handling, cash impact calculations,
NAV invariants, and portfolio-level value correctness.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from backend.app.orchestrator.broker_paper import PersistentPaperBroker, _parse_futures_root


# ── Helpers ──────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_state_dir(tmp_path):
    """Return a fresh temp directory for broker state."""
    d = tmp_path / "paper_account"
    d.mkdir()
    return d


def _make_broker(tmp_state_dir, starting_equity=100_000.0):
    return PersistentPaperBroker(
        state_dir=tmp_state_dir,
        starting_equity=starting_equity,
        slippage_ticks=0,
        spread_bps=0.0,
    )


def _place(broker, symbol, qty, side, est_price):
    """Shorthand: submit a single order."""
    return broker.place_orders({
        "orders": [{
            "symbol": symbol,
            "qty": qty,
            "side": side,
            "est_price": est_price,
        }]
    })


# ── _parse_futures_root ─────────────────────────────────────────────────

class TestParseFuturesRoot:
    def test_rty(self):
        assert _parse_futures_root("RTYM6") == "RTY"

    def test_es(self):
        assert _parse_futures_root("ESZ5") == "ES"

    def test_mes_two_digit(self):
        assert _parse_futures_root("MESZ25") == "MES"

    def test_equity_returns_none(self):
        assert _parse_futures_root("AAPL") is None

    def test_short_ticker_none(self):
        assert _parse_futures_root("TE") is None


# ── Equity fill accounting ──────────────────────────────────────────────

class TestEquityAccounting:
    def test_buy_reduces_cash(self, tmp_state_dir):
        b = _make_broker(tmp_state_dir, 50_000)
        _place(b, "AAPL", 10, "BUY", 150.0)
        assert b.cash == pytest.approx(50_000 - 10 * 150.0)

    def test_sell_short_increases_cash(self, tmp_state_dir):
        b = _make_broker(tmp_state_dir, 50_000)
        _place(b, "TSLA", 5, "SELL", 200.0)
        assert b.cash == pytest.approx(50_000 + 5 * 200.0)

    def test_position_created(self, tmp_state_dir):
        b = _make_broker(tmp_state_dir)
        _place(b, "AAPL", 10, "BUY", 150.0)
        positions = b.get_positions()["positions"]
        assert len(positions) == 1
        assert positions[0]["symbol"] == "AAPL"
        assert positions[0]["quantity"] == 10

    def test_round_trip_preserves_cash(self, tmp_state_dir):
        """Buy then sell same qty should return cash to starting equity."""
        b = _make_broker(tmp_state_dir, 50_000)
        _place(b, "AAPL", 10, "BUY", 150.0)
        _place(b, "AAPL", 10, "SELL", 150.0)
        assert b.cash == pytest.approx(50_000)

    def test_multiple_equity_fills_net_cash(self, tmp_state_dir):
        """Long and short fills of equal notional should roughly cancel."""
        b = _make_broker(tmp_state_dir, 100_000)
        _place(b, "AAPL", 100, "BUY", 37.50)   # -3750
        _place(b, "TSLA", 50, "SELL", 75.00)    # +3750
        assert b.cash == pytest.approx(100_000)


# ── Futures fill accounting ─────────────────────────────────────────────

class TestFuturesAccounting:
    def test_futures_buy_does_not_touch_cash(self, tmp_state_dir):
        """Buying futures should NOT deduct anything from cash."""
        b = _make_broker(tmp_state_dir, 100_000)
        _place(b, "RTYM6", 2, "BUY", 2666.15)
        assert b.cash == pytest.approx(100_000)

    def test_futures_sell_does_not_touch_cash(self, tmp_state_dir):
        """Selling (shorting) futures should NOT credit cash."""
        b = _make_broker(tmp_state_dir, 100_000)
        _place(b, "ESZ5", 1, "SELL", 5000.0)
        assert b.cash == pytest.approx(100_000)

    def test_futures_position_avg_cost_is_price(self, tmp_state_dir):
        """Futures position should store index-point price, not margin."""
        b = _make_broker(tmp_state_dir, 100_000)
        _place(b, "RTYM6", 2, "BUY", 2666.15)
        positions = b.get_positions()["positions"]
        assert len(positions) == 1
        assert positions[0]["avg_cost"] == pytest.approx(2666.15, abs=0.01)

    def test_futures_round_trip_no_cash_change(self, tmp_state_dir):
        """Buy then sell same futures should leave cash unchanged."""
        b = _make_broker(tmp_state_dir, 100_000)
        _place(b, "RTYM6", 2, "BUY", 2666.15)
        _place(b, "RTYM6", 2, "SELL", 2666.15)
        assert b.cash == pytest.approx(100_000)

    def test_futures_flip_no_cash_change(self, tmp_state_dir):
        """BUY 2, SELL 3 (flip to short 1) should leave cash unchanged."""
        b = _make_broker(tmp_state_dir, 100_000)
        _place(b, "RTYM6", 2, "BUY", 2666.15)
        _place(b, "RTYM6", 3, "SELL", 2666.15)
        assert b.cash == pytest.approx(100_000)
        # Realized P&L should be 0 (same price)
        assert b.realized_pnl == pytest.approx(0.0)

    def test_futures_realized_pnl_uses_multiplier(self, tmp_state_dir):
        """Closing a futures position should realize P&L with multiplier."""
        b = _make_broker(tmp_state_dir, 100_000)
        _place(b, "RTYM6", 2, "BUY", 2666.15)
        _place(b, "RTYM6", 2, "SELL", 2676.15)  # +10 points
        # Expected: (2676.15 - 2666.15) * 2 * 50 = 1000
        assert b.realized_pnl == pytest.approx(1000.0)
        assert b.cash == pytest.approx(100_000)  # cash unchanged (P&L in realized_pnl)


# ── NAV invariant tests ─────────────────────────────────────────────────

class TestNAVInvariant:
    """NAV = starting_equity + realized_pnl + unrealized_pnl.
    With no price changes, NAV must always equal starting_equity."""

    def test_equity_only_nav(self, tmp_state_dir):
        b = _make_broker(tmp_state_dir, 100_000)
        _place(b, "AAPL", 100, "BUY", 150.0)
        _place(b, "TSLA", 50, "SELL", 200.0)
        # No price change => unrealized = 0, realized = 0
        nav = b.starting_equity + b.realized_pnl  # + 0 unrealized
        assert nav == pytest.approx(100_000)

    def test_futures_only_nav(self, tmp_state_dir):
        b = _make_broker(tmp_state_dir, 100_000)
        _place(b, "RTYM6", 2, "BUY", 2666.15)
        _place(b, "ESZ5", 1, "SELL", 5000.0)
        nav = b.starting_equity + b.realized_pnl
        assert nav == pytest.approx(100_000)

    def test_mixed_nav(self, tmp_state_dir):
        b = _make_broker(tmp_state_dir, 100_000)
        _place(b, "RTYM6", 2, "BUY", 2666.15)
        _place(b, "AAPL", 100, "BUY", 150.0)
        _place(b, "TSLA", 50, "SELL", 200.0)
        nav = b.starting_equity + b.realized_pnl
        assert nav == pytest.approx(100_000)

    def test_nav_after_futures_close_with_profit(self, tmp_state_dir):
        b = _make_broker(tmp_state_dir, 100_000)
        _place(b, "RTYM6", 2, "BUY", 2666.15)
        _place(b, "RTYM6", 2, "SELL", 2676.15)  # +10 pts, +1000
        nav = b.starting_equity + b.realized_pnl
        assert nav == pytest.approx(101_000)


# ── Full replay test ────────────────────────────────────────────────────

class TestFullReplay:
    def test_21_fills_correct_cash(self, tmp_state_dir):
        """Replay the actual 21 fills and verify final cash = 99964.87."""
        b = _make_broker(tmp_state_dir, 100_000)
        fills = [
            ("RTYM6", 2, 2666.15, "BUY"),
            ("ASTS", 44, 85.89, "BUY"),
            ("CCOI", 201, 18.7, "SELL"),
            ("DAVA", 805, 4.66, "BUY"),
            ("DUOL", 34, 110.58, "SELL"),
            ("IBTA", 182, 20.66, "SELL"),
            ("LAKE", 395, 9.5, "SELL"),
            ("LRN", 45, 82.7, "SELL"),
            ("LW", 80, 46.68, "SELL"),
            ("MARA", 432, 8.675, "SELL"),
            ("MBLY", 430, 8.72, "SELL"),
            ("METC", 207, 18.1, "BUY"),
            ("NEXT", 706, 5.31, "SELL"),
            ("OPEN", 752, 4.99, "BUY"),
            ("RCAT", 290, 12.925, "BUY"),
            ("RGTI", 212, 17.665, "BUY"),
            ("SGML", 239, 15.68, "BUY"),
            ("SRPT", 196, 19.14, "BUY"),
            ("TE", 500, 7.5, "BUY"),
            ("TTD", 149, 25.155, "SELL"),
            ("WOLF", 184, 20.35, "BUY"),
        ]
        for sym, qty, price, side in fills:
            _place(b, sym, qty, side, price)

        # Cash = 100K - net equity deployed ($35.13)
        assert b.cash == pytest.approx(99_964.87, abs=0.01)
        positions = b.get_positions()["positions"]
        assert len(positions) == 21
        # NAV = starting + realized (0) = 100K
        assert b.starting_equity + b.realized_pnl == pytest.approx(100_000)


# ── Persistence test ────────────────────────────────────────────────────

class TestPersistence:
    def test_state_survives_reload(self, tmp_state_dir):
        b = _make_broker(tmp_state_dir, 100_000)
        _place(b, "AAPL", 10, "BUY", 150.0)
        cash_before = b.cash

        # Create new broker from same state dir (simulates restart)
        b2 = PersistentPaperBroker(state_dir=tmp_state_dir)
        assert b2.cash == pytest.approx(cash_before)
        assert len(b2.get_positions()["positions"]) == 1
