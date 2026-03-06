"""Test: PaperBroker synthetic slippage penalty.

Validates that the PersistentPaperBroker applies adverse slippage
to every paper fill:
  - BUY fills execute ABOVE the requested price
  - SELL fills execute BELOW the requested price
  - Slippage magnitude matches the formula: half_spread + slippage_ticks * tick
"""
from __future__ import annotations

from pathlib import Path

import pytest

from backend.app.orchestrator.broker_paper import PersistentPaperBroker


@pytest.fixture
def broker(tmp_path: Path) -> PersistentPaperBroker:
    return PersistentPaperBroker(
        state_dir=tmp_path,
        starting_equity=100_000.0,
        slippage_ticks=1,
        spread_bps=2.0,
    )


def _place(broker: PersistentPaperBroker, symbol: str, qty: float, side: str, price: float) -> dict:
    return broker.place_orders({
        "orders": [{"symbol": symbol, "qty": qty, "side": side, "est_price": price}]
    })


class TestSlippagePenalty:
    """BUY fills execute above mid-price; SELL fills execute below."""

    def test_buy_fills_above_requested_price(self, broker: PersistentPaperBroker) -> None:
        result = _place(broker, "AAPL", 10, "BUY", 150.0)
        fill = result["routed"][0]
        assert fill["status"] == "filled"
        assert fill["fill_price"] > 150.0, (
            f"BUY fill should be above 150.0, got {fill['fill_price']}"
        )

    def test_sell_fills_below_requested_price(self, broker: PersistentPaperBroker) -> None:
        # First buy to have a position
        _place(broker, "AAPL", 10, "BUY", 150.0)
        result = _place(broker, "AAPL", 10, "SELL", 155.0)
        fill = result["routed"][0]
        assert fill["status"] == "filled"
        assert fill["fill_price"] < 155.0, (
            f"SELL fill should be below 155.0, got {fill['fill_price']}"
        )

    def test_slippage_magnitude_matches_formula(self, broker: PersistentPaperBroker) -> None:
        """Verify the exact slippage = half_spread + slippage_ticks * tick."""
        mid_price = 100.0
        spread_bps = 2.0
        ticks = 1

        half_spread = mid_price * (spread_bps / 10_000) / 2.0
        tick = half_spread
        expected_adverse = half_spread + ticks * tick
        expected_buy = mid_price + expected_adverse

        result = _place(broker, "TEST", 1, "BUY", mid_price)
        actual = result["routed"][0]["fill_price"]
        assert abs(actual - expected_buy) < 1e-8, (
            f"Expected {expected_buy}, got {actual}"
        )

    def test_zero_slippage_mode(self, tmp_path: Path) -> None:
        """With ticks=0 and spread_bps=0, fills at exact price."""
        broker = PersistentPaperBroker(
            state_dir=tmp_path / "zero",
            slippage_ticks=0,
            spread_bps=0.0,
        )
        result = _place(broker, "SPY", 5, "BUY", 500.0)
        assert result["routed"][0]["fill_price"] == 500.0

    def test_futures_slippage_no_cash_impact(self, broker: PersistentPaperBroker) -> None:
        """Futures orders should receive slippage but NOT affect cash."""
        initial_cash = broker.cash
        result = _place(broker, "ESZ5", 1, "BUY", 5000.0)
        fill = result["routed"][0]
        assert fill["fill_price"] > 5000.0, "Futures BUY should have slippage"
        assert broker.cash == initial_cash, "Futures should not impact cash"
