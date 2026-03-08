"""Tests for Phase 2 snapshot providers.

Validates:
  - BrokerMarketDataProvider produces a valid frozen MarketDataSnapshot
  - BrokerPortfolioStateProvider produces a valid frozen PortfolioStateSnapshot
  - freeze_control_snapshot bridges dict → typed ControlSnapshot
  - MarketDataSnapshot.is_fresh() works correctly
  - MarketDataSnapshot.has_quote() / has_surface() work correctly
  - PortfolioStateSnapshot.position_qty() works correctly
  - Snapshots are frozen (immutable)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

import pytest

from backend.app.contracts.providers import (
    ControlSnapshot,
    MarketDataSnapshot,
    PortfolioStateSnapshot,
    Position,
)
from backend.app.orchestrator.snapshot_providers import (
    BrokerMarketDataProvider,
    BrokerPortfolioStateProvider,
    freeze_control_snapshot,
)


# ── Fake broker ───────────────────────────────────────────────────────


@dataclass
class FakeBroker:
    price_map: dict[str, float] = field(default_factory=dict)
    positions_data: list[dict] = field(default_factory=list)

    def get_quote(self, symbol: str) -> float | None:
        return self.price_map.get(symbol)

    def get_positions(self) -> dict:
        return {"positions": self.positions_data}


# ── MarketDataSnapshot ────────────────────────────────────────────────


class TestBrokerMarketDataProvider:
    def test_freeze_captures_quotes(self):
        broker = FakeBroker(price_map={"SPY": 500.0, "QQQ": 420.0})
        provider = BrokerMarketDataProvider(broker, universe=["SPY", "QQQ", "IWM"])
        snap = provider.freeze_snapshot(date(2026, 3, 7), "INTRADAY")

        assert snap.quotes["SPY"] == 500.0
        assert snap.quotes["QQQ"] == 420.0
        assert "IWM" not in snap.quotes  # price_map returns None
        assert snap.snapshot_id
        assert snap.asof_date == date(2026, 3, 7)

    def test_freeze_returns_frozen_snapshot(self):
        broker = FakeBroker(price_map={"SPY": 500.0})
        provider = BrokerMarketDataProvider(broker, universe=["SPY"])
        snap = provider.freeze_snapshot(date(2026, 3, 7), "INTRADAY")

        with pytest.raises(AttributeError):
            snap.snapshot_id = "mutated"  # type: ignore[misc]

    def test_empty_universe_returns_valid_snapshot(self):
        broker = FakeBroker()
        provider = BrokerMarketDataProvider(broker, universe=[])
        snap = provider.freeze_snapshot(date(2026, 3, 7), "INTRADAY")

        assert snap.quotes == {}
        assert snap.snapshot_id
        assert snap.freshness.get("quotes") is not None


class TestMarketDataSnapshotFreshness:
    def test_is_fresh_within_bounds(self):
        now = datetime(2026, 3, 7, 10, 0, 0, tzinfo=timezone.utc)
        snap = MarketDataSnapshot(
            snapshot_id="test",
            asof_date=date(2026, 3, 7),
            created_at=now,
            quotes={"SPY": 500.0},
            historical_bars={},
            option_surfaces={},
            feature_frames={},
            freshness={"quotes": now - timedelta(minutes=5)},
        )
        assert snap.is_fresh("quotes", timedelta(minutes=10), now=now) is True

    def test_is_fresh_stale(self):
        now = datetime(2026, 3, 7, 10, 0, 0, tzinfo=timezone.utc)
        snap = MarketDataSnapshot(
            snapshot_id="test",
            asof_date=date(2026, 3, 7),
            created_at=now,
            quotes={"SPY": 500.0},
            historical_bars={},
            option_surfaces={},
            feature_frames={},
            freshness={"quotes": now - timedelta(hours=3)},
        )
        assert snap.is_fresh("quotes", timedelta(hours=1), now=now) is False

    def test_is_fresh_missing_entry(self):
        now = datetime(2026, 3, 7, 10, 0, 0, tzinfo=timezone.utc)
        snap = MarketDataSnapshot(
            snapshot_id="test",
            asof_date=date(2026, 3, 7),
            created_at=now,
            quotes={},
            historical_bars={},
            option_surfaces={},
            feature_frames={},
            freshness={},
        )
        assert snap.is_fresh("quotes", timedelta(minutes=10), now=now) is False

    def test_has_quote(self):
        snap = MarketDataSnapshot(
            snapshot_id="test",
            asof_date=date(2026, 3, 7),
            created_at=datetime.now(timezone.utc),
            quotes={"SPY": 500.0},
            historical_bars={},
            option_surfaces={},
            feature_frames={},
            freshness={},
        )
        assert snap.has_quote("SPY") is True
        assert snap.has_quote("QQQ") is False

    def test_has_surface(self):
        snap = MarketDataSnapshot(
            snapshot_id="test",
            asof_date=date(2026, 3, 7),
            created_at=datetime.now(timezone.utc),
            quotes={},
            historical_bars={},
            option_surfaces={"SPY": {"30d": 0.20}},
            feature_frames={},
            freshness={},
        )
        assert snap.has_surface("SPY") is True
        assert snap.has_surface("QQQ") is False


# ── PortfolioStateSnapshot ────────────────────────────────────────────


class TestBrokerPortfolioStateProvider:
    def test_freeze_captures_positions(self):
        broker = FakeBroker(positions_data=[
            {"symbol": "SPY", "quantity": 100, "avg_cost": 490.0, "market_value": 50000.0},
            {"symbol": "QQQ", "position": 50, "avgCost": 410.0, "marketValue": 21000.0},
        ])
        provider = BrokerPortfolioStateProvider(broker)
        snap = provider.freeze_snapshot(date(2026, 3, 7))

        assert "SPY" in snap.positions
        assert snap.positions["SPY"].quantity == 100
        assert snap.positions["SPY"].avg_cost == 490.0
        assert "QQQ" in snap.positions
        assert snap.positions["QQQ"].quantity == 50
        assert snap.snapshot_id

    def test_empty_positions(self):
        broker = FakeBroker()
        provider = BrokerPortfolioStateProvider(broker)
        snap = provider.freeze_snapshot(date(2026, 3, 7))

        assert snap.positions == {}
        assert snap.snapshot_id

    def test_frozen(self):
        broker = FakeBroker()
        provider = BrokerPortfolioStateProvider(broker)
        snap = provider.freeze_snapshot(date(2026, 3, 7))

        with pytest.raises(AttributeError):
            snap.snapshot_id = "mutated"  # type: ignore[misc]

    def test_position_qty(self):
        snap = PortfolioStateSnapshot(
            snapshot_id="test",
            asof_date=date(2026, 3, 7),
            created_at=datetime.now(timezone.utc),
            positions={
                "SPY": Position(symbol="SPY", quantity=100, avg_cost=490.0, market_value=50000.0),
            },
        )
        assert snap.position_qty("SPY") == 100
        assert snap.position_qty("QQQ") == 0.0


# ── ControlSnapshot ──────────────────────────────────────────────────


class TestFreezeControlSnapshot:
    def test_converts_dict_to_typed(self):
        raw = {
            "snapshot_id": "ctrl-abc",
            "paused": False,
            "execution_mode": "paper",
            "blocked_symbols": ["AAPL", "MSFT"],
            "frozen_sleeves": ["vrp"],
            "gross_exposure_cap": 1.5,
            "vol_regime_override": None,
        }
        snap = freeze_control_snapshot(raw, date(2026, 3, 7))

        assert snap.snapshot_id == "ctrl-abc"
        assert snap.paused is False
        assert snap.execution_mode == "paper"
        assert "AAPL" in snap.blocked_symbols
        assert "vrp" in snap.frozen_sleeves
        assert snap.gross_cap == 1.5
        assert snap.asof_date == date(2026, 3, 7)

    def test_frozen(self):
        raw = {
            "snapshot_id": "ctrl-abc",
            "paused": False,
            "execution_mode": "paper",
            "blocked_symbols": [],
            "frozen_sleeves": [],
            "gross_exposure_cap": 1.5,
        }
        snap = freeze_control_snapshot(raw, date(2026, 3, 7))

        with pytest.raises(AttributeError):
            snap.paused = True  # type: ignore[misc]

    def test_defaults_on_missing_fields(self):
        snap = freeze_control_snapshot({}, date(2026, 3, 7))

        assert snap.paused is False
        assert snap.execution_mode == "paper"
        assert snap.gross_cap == 1.5
        assert snap.blocked_symbols == frozenset()
