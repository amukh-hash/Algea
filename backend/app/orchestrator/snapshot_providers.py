"""Frozen snapshot providers for Intent Supremacy architecture.

Phase 2 implementations: each provider wraps an existing data source
(broker, yfinance, control_state_provider) and freezes it into a
canonical snapshot once per tick.  All downstream consumers read from
the same immutable snapshot, guaranteeing consistency.

These are behind ``FF_CANONICAL_SLEEVE_OUTPUTS`` — when the flag is off,
the orchestrator does not instantiate or inject snapshots.
"""
from __future__ import annotations

import logging
import uuid
from datetime import date, datetime, timezone
from typing import Any

from backend.app.contracts.providers import (
    ControlSnapshot,
    MarketDataSnapshot,
    PortfolioStateSnapshot,
    Position,
)

logger = logging.getLogger(__name__)


class BrokerMarketDataProvider:
    """Produces a frozen ``MarketDataSnapshot`` from the broker adapter.

    Quotes are pulled live via ``broker.get_quote()`` for a configured
    symbol universe.  Historical bars and option surfaces are stubbed
    until Phase 3 sleeve remediation wires real data paths.
    """

    def __init__(self, broker: Any, universe: list[str] | None = None) -> None:
        self._broker = broker
        self._universe = universe or []

    def freeze_snapshot(
        self,
        asof_date: date,
        session: str,
        *,
        universe: list[str] | None = None,
    ) -> MarketDataSnapshot:
        symbols = universe or self._universe
        now = datetime.now(timezone.utc)
        snapshot_id = str(uuid.uuid4())

        # ── Quotes ──
        quotes: dict[str, float] = {}
        freshness: dict[str, datetime] = {}
        for sym in symbols:
            try:
                price = self._broker.get_quote(sym)
                if price is not None:
                    quotes[sym] = float(price)
                    freshness[f"quote.{sym}"] = now
            except Exception:
                logger.debug("Failed to get quote for %s", sym, exc_info=True)

        freshness["quotes"] = now

        snap = MarketDataSnapshot(
            snapshot_id=snapshot_id,
            asof_date=asof_date,
            created_at=now,
            quotes=quotes,
            historical_bars={},
            option_surfaces={},
            feature_frames={},
            freshness=freshness,
        )
        logger.info(
            "market_data_snapshot_frozen snapshot_id=%s n_quotes=%d session=%s",
            snapshot_id, len(quotes), session,
        )
        return snap


class BrokerPortfolioStateProvider:
    """Produces a frozen ``PortfolioStateSnapshot`` from the broker adapter."""

    def __init__(self, broker: Any) -> None:
        self._broker = broker

    def freeze_snapshot(self, asof_date: date) -> PortfolioStateSnapshot:
        now = datetime.now(timezone.utc)
        snapshot_id = str(uuid.uuid4())

        # ── Positions ──
        raw = self._broker.get_positions()
        positions: dict[str, Position] = {}
        for p in raw.get("positions", []):
            sym = str(p.get("symbol", p.get("conId", "")))
            if not sym:
                continue
            positions[sym] = Position(
                symbol=sym,
                quantity=float(p.get("quantity", p.get("position", 0))),
                avg_cost=float(p.get("avg_cost", p.get("avgCost", 0))),
                market_value=float(p.get("market_value", p.get("marketValue", 0))),
                asset_class=str(p.get("asset_class", "EQUITY")),
            )

        # ── Sleeve exposures ──
        sleeve_exposures: dict[str, float] = {}

        snap = PortfolioStateSnapshot(
            snapshot_id=snapshot_id,
            asof_date=asof_date,
            created_at=now,
            positions=positions,
            pending_orders=(),
            sleeve_exposures=sleeve_exposures,
            turnover_budget_remaining=1.0,
            account_equity=0.0,
        )
        logger.info(
            "portfolio_state_snapshot_frozen snapshot_id=%s n_positions=%d",
            snapshot_id, len(positions),
        )
        return snap


def freeze_control_snapshot(
    raw_snapshot: dict[str, Any],
    asof_date: date,
) -> ControlSnapshot:
    """Convert the legacy dict-based control snapshot to a typed ``ControlSnapshot``.

    This bridges the existing ``ControlStateProvider.snapshot()`` output
    into the canonical frozen format without modifying the provider itself.
    """
    return ControlSnapshot(
        snapshot_id=str(raw_snapshot.get("snapshot_id", "")),
        asof_date=asof_date,
        paused=bool(raw_snapshot.get("paused", False)),
        execution_mode=raw_snapshot.get("execution_mode", "paper"),
        blocked_symbols=frozenset(str(s) for s in raw_snapshot.get("blocked_symbols", [])),
        frozen_sleeves=frozenset(str(s) for s in raw_snapshot.get("frozen_sleeves", [])),
        gross_cap=float(raw_snapshot.get("gross_exposure_cap", 1.5) or 1.5),
        vol_regime_override=raw_snapshot.get("vol_regime_override"),
    )
