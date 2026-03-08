"""Snapshot provider contracts for Intent Supremacy architecture.

All per-tick data dependencies are frozen once and reused across sleeves.
Provider protocols define the interface; implementations live in orchestrator/.

Data validity semantics
-----------------------

* **Missing data**: A required field is absent entirely  →  snapshot is
  invalid for any sleeve that needs it  →  sleeve must FAIL.
* **Stale data**: A field is present but older than maximum acceptable
  freshness  →  sleeve must HALT (valid-but-no-trade) or FAIL depending
  on its criticality rules.
* **Fresh data**: A field is present and within freshness bounds  →  OK.

Sleeves query freshness via ``MarketDataSnapshot.is_fresh()`` rather than
implementing their own staleness logic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Literal, Protocol


# ── Position ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Position:
    """A single portfolio position snapshot."""
    symbol: str
    quantity: float
    avg_cost: float
    market_value: float
    asset_class: str = "EQUITY"


# ── Market Data Snapshot ──────────────────────────────────────────────


@dataclass(frozen=True)
class MarketDataSnapshot:
    """Frozen market data captured once per tick.

    All sleeves read from the same snapshot to guarantee consistency.

    Required fields (non-empty for a valid snapshot):

    * ``snapshot_id``  — stable UUID for this freeze
    * ``asof_date``    — the trading date
    * ``created_at``   — wall-clock time the snapshot was captured
    * ``quotes``       — last prices keyed by symbol (may be empty)

    Optional enrichment fields:

    * ``historical_bars``  — keyed by symbol → list of recent close prices
    * ``option_surfaces``  — keyed by symbol → per-tenor IV structure
    * ``feature_frames``   — keyed by sleeve/model → arbitrary feature tensors

    Freshness:

    * ``freshness`` maps field names to their capture timestamps.
    * ``is_fresh()`` checks whether a specific field is within bounds.
    * Missing freshness entry for a field means freshness is unknown  →
      treat as stale.
    """
    snapshot_id: str
    asof_date: date
    created_at: datetime
    quotes: dict[str, float]
    historical_bars: dict[str, list[float]]
    option_surfaces: dict[str, Any]
    feature_frames: dict[str, Any]
    freshness: dict[str, datetime]

    def is_fresh(
        self,
        field_name: str,
        max_age: timedelta,
        *,
        now: datetime | None = None,
    ) -> bool:
        """Check whether ``field_name`` was captured within ``max_age``.

        Returns ``False`` if ``field_name`` has no freshness entry.
        """
        ts = self.freshness.get(field_name)
        if ts is None:
            return False
        reference = now or datetime.now(ts.tzinfo)
        return (reference - ts) <= max_age

    def has_quote(self, symbol: str) -> bool:
        """Check whether a quote exists for ``symbol``."""
        return symbol in self.quotes and self.quotes[symbol] is not None

    def has_surface(self, symbol: str) -> bool:
        """Check whether an option surface exists for ``symbol``."""
        return symbol in self.option_surfaces and self.option_surfaces[symbol] is not None


# ── Portfolio State Snapshot ──────────────────────────────────────────


@dataclass(frozen=True)
class PortfolioStateSnapshot:
    """Frozen portfolio state captured once per tick.

    Required fields:

    * ``snapshot_id``  — stable UUID
    * ``asof_date``    — the trading date
    * ``created_at``   — wall-clock capture time
    * ``positions``    — keyed by symbol → Position (may be empty dict)

    Optional enrichment fields:

    * ``pending_orders``           — orders already in-flight
    * ``sleeve_exposures``         — gross exposure per sleeve (pre-tick)
    * ``turnover_budget_remaining`` — remaining daily turnover budget
    * ``account_equity``           — total account equity for sizing
    """
    snapshot_id: str
    asof_date: date
    created_at: datetime
    positions: dict[str, Position]
    pending_orders: tuple[dict[str, Any], ...] = ()
    sleeve_exposures: dict[str, float] = field(default_factory=dict)
    turnover_budget_remaining: float = 1.0
    account_equity: float = 0.0

    def position_qty(self, symbol: str) -> float:
        """Return current quantity for ``symbol``, or 0.0 if absent."""
        pos = self.positions.get(symbol)
        return pos.quantity if pos else 0.0


# ── Control Snapshot ──────────────────────────────────────────────────


@dataclass(frozen=True)
class ControlSnapshot:
    """Typed control state snapshot with stable ID.

    Required fields:

    * ``snapshot_id``       — UUID that changes on every mutation
    * ``asof_date``         — the trading date
    * ``paused``            — if True, risk gateway rejects all intents
    * ``execution_mode``    — ``noop`` | ``paper`` | ``ibkr``
    * ``blocked_symbols``   — symbols excluded from order routing
    * ``frozen_sleeves``    — sleeves that should not produce intents
    * ``gross_cap``         — total gross exposure limit for risk gateway

    Optional:

    * ``vol_regime_override`` — override for volatility regime detection
    """
    snapshot_id: str
    asof_date: date
    paused: bool
    execution_mode: Literal["noop", "paper", "ibkr"]
    blocked_symbols: frozenset[str]
    frozen_sleeves: frozenset[str]
    gross_cap: float
    vol_regime_override: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Provider Protocols ────────────────────────────────────────────────


class MarketDataProvider(Protocol):
    """Produces a frozen market data snapshot for a given date/session."""
    def freeze_snapshot(self, asof_date: date, session: str) -> MarketDataSnapshot: ...


class PortfolioStateProvider(Protocol):
    """Produces a frozen portfolio state snapshot for a given date."""
    def freeze_snapshot(self, asof_date: date) -> PortfolioStateSnapshot: ...


class Sleeve(Protocol):
    """Canonical sleeve interface.

    Every sleeve must implement ``run()`` and return a ``SleeveDecision``.
    The ``SleeveDecision`` type is imported from ``canonical`` to avoid
    circular imports — the return type at runtime is always
    ``contracts.canonical.SleeveDecision``.
    """
    name: str

    def run(
        self,
        asof_date: date,
        session: str,
        run_id: str,
        control: ControlSnapshot,
        market_data: MarketDataSnapshot,
        portfolio_state: PortfolioStateSnapshot,
    ) -> Any:
        """Return a SleeveDecision."""
        ...
