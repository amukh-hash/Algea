"""Tests for risk gateway, durable control state, and migration v3.

Verifies the core invariants from the flaw remediation plan:
  - F1: Futures notional exposure breach + atomicity
  - F2: Durable state survives reconnect
  - F3: DTE-0 assignment prevention (timezone-aware)
  - F4: Idempotent UPSERT + delta-qty-based deterministic UUID routing
"""
from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from backend.app.core.risk_gateway import (
    EASTERN_TZ,
    route_phase_orders,
    validate_and_store_intents,
)
from backend.app.core.schemas import ExecutionPhase, TargetIntent
from backend.app.orchestrator.durable_control_state import DurableControlState
from backend.app.orchestrator.migrations import apply_migrations


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture()
def db(tmp_path: Path) -> Path:
    """Create a fresh migrated database."""
    db_path = tmp_path / "state.sqlite3"
    conn = sqlite3.connect(db_path)
    apply_migrations(conn)
    conn.close()
    return db_path


def _intent(
    *,
    symbol: str = "SPY",
    weight: float = 0.10,
    asset_class: str = "EQUITY",
    multiplier: float = 1.0,
    phase: ExecutionPhase = ExecutionPhase.INTRADAY,
    sleeve: str = "mera",
    asof_date: str = "2026-02-17",
    dte: int = -1,
) -> TargetIntent:
    return TargetIntent(
        asof_date=asof_date,
        sleeve=sleeve,
        symbol=symbol,
        asset_class=asset_class,
        target_weight=weight,
        execution_phase=phase,
        multiplier=multiplier,
        dte=dte,
    )


# ── Migration v3 ────────────────────────────────────────────────────


class TestMigrationV3:
    def test_creates_app_control_state(self, db: Path):
        conn = sqlite3.connect(db)
        tables = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "app_control_state" in tables
        conn.close()

    def test_creates_order_intents(self, db: Path):
        conn = sqlite3.connect(db)
        tables = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "order_intents" in tables
        conn.close()

    def test_singleton_row_exists(self, db: Path):
        conn = sqlite3.connect(db)
        row = conn.execute("SELECT * FROM app_control_state WHERE id=1").fetchone()
        assert row is not None
        conn.close()

    def test_migration_is_idempotent(self, db: Path):
        conn = sqlite3.connect(db)
        apply_migrations(conn)
        count = conn.execute("SELECT COUNT(*) FROM app_control_state").fetchone()[0]
        assert count == 1
        conn.close()

    def test_order_intents_pk_is_correct(self, db: Path):
        """PK must be (asof_date, execution_phase, sleeve, symbol)."""
        conn = sqlite3.connect(db)
        info = conn.execute("PRAGMA table_info(order_intents)").fetchall()
        # pk column is index 5 in PRAGMA table_info; nonzero = part of PK
        pk_cols = [row[1] for row in info if row[5] > 0]
        assert set(pk_cols) == {"asof_date", "execution_phase", "sleeve", "symbol"}
        conn.close()


# ── Durable Control State (F2) ──────────────────────────────────────


class TestDurableControlState:
    def test_survives_reconnect(self, db: Path):
        """F2: Setting paused=True, closing, reopening → still paused."""
        state1 = DurableControlState(db)
        state1.set_paused(True)
        del state1

        state2 = DurableControlState(db)
        assert state2.is_paused() is True

    def test_exposure_cap_persists(self, db: Path):
        state = DurableControlState(db)
        state.set_exposure_cap(5.0)
        del state

        state2 = DurableControlState(db)
        assert state2.get_exposure_cap() == 5.0

    def test_execution_mode_validation(self, db: Path):
        state = DurableControlState(db)
        with pytest.raises(ValueError, match="Invalid execution mode"):
            state.set_execution_mode("yolo")

    def test_snapshot_returns_all_fields(self, db: Path):
        state = DurableControlState(db)
        snap = state.snapshot()
        assert "paused" in snap
        assert "gross_exposure_cap" in snap
        assert "execution_mode" in snap


# ── Risk Gateway (F1 / F3 / F4) ─────────────────────────────────────


class TestRiskGateway:
    def test_futures_notional_breach(self, db: Path):
        """F1: /ES with weight=0.5 × multiplier=50 → notional 25.0 > cap 1.5."""
        intent = _intent(
            symbol="ES", weight=0.5, asset_class="FUTURE", multiplier=50.0
        )
        with pytest.raises(RuntimeError, match="RISK BREACH"):
            validate_and_store_intents(db, [intent])

    def test_cash_exposure_within_cap(self, db: Path):
        """Equity weight 0.10 should pass default cap 1.5."""
        intent = _intent(symbol="SPY", weight=0.10, asset_class="EQUITY")
        result = validate_and_store_intents(db, [intent])
        assert result["status"] == "ok"
        assert result["n_stored"] == 1

    def test_dte0_halts_nonzero_weight(self, db: Path):
        """F3: DTE-0 option with nonzero weight after 15:00 EST → FATAL."""
        now_3pm = datetime(2026, 2, 17, 15, 30, tzinfo=EASTERN_TZ)
        intent = _intent(
            symbol="SPY240315C500",
            weight=0.05,
            asset_class="OPTION",
            multiplier=100.0,
            sleeve="vrp",
            dte=0,
        )
        with pytest.raises(RuntimeError, match="DTE-0 Assignment Risk"):
            validate_and_store_intents(db, [intent], now=now_3pm)

    def test_dte0_allows_zero_weight_flatten(self, db: Path):
        """F3: DTE-0 with weight=0 (flatten intent) should succeed."""
        now_3pm = datetime(2026, 2, 17, 15, 30, tzinfo=EASTERN_TZ)
        intent = _intent(
            symbol="SPY240315C500",
            weight=0.0,
            asset_class="OPTION",
            multiplier=100.0,
            sleeve="vrp",
            dte=0,
        )
        result = validate_and_store_intents(db, [intent], now=now_3pm)
        assert result["status"] == "ok"

    def test_upsert_idempotency(self, db: Path):
        """F4: Submit weight=0.10, resubmit weight=0.15 → DB count=1, value=0.15."""
        i1 = _intent(symbol="AAPL", weight=0.10, sleeve="mera")
        i2 = _intent(symbol="AAPL", weight=0.15, sleeve="mera")

        validate_and_store_intents(db, [i1])
        validate_and_store_intents(db, [i2])

        conn = sqlite3.connect(db)
        rows = conn.execute(
            "SELECT target_weight FROM order_intents "
            "WHERE asof_date=? AND sleeve=? AND symbol=?",
            ("2026-02-17", "mera", "AAPL"),
        ).fetchall()
        conn.close()

        assert len(rows) == 1
        assert abs(rows[0][0] - 0.15) < 1e-6

    def test_paused_rejects_intents(self, db: Path):
        """F2: Paused state rejects all intents."""
        state = DurableControlState(db)
        state.set_paused(True)

        intent = _intent(symbol="SPY", weight=0.10)
        with pytest.raises(RuntimeError, match="paused"):
            validate_and_store_intents(db, [intent])

    def test_atomicity_zero_on_breach(self, db: Path):
        """F1: On risk breach, zero intents should be written (atomicity)."""
        intents = [
            _intent(symbol="ES", weight=0.5, asset_class="FUTURE", multiplier=50.0),
            _intent(symbol="SPY", weight=0.10, asset_class="EQUITY"),
        ]
        with pytest.raises(RuntimeError, match="RISK BREACH"):
            validate_and_store_intents(db, intents)

        conn = sqlite3.connect(db)
        count = conn.execute("SELECT COUNT(*) FROM order_intents").fetchone()[0]
        conn.close()
        assert count == 0

    def test_risk_atomicity_invariant_exact(self, db: Path):
        """Plan invariant: weight=0.10 × mult=50 + weight=0.10 × mult=1 → 5.10 > 5.0."""
        # Set cap to 5.0
        state = DurableControlState(db)
        state.set_exposure_cap(5.0)

        intents = [
            _intent(symbol="ES", weight=0.10, asset_class="FUTURE", multiplier=50.0),
            _intent(symbol="SPY", weight=0.10, asset_class="EQUITY", sleeve="mera2"),
        ]
        with pytest.raises(RuntimeError, match="RISK BREACH"):
            validate_and_store_intents(db, intents)

        conn = sqlite3.connect(db)
        count = conn.execute("SELECT COUNT(*) FROM order_intents").fetchone()[0]
        conn.close()
        assert count == 0

    def test_date_validation(self):
        """Pydantic field_validator must reject invalid date formats."""
        with pytest.raises(Exception):
            _intent(asof_date="02-17-2026")  # Wrong format

    def test_weight_bounds(self):
        """Pydantic Field(ge=-2.0, le=2.0) must reject out-of-range weights."""
        with pytest.raises(Exception):
            _intent(weight=3.0)  # Exceeds 2.0


# ── Phase-Aware Routing (F4: delta_qty UUID) ─────────────────────────


class _DeltaQtyBroker:
    """Broker spy that exposes live positions for delta calculation."""

    def __init__(
        self,
        equity: float = 100_000,
        positions: dict[str, int] | None = None,
        prices: dict[str, float] | None = None,
    ):
        self.equity = equity
        self.positions = positions or {}
        self.prices = prices or {"SPY": 500.0, "ES": 5000.0}
        self.placed: list[dict] = []

    def get_account_equity(self) -> float:
        return self.equity

    def get_positions(self) -> dict[str, int]:
        return dict(self.positions)

    def get_price(self, symbol: str) -> float:
        return self.prices.get(symbol, 100.0)

    def place_order(self, *, client_order_id: str, symbol: str, qty: int) -> dict:
        self.placed.append({
            "client_order_id": client_order_id,
            "symbol": symbol,
            "qty": qty,
        })
        return {"status": "ok"}


class TestRoutePhaseOrders:
    def test_routes_with_delta_qty(self, db: Path):
        """Route computes delta from broker positions."""
        intent = _intent(
            symbol="SPY", weight=0.10, phase=ExecutionPhase.AUCTION_OPEN,
        )
        validate_and_store_intents(db, [intent])

        # equity=100K, price=500 → target_qty=20, current=0 → delta=20
        broker = _DeltaQtyBroker(equity=100_000, prices={"SPY": 500.0})
        result = route_phase_orders(db, broker, ExecutionPhase.AUCTION_OPEN, "2026-02-17")
        assert result["routed_count"] == 1
        assert len(broker.placed) == 1
        assert broker.placed[0]["qty"] == 20  # 10_000 / 500

    def test_skips_when_position_matches(self, db: Path):
        """If current_qty == target_qty, delta=0 → SKIPPED."""
        intent = _intent(
            symbol="SPY", weight=0.10, phase=ExecutionPhase.INTRADAY,
        )
        validate_and_store_intents(db, [intent])

        # equity=100K, price=500 → target_qty=20, current=20 → delta=0
        broker = _DeltaQtyBroker(
            equity=100_000,
            prices={"SPY": 500.0},
            positions={"SPY": 20},
        )
        result = route_phase_orders(db, broker, ExecutionPhase.INTRADAY, "2026-02-17")
        assert result["skipped_count"] == 1
        assert result["routed_count"] == 0
        assert len(broker.placed) == 0

    def test_deterministic_uuid_with_delta(self, db: Path):
        """F4: Same delta_qty produces identical UUID on retry."""
        intent = _intent(
            symbol="SPY", weight=0.10, phase=ExecutionPhase.INTRADAY,
        )
        validate_and_store_intents(db, [intent])

        # First routing
        broker1 = _DeltaQtyBroker(equity=100_000, prices={"SPY": 500.0})
        route_phase_orders(db, broker1, ExecutionPhase.INTRADAY, "2026-02-17")
        first_order_id = broker1.placed[0]["client_order_id"]

        # Re-UPSERT resets status to PENDING (correct F4 behavior)
        validate_and_store_intents(db, [intent])

        # Second routing with same positions → same delta → same UUID
        broker2 = _DeltaQtyBroker(equity=100_000, prices={"SPY": 500.0})
        route_phase_orders(db, broker2, ExecutionPhase.INTRADAY, "2026-02-17")
        second_order_id = broker2.placed[0]["client_order_id"]

        assert first_order_id == second_order_id  # Deterministic UUID

    def test_wrong_phase_not_routed(self, db: Path):
        """Intents for AUCTION_OPEN should not route during INTRADAY."""
        intent = _intent(
            symbol="SPY", weight=0.10, phase=ExecutionPhase.AUCTION_OPEN,
        )
        validate_and_store_intents(db, [intent])

        broker = _DeltaQtyBroker()
        result = route_phase_orders(db, broker, ExecutionPhase.INTRADAY, "2026-02-17")
        assert result["routed_count"] == 0
        assert len(broker.placed) == 0
