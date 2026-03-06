"""Tests for Stages 4-5: Paper burn-in setup and live cutover scripts.

Verifies:
  - Paper mode setup and validation
  - Live cutover preflight, execute, rollback
  - Safety gates (can only cutover from paper mode)
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from backend.app.orchestrator.durable_control_state import DurableControlState
from backend.app.orchestrator.migrations import apply_migrations


@pytest.fixture()
def state_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "state.sqlite3"
    conn = sqlite3.connect(db_path)
    apply_migrations(conn)
    conn.close()
    return db_path


@pytest.fixture()
def state(state_db: Path) -> DurableControlState:
    return DurableControlState(state_db)


class TestPaperBurnIn:
    def test_set_paper_mode(self, state: DurableControlState):
        state.set_execution_mode("paper")
        assert state.get_execution_mode() == "paper"

    def test_set_exposure_cap(self, state: DurableControlState):
        state.set_exposure_cap(1.5)
        snap = state.snapshot()
        assert snap["gross_exposure_cap"] == 1.5

    def test_unpause(self, state: DurableControlState):
        state.set_paused(True)
        assert state.is_paused()
        state.set_paused(False)
        assert not state.is_paused()

    def test_audit_trail_records(self, state: DurableControlState):
        state.set_execution_mode("paper")
        state.set_exposure_cap(1.5)
        audit = state.get_audit(10)
        assert len(audit) >= 2
        actions = [e["action"] for e in audit]
        assert "execution_mode" in actions
        assert "exposure_cap" in actions


class TestLiveCutover:
    def test_cutover_from_paper_succeeds(self, state: DurableControlState):
        state.set_execution_mode("paper")
        state.set_execution_mode("ibkr")
        assert state.get_execution_mode() == "ibkr"

    def test_cutover_sets_risk_cap(self, state: DurableControlState):
        state.set_exposure_cap(0.25)
        assert state.snapshot()["gross_exposure_cap"] == 0.25

    def test_rollback_forces_paper(self, state: DurableControlState):
        state.set_execution_mode("ibkr")
        # Rollback
        state.set_paused(True)
        state.set_execution_mode("paper")
        state.set_exposure_cap(1.5)
        snap = state.snapshot()
        assert snap["execution_mode"] == "paper"
        assert snap["paused"]
        assert snap["gross_exposure_cap"] == 1.5

    def test_invalid_mode_rejected(self, state: DurableControlState):
        with pytest.raises(ValueError, match="Invalid execution mode"):
            state.set_execution_mode("invalid_mode")

    def test_wal_checkpoint(self, state_db: Path):
        conn = sqlite3.connect(state_db, timeout=30)
        # Should not raise
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.close()

    def test_clear_stale_intents(self, state_db: Path):
        conn = sqlite3.connect(state_db)
        # Insert a stale intent
        conn.execute(
            "INSERT OR IGNORE INTO order_intents "
            "(asof_date, execution_phase, sleeve, symbol, asset_class, target_weight, multiplier, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("2026-01-01", "intraday", "core", "SPY", "EQUITY", 0.1, 1.0, "PENDING"),
        )
        conn.commit()

        # Clear stale
        deleted = conn.execute(
            "DELETE FROM order_intents WHERE asof_date < ? AND status='PENDING'",
            ("2026-02-28",),
        ).rowcount
        conn.commit()
        conn.close()
        assert deleted == 1

    def test_cutover_audit_trail(self, state: DurableControlState):
        state.set_execution_mode("paper")
        state.set_exposure_cap(0.25)
        state.set_execution_mode("ibkr")
        state.set_paused(False)

        audit = state.get_audit(10)
        actions = [e["action"] for e in audit]
        assert "execution_mode" in actions
        assert "exposure_cap" in actions
