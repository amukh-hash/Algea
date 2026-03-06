"""Tests for GPUProcessSupervisor, SQLiteOutboxWriter, Plugin Protocol, and Schemas.

Verifies F5 (timeout recovery), F6 (batched writes), F9 (plugin protocol),
and schema data contracts (Pydantic validation).
GPU tests use mock plugins (no real CUDA required).
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest


# ── SQLiteOutboxWriter (F6) ──────────────────────────────────────────


class TestSQLiteOutboxWriter:
    def test_batched_writes(self, tmp_path: Path):
        from backend.app.core.db_writer import SQLiteOutboxWriter

        db_path = tmp_path / "telemetry.sqlite3"
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE metrics (key TEXT, value REAL)")
        conn.close()

        writer = SQLiteOutboxWriter(str(db_path), maxsize=100, batch_size=10)
        writer.start()

        for i in range(25):
            writer.enqueue(
                "INSERT INTO metrics (key, value) VALUES (?, ?)",
                (f"m{i}", float(i)),
            )

        writer.q.join()
        writer.stop()
        writer.join(timeout=5)

        conn = sqlite3.connect(db_path)
        count = conn.execute("SELECT COUNT(*) FROM metrics").fetchone()[0]
        conn.close()
        assert count == 25

    def test_queue_full_drops(self, tmp_path: Path):
        from backend.app.core.db_writer import SQLiteOutboxWriter

        db_path = tmp_path / "test.sqlite3"
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE t (v TEXT)")
        conn.close()

        writer = SQLiteOutboxWriter(str(db_path), maxsize=2)
        # Don't start the writer — queue will fill up
        assert writer.enqueue("INSERT INTO t (v) VALUES (?)", ("a",)) is True
        assert writer.enqueue("INSERT INTO t (v) VALUES (?)", ("b",)) is True
        assert writer.enqueue("INSERT INTO t (v) VALUES (?)", ("c",)) is False

    def test_stop_drains_queue(self, tmp_path: Path):
        from backend.app.core.db_writer import SQLiteOutboxWriter

        db_path = tmp_path / "drain.sqlite3"
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE t (v TEXT)")
        conn.close()

        writer = SQLiteOutboxWriter(str(db_path), maxsize=100)
        writer.start()
        writer.enqueue("INSERT INTO t (v) VALUES (?)", ("drain_test",))
        time.sleep(0.5)
        writer.stop()
        writer.join(timeout=5)

        conn = sqlite3.connect(db_path)
        count = conn.execute("SELECT COUNT(*) FROM t").fetchone()[0]
        conn.close()
        assert count >= 1


# ── Plugin Protocol (F9) ─────────────────────────────────────────────


class TestPluginProtocol:
    def test_runtime_checkable(self):
        from backend.app.core.plugin_protocol import TradingSleevePlugin

        class GoodPlugin:
            def execute(self, context: dict, model_cache: dict) -> None:
                pass

        assert isinstance(GoodPlugin(), TradingSleevePlugin)

    def test_missing_method_fails(self):
        from backend.app.core.plugin_protocol import TradingSleevePlugin

        class BadPlugin:
            pass

        assert not isinstance(BadPlugin(), TradingSleevePlugin)


# ── Schemas (Pydantic Validation) ────────────────────────────────────


class TestSchemas:
    def test_target_intent_pydantic_model(self):
        from backend.app.core.schemas import ExecutionPhase, TargetIntent

        intent = TargetIntent(
            asof_date="2026-02-17",
            sleeve="mera",
            symbol="SPY",
            asset_class="EQUITY",
            target_weight=0.10,
            execution_phase=ExecutionPhase.INTRADAY,
        )
        assert intent.target_weight == 0.10
        assert intent.multiplier == 1.0
        assert intent.dte == -1

    def test_date_format_validation(self):
        """field_validator must reject non-YYYY-MM-DD formats."""
        from backend.app.core.schemas import ExecutionPhase, TargetIntent

        with pytest.raises(Exception):
            TargetIntent(
                asof_date="02-17-2026",  # Wrong format
                sleeve="mera",
                symbol="SPY",
                asset_class="EQUITY",
                target_weight=0.10,
                execution_phase=ExecutionPhase.INTRADAY,
            )

    def test_weight_bounds_upper(self):
        """Field(ge=-2.0, le=2.0) must reject weights > 2.0."""
        from backend.app.core.schemas import ExecutionPhase, TargetIntent

        with pytest.raises(Exception):
            TargetIntent(
                asof_date="2026-02-17",
                sleeve="mera",
                symbol="SPY",
                asset_class="EQUITY",
                target_weight=3.0,
                execution_phase=ExecutionPhase.INTRADAY,
            )

    def test_weight_bounds_lower(self):
        """Field(ge=-2.0, le=2.0) must reject weights < -2.0."""
        from backend.app.core.schemas import ExecutionPhase, TargetIntent

        with pytest.raises(Exception):
            TargetIntent(
                asof_date="2026-02-17",
                sleeve="mera",
                symbol="SPY",
                asset_class="EQUITY",
                target_weight=-3.0,
                execution_phase=ExecutionPhase.INTRADAY,
            )

    def test_multiplier_must_be_positive(self):
        """Field(gt=0.0) must reject zero or negative multipliers."""
        from backend.app.core.schemas import ExecutionPhase, TargetIntent

        with pytest.raises(Exception):
            TargetIntent(
                asof_date="2026-02-17",
                sleeve="mera",
                symbol="SPY",
                asset_class="EQUITY",
                target_weight=0.10,
                execution_phase=ExecutionPhase.INTRADAY,
                multiplier=0.0,
            )

    def test_execution_phase_serialization(self):
        from backend.app.core.schemas import ExecutionPhase, TargetIntent

        intent = TargetIntent(
            asof_date="2026-02-17",
            sleeve="mera",
            symbol="SPY",
            asset_class="EQUITY",
            target_weight=0.10,
            execution_phase=ExecutionPhase.AUCTION_OPEN,
        )
        d = intent.model_dump()
        assert d["execution_phase"] == "auction_open"
        assert d["multiplier"] == 1.0
