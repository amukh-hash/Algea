"""Tests for Stage 2: DAG loader, intent aggregation, and core handler refactoring.

Verifies:
  - YAML DAG topology → Job objects (dag_loader)
  - Intent aggregation barrier (intent_aggregator)
  - Core handler emits TargetIntent JSON (no broker calls)
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

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


@pytest.fixture()
def dag_yaml(tmp_path: Path) -> Path:
    """Create a minimal dag_topology.yaml for testing."""
    yaml_path = tmp_path / "dag_topology.yaml"
    yaml_path.write_text(
        """
jobs:
  - id: test_mera
    plugin_path: backend.app.sleeves.mera_plugin
    target_gpu: 0
    timeout_s: 60
    sessions: [premarket, intraday]
    deps: []
    modes: [paper, live, noop]

  - id: test_kronos
    plugin_path: backend.app.sleeves.kronos_plugin
    target_gpu: 1
    timeout_s: 120
    optimize_ada: true
    sessions: [premarket]
    deps: [test_mera]
    modes: [paper, live]
""",
        encoding="utf-8",
    )
    return yaml_path


# ── DAG Loader ───────────────────────────────────────────────────────


class TestDagLoader:
    def test_loads_jobs_from_yaml(self, dag_yaml: Path):
        from backend.app.orchestrator.dag_loader import load_yaml_jobs

        jobs = load_yaml_jobs(dag_yaml)
        assert len(jobs) == 2
        names = {j.name for j in jobs}
        assert "test_mera" in names
        assert "test_kronos" in names

    def test_job_sessions_parsed(self, dag_yaml: Path):
        from backend.app.orchestrator.calendar import Session
        from backend.app.orchestrator.dag_loader import load_yaml_jobs

        jobs = load_yaml_jobs(dag_yaml)
        mera = next(j for j in jobs if j.name == "test_mera")
        assert Session.PREMARKET in mera.sessions
        assert Session.INTRADAY in mera.sessions

    def test_job_deps_preserved(self, dag_yaml: Path):
        from backend.app.orchestrator.dag_loader import load_yaml_jobs

        jobs = load_yaml_jobs(dag_yaml)
        kronos = next(j for j in jobs if j.name == "test_kronos")
        assert "test_mera" in kronos.deps

    def test_job_modes_parsed(self, dag_yaml: Path):
        from backend.app.orchestrator.dag_loader import load_yaml_jobs

        jobs = load_yaml_jobs(dag_yaml)
        kronos = next(j for j in jobs if j.name == "test_kronos")
        assert "paper" in kronos.mode_allow
        assert "live" in kronos.mode_allow
        assert "noop" not in kronos.mode_allow

    def test_missing_yaml_returns_empty(self, tmp_path: Path):
        from backend.app.orchestrator.dag_loader import load_yaml_jobs

        jobs = load_yaml_jobs(tmp_path / "nonexistent.yaml")
        assert jobs == []

    def test_handler_is_callable(self, dag_yaml: Path):
        from backend.app.orchestrator.dag_loader import load_yaml_jobs

        jobs = load_yaml_jobs(dag_yaml)
        for job in jobs:
            assert callable(job.handler)


# ── Intent Aggregator ────────────────────────────────────────────────


class TestIntentAggregator:
    def test_collects_valid_intents(self, tmp_path: Path, db: Path):
        from backend.app.orchestrator.intent_aggregator import (
            collect_and_validate_intents,
        )

        # Write a test intent file
        intents_dir = tmp_path / "intents"
        intents_dir.mkdir()
        (intents_dir / "core_intents.json").write_text(
            json.dumps([
                {
                    "asof_date": "2026-02-17",
                    "sleeve": "core",
                    "symbol": "SPY",
                    "asset_class": "EQUITY",
                    "target_weight": 0.10,
                    "execution_phase": "intraday",
                    "multiplier": 1.0,
                    "dte": -1,
                }
            ]),
            encoding="utf-8",
        )

        result = collect_and_validate_intents(tmp_path, db, "2026-02-17")
        assert result["status"] == "ok"
        assert result["n_collected"] == 1
        assert result["n_validated"] == 1

    def test_rejects_risk_breach(self, tmp_path: Path, db: Path):
        from backend.app.orchestrator.intent_aggregator import (
            collect_and_validate_intents,
        )

        intents_dir = tmp_path / "intents"
        intents_dir.mkdir()
        (intents_dir / "core_intents.json").write_text(
            json.dumps([
                {
                    "asof_date": "2026-02-17",
                    "sleeve": "core",
                    "symbol": "ES",
                    "asset_class": "FUTURE",
                    "target_weight": 0.5,
                    "execution_phase": "auction_open",
                    "multiplier": 50.0,
                    "dte": -1,
                }
            ]),
            encoding="utf-8",
        )

        result = collect_and_validate_intents(tmp_path, db, "2026-02-17")
        assert result["status"] == "rejected"
        assert "RISK BREACH" in result["error"]

    def test_no_intents_returns_no_intents(self, tmp_path: Path, db: Path):
        from backend.app.orchestrator.intent_aggregator import (
            collect_and_validate_intents,
        )

        result = collect_and_validate_intents(tmp_path, db, "2026-02-17")
        assert result["status"] == "no_intents"
        assert result["n_collected"] == 0

    def test_injects_asof_date_when_missing(self, tmp_path: Path, db: Path):
        from backend.app.orchestrator.intent_aggregator import (
            collect_and_validate_intents,
        )

        intents_dir = tmp_path / "intents"
        intents_dir.mkdir()
        # Intent without asof_date — should be injected
        (intents_dir / "mera_intents.json").write_text(
            json.dumps([
                {
                    "sleeve": "mera",
                    "symbol": "AAPL",
                    "asset_class": "EQUITY",
                    "target_weight": 0.05,
                    "execution_phase": "intraday",
                }
            ]),
            encoding="utf-8",
        )

        result = collect_and_validate_intents(tmp_path, db, "2026-02-17")
        assert result["status"] == "ok"
        assert result["n_collected"] == 1

    def test_multiple_sleeve_intents(self, tmp_path: Path, db: Path):
        from backend.app.orchestrator.intent_aggregator import (
            collect_and_validate_intents,
        )

        intents_dir = tmp_path / "intents"
        intents_dir.mkdir()
        (intents_dir / "mera_intents.json").write_text(
            json.dumps([
                {
                    "asof_date": "2026-02-17",
                    "sleeve": "mera",
                    "symbol": "AAPL",
                    "asset_class": "EQUITY",
                    "target_weight": 0.05,
                    "execution_phase": "intraday",
                }
            ]),
            encoding="utf-8",
        )
        (intents_dir / "vrp_intents.json").write_text(
            json.dumps([
                {
                    "asof_date": "2026-02-17",
                    "sleeve": "vrp",
                    "symbol": "SPY",
                    "asset_class": "EQUITY",
                    "target_weight": 0.03,
                    "execution_phase": "intraday",
                }
            ]),
            encoding="utf-8",
        )

        result = collect_and_validate_intents(tmp_path, db, "2026-02-17")
        assert result["status"] == "ok"
        assert result["n_collected"] == 2
        assert result["n_validated"] == 2
