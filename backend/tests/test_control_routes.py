"""Tests for the control API routes."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def _reset_control_state():
    """Reset control state singleton between tests."""
    from backend.app.orchestrator.control_state import control_state
    control_state.paused = False
    control_state.vol_regime_override = None
    control_state.blocked_symbols = set()
    control_state.frozen_sleeves = set()
    control_state.gross_exposure_cap = None
    control_state.execution_mode = "paper"
    control_state.audit_log = []
    yield


@pytest.fixture()
def client(tmp_path: Path):
    db_path = tmp_path / "state" / "state.sqlite3"
    artifact_root = tmp_path / "artifacts"

    db_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)

    # Seed a minimal DB
    import sqlite3
    from backend.app.orchestrator.migrations import apply_migrations
    conn = sqlite3.connect(db_path)
    apply_migrations(conn)
    conn.close()

    with (
        patch("backend.app.api.orch_routes._DB_PATH", db_path),
        patch("backend.app.api.orch_routes._ARTIFACT_ROOT", artifact_root),
        patch("backend.app.api.control_routes._DB_PATH", db_path),
        patch("backend.app.api.control_routes._ARTIFACT_ROOT", artifact_root),
    ):
        from backend.app.api.main import app
        yield TestClient(app, raise_server_exceptions=False)


# ── GET /api/control/state ──────────────────────────────────────────

class TestGetState:
    def test_returns_default_state(self, client):
        resp = client.get("/api/control/state")
        assert resp.status_code == 200
        data = resp.json()
        assert data["paused"] is False
        assert data["execution_mode"] == "paper"
        assert data["vol_regime_override"] is None
        assert data["blocked_symbols"] == []
        assert data["frozen_sleeves"] == []
        assert data["gross_exposure_cap"] is None


# ── PUT /api/control/pause ──────────────────────────────────────────

class TestPause:
    def test_pause_and_resume(self, client):
        resp = client.put("/api/control/pause", json={"paused": True})
        assert resp.status_code == 200
        assert resp.json()["paused"] is True

        state = client.get("/api/control/state").json()
        assert state["paused"] is True

        resp = client.put("/api/control/resume")
        assert resp.status_code == 200
        assert resp.json()["paused"] is False

        state = client.get("/api/control/state").json()
        assert state["paused"] is False


# ── PUT /api/control/vol-regime ─────────────────────────────────────

class TestVolRegime:
    def test_set_crash_risk(self, client):
        resp = client.put("/api/control/vol-regime", json={"regime": "CRASH_RISK"})
        assert resp.status_code == 200

        state = client.get("/api/control/state").json()
        assert state["vol_regime_override"] == "CRASH_RISK"

    def test_set_normal_clears(self, client):
        client.put("/api/control/vol-regime", json={"regime": "CRASH_RISK"})
        resp = client.put("/api/control/vol-regime", json={"regime": "NORMAL"})
        assert resp.status_code == 200

        state = client.get("/api/control/state").json()
        assert state["vol_regime_override"] is None

    def test_invalid_regime_rejected(self, client):
        resp = client.put("/api/control/vol-regime", json={"regime": "BOGUS"})
        assert resp.status_code == 400


# ── PUT /api/control/blocked-symbols ────────────────────────────────

class TestBlockedSymbols:
    def test_set_and_get(self, client):
        resp = client.put("/api/control/blocked-symbols", json={"symbols": ["SPY", "aapl"]})
        assert resp.status_code == 200

        state = client.get("/api/control/state").json()
        assert set(state["blocked_symbols"]) == {"SPY", "AAPL"}  # uppercased


# ── PUT /api/control/frozen-sleeves ─────────────────────────────────

class TestFrozenSleeves:
    def test_set_valid_sleeves(self, client):
        resp = client.put("/api/control/frozen-sleeves", json={"sleeves": ["core", "vrp"]})
        assert resp.status_code == 200

        state = client.get("/api/control/state").json()
        assert set(state["frozen_sleeves"]) == {"core", "vrp"}

    def test_reject_unknown_sleeve(self, client):
        resp = client.put("/api/control/frozen-sleeves", json={"sleeves": ["bogus"]})
        assert resp.status_code == 400


# ── PUT /api/control/exposure-cap ───────────────────────────────────

class TestExposureCap:
    def test_set_and_clear(self, client):
        resp = client.put("/api/control/exposure-cap", json={"cap": 1.5})
        assert resp.status_code == 200

        state = client.get("/api/control/state").json()
        assert state["gross_exposure_cap"] == 1.5

        resp = client.put("/api/control/exposure-cap", json={"cap": None})
        assert resp.status_code == 200

        state = client.get("/api/control/state").json()
        assert state["gross_exposure_cap"] is None

    def test_reject_negative_cap(self, client):
        resp = client.put("/api/control/exposure-cap", json={"cap": -0.5})
        assert resp.status_code == 400


# ── PUT /api/control/execution-mode ─────────────────────────────────

class TestExecutionMode:
    def test_valid_modes(self, client):
        for mode in ["noop", "paper", "ibkr"]:
            resp = client.put("/api/control/execution-mode", json={"mode": mode})
            assert resp.status_code == 200
            state = client.get("/api/control/state").json()
            assert state["execution_mode"] == mode

    def test_invalid_mode_rejected(self, client):
        resp = client.put("/api/control/execution-mode", json={"mode": "invalid"})
        assert resp.status_code == 400


# ── POST /api/control/manual-order ──────────────────────────────────

class TestManualOrder:
    def test_submit_order(self, client):
        resp = client.post("/api/control/manual-order", json={
            "symbol": "SPY",
            "qty": 10,
            "side": "buy",
            "order_type": "MKT",
        })
        assert resp.status_code == 200
        assert resp.json()["order"]["symbol"] == "SPY"

    def test_blocked_symbol_rejected(self, client):
        client.put("/api/control/blocked-symbols", json={"symbols": ["SPY"]})
        resp = client.post("/api/control/manual-order", json={
            "symbol": "SPY",
            "qty": 10,
            "side": "buy",
            "order_type": "MKT",
        })
        assert resp.status_code == 409

    def test_ibkr_mode_rejected(self, client):
        client.put("/api/control/execution-mode", json={"mode": "ibkr"})
        resp = client.post("/api/control/manual-order", json={
            "symbol": "SPY",
            "qty": 10,
            "side": "buy",
            "order_type": "MKT",
        })
        assert resp.status_code == 403


# ── POST /api/control/flatten ───────────────────────────────────────

class TestFlatten:
    def test_flatten_all(self, client):
        resp = client.post("/api/control/flatten", json={})
        assert resp.status_code == 200
        assert resp.json()["action"] == "flatten_all"

    def test_flatten_sleeve(self, client):
        resp = client.post("/api/control/flatten", json={"sleeve": "core"})
        assert resp.status_code == 200
        assert resp.json()["action"] == "flatten_core"

    def test_flatten_rejected_in_ibkr(self, client):
        client.put("/api/control/execution-mode", json={"mode": "ibkr"})
        resp = client.post("/api/control/flatten", json={})
        assert resp.status_code == 403


# ── GET /api/control/audit ──────────────────────────────────────────

class TestAudit:
    def test_audit_populates(self, client):
        client.put("/api/control/pause", json={"paused": True})
        client.put("/api/control/resume")

        resp = client.get("/api/control/audit")
        assert resp.status_code == 200
        items = resp.json()["items"]
        assert len(items) >= 2  # pause + resume
        assert items[0]["action"] == "resume"


# ── GET /api/control/job-graph ──────────────────────────────────────

class TestJobGraph:
    def test_returns_jobs(self, client):
        resp = client.get("/api/control/job-graph")
        assert resp.status_code == 200
        jobs = resp.json()["jobs"]
        assert len(jobs) > 0
        job_names = [j["name"] for j in jobs]
        assert "data_refresh_intraday" in job_names
        assert "risk_checks_global" in job_names

    def test_job_has_deps(self, client):
        jobs = client.get("/api/control/job-graph").json()["jobs"]
        risk = next(j for j in jobs if j["name"] == "risk_checks_global")
        assert len(risk["deps"]) > 0


# ── GET /api/control/calendar ───────────────────────────────────────

class TestCalendar:
    def test_returns_session_info(self, client):
        resp = client.get("/api/control/calendar")
        assert resp.status_code == 200
        data = resp.json()
        assert "current_session" in data
        assert "session_windows" in data
        assert len(data["session_windows"]) > 0


# ── GET /api/control/config ─────────────────────────────────────────

class TestConfig:
    def test_returns_config(self, client):
        resp = client.get("/api/control/config")
        assert resp.status_code == 200
        data = resp.json()
        assert "mode" in data
        assert "account_equity" in data
        assert "max_orders" in data


# ── GET /api/control/broker-status ──────────────────────────────────

class TestBrokerStatus:
    def test_returns_status(self, client):
        resp = client.get("/api/control/broker-status")
        assert resp.status_code == 200
        data = resp.json()
        assert "connected" in data
        assert "mode" in data
