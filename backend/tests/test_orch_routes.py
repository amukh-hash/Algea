"""Tests for the orchestrator status API routes."""
from __future__ import annotations

import json
import sqlite3
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from backend.app.orchestrator.migrations import apply_migrations


# ── helpers ──────────────────────────────────────────────────────────

def _seed_db(db_path: Path) -> None:
    """Create an orchestrator-state SQLite DB with sample data."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    apply_migrations(conn)

    conn.execute(
        "INSERT INTO runs(run_id, asof_date, session, started_at, ended_at, status, meta_json) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("run-1", "2026-02-18", "open", "2026-02-18T10:00:00", "2026-02-18T10:01:00", "success",
         json.dumps({"dry_run": False, "ran_jobs": ["core_signals"], "failed_jobs": [], "skipped_jobs": ["vrp_signals"]})),
    )
    conn.execute(
        "INSERT INTO runs(run_id, asof_date, session, started_at, ended_at, status, meta_json) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("run-2", "2026-02-18", "close", "2026-02-18T16:00:00", None, "running",
         json.dumps({"dry_run": True})),
    )
    conn.execute(
        """INSERT INTO jobs(run_id, asof_date, session, job_name, status, started_at, ended_at, exit_code, error_summary)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("run-1", "2026-02-18", "open", "core_signals", "success",
         "2026-02-18T10:00:01", "2026-02-18T10:00:30", 0, None),
    )
    conn.execute(
        """INSERT INTO jobs(run_id, asof_date, session, job_name, status, started_at, ended_at, exit_code, error_summary)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("run-1", "2026-02-18", "open", "vrp_signals", "skipped",
         None, None, None, "idempotent_success"),
    )
    conn.commit()
    conn.close()


def _seed_artifacts(artifact_root: Path, day: str = "2026-02-18") -> None:
    """Write sample artifact files (heartbeat, positions, targets, fills)."""
    day_dir = artifact_root / day
    day_dir.mkdir(parents=True, exist_ok=True)

    # heartbeat
    (day_dir / "heartbeat.json").write_text(
        json.dumps({"timestamp": "2026-02-18T10:01:00", "session": "open", "state": "success", "mode": "paper"}),
        encoding="utf-8",
    )

    # positions
    fills_dir = day_dir / "fills"
    fills_dir.mkdir(exist_ok=True)
    (fills_dir / "positions.json").write_text(
        json.dumps({"positions": [{"symbol": "SPY", "qty": 10, "avg_cost": 520.5}]}),
        encoding="utf-8",
    )
    (fills_dir / "fills.json").write_text(
        json.dumps({"fills": [{"symbol": "SPY", "side": "buy", "qty": 10, "price": 520.5}]}),
        encoding="utf-8",
    )

    # targets
    targets_dir = day_dir / "targets"
    targets_dir.mkdir(exist_ok=True)
    (targets_dir / "core_targets.json").write_text(
        json.dumps({"targets": [{"symbol": "AAPL", "target_weight": 0.05, "score": 1.2, "side": "buy"}]}),
        encoding="utf-8",
    )

    # a generic artifact file
    (day_dir / "summary.json").write_text(json.dumps({"ok": True}), encoding="utf-8")


@pytest.fixture()
def test_client(tmp_path: Path):
    """Create a TestClient with mocked DB and artifact paths."""
    db_path = tmp_path / "state" / "state.sqlite3"
    artifact_root = tmp_path / "artifacts"

    _seed_db(db_path)
    _seed_artifacts(artifact_root)

    with (
        patch("backend.app.api.orch_routes._DB_PATH", db_path),
        patch("backend.app.api.orch_routes._ARTIFACT_ROOT", artifact_root),
        patch("backend.app.api.orch_routes._today", return_value="2026-02-18"),
    ):
        from backend.app.api.main import app
        yield TestClient(app, raise_server_exceptions=False)


# ── GET /api/orchestrator/status ─────────────────────────────────────

class TestGetStatus:
    def test_returns_heartbeat_and_last_run(self, test_client):
        resp = test_client.get("/api/orchestrator/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["asof_date"] == "2026-02-18"
        assert data["heartbeat"] is not None
        assert data["heartbeat"]["session"] == "open"
        assert data["heartbeat"]["mode"] == "paper"
        assert data["last_run"] is not None
        assert data["last_run"]["status"] == "running"  # run-2 is latest by started_at

    def test_missing_heartbeat_returns_null(self, test_client, tmp_path):
        # Remove the heartbeat file
        hb = tmp_path / "artifacts" / "2026-02-18" / "heartbeat.json"
        hb.unlink(missing_ok=True)
        resp = test_client.get("/api/orchestrator/status")
        assert resp.status_code == 200
        assert resp.json()["heartbeat"] is None


# ── GET /api/orchestrator/runs ───────────────────────────────────────

class TestListRuns:
    def test_lists_runs(self, test_client):
        resp = test_client.get("/api/orchestrator/runs")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert len(data["items"]) == 2
        # Newest first
        assert data["items"][0]["run_id"] == "run-2"

    def test_limit_param(self, test_client):
        resp = test_client.get("/api/orchestrator/runs?limit=1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1

    def test_meta_json_parsed(self, test_client):
        resp = test_client.get("/api/orchestrator/runs")
        run1 = next(r for r in resp.json()["items"] if r["run_id"] == "run-1")
        assert "meta" in run1
        assert run1["meta"]["ran_jobs"] == ["core_signals"]


# ── GET /api/orchestrator/runs/{run_id}/jobs ─────────────────────────

class TestGetRunJobs:
    def test_returns_jobs_for_run(self, test_client):
        resp = test_client.get("/api/orchestrator/runs/run-1/jobs")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["items"]) == 2
        names = [j["job_name"] for j in data["items"]]
        assert "core_signals" in names
        assert "vrp_signals" in names

    def test_no_jobs_returns_empty(self, test_client):
        resp = test_client.get("/api/orchestrator/runs/nonexistent/jobs")
        assert resp.status_code == 200
        assert resp.json()["items"] == []


# ── GET /api/orchestrator/positions ──────────────────────────────────

class TestGetPositions:
    def test_returns_positions(self, test_client):
        resp = test_client.get("/api/orchestrator/positions")
        assert resp.status_code == 200
        data = resp.json()
        assert data["asof_date"] == "2026-02-18"
        assert len(data["positions"]) == 1
        assert data["positions"][0]["symbol"] == "SPY"

    def test_missing_positions_returns_empty(self, test_client):
        resp = test_client.get("/api/orchestrator/positions?asof=1999-01-01")
        assert resp.status_code == 200
        assert resp.json()["positions"] == []


# ── GET /api/orchestrator/targets ────────────────────────────────────

class TestGetTargets:
    def test_returns_targets_by_sleeve(self, test_client):
        resp = test_client.get("/api/orchestrator/targets")
        assert resp.status_code == 200
        data = resp.json()
        assert "core" in data["sleeves"]
        core = data["sleeves"]["core"]
        assert core["targets"][0]["symbol"] == "AAPL"

    def test_missing_targets_returns_empty(self, test_client):
        resp = test_client.get("/api/orchestrator/targets?asof=1999-01-01")
        assert resp.status_code == 200
        assert resp.json()["sleeves"] == {}


# ── GET /api/orchestrator/fills ──────────────────────────────────────

class TestGetFills:
    def test_returns_fills(self, test_client):
        resp = test_client.get("/api/orchestrator/fills")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["fills"]) == 1

    def test_missing_fills_returns_empty(self, test_client):
        resp = test_client.get("/api/orchestrator/fills?asof=1999-01-01")
        assert resp.status_code == 200
        assert resp.json()["fills"] == []


# ── GET /api/orchestrator/artifacts/{day}/{path} ─────────────────────

class TestGetArtifact:
    def test_serves_json_file(self, test_client):
        resp = test_client.get("/api/orchestrator/artifacts/2026-02-18/summary.json")
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}

    def test_missing_artifact_returns_404(self, test_client):
        resp = test_client.get("/api/orchestrator/artifacts/2026-02-18/nope.json")
        assert resp.status_code == 404
