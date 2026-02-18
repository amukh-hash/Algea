"""Tests for the /healthz endpoint."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from backend.app.orchestrator.migrations import apply_migrations


@pytest.fixture()
def test_client(tmp_path: Path):
    db_path = tmp_path / "state" / "state.sqlite3"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    apply_migrations(conn)
    conn.execute(
        "INSERT INTO runs(run_id, asof_date, session, started_at, status, meta_json) "
        "VALUES ('r1', '2026-01-01', 'open', '2026-01-01T10:00:00', 'success', '{}')"
    )
    conn.commit()
    conn.close()

    artifact_root = tmp_path / "artifacts"

    with (
        patch("backend.app.api.healthz._DB_PATH", db_path),
        patch("backend.app.api.healthz._ARTIFACT_ROOT", artifact_root),
    ):
        from backend.app.api.main import app
        yield TestClient(app)


class TestHealthz:
    def test_returns_200_when_db_reachable(self, test_client):
        resp = test_client.get("/healthz")
        data = resp.json()
        assert data["checks"]["event_loop"]["ok"] is True
        assert data["checks"]["state_db"]["ok"] is True
        assert data["checks"]["state_db"]["run_count"] == 1

    def test_heartbeat_missing_flagged(self, test_client):
        resp = test_client.get("/healthz")
        data = resp.json()
        assert data["checks"]["heartbeat"]["ok"] is False
        assert "no heartbeat" in data["checks"]["heartbeat"]["error"]

    def test_heartbeat_present_and_fresh(self, test_client, tmp_path):
        from datetime import datetime, timezone
        artifact_root = tmp_path / "artifacts"
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        day_dir = artifact_root / today
        day_dir.mkdir(parents=True, exist_ok=True)
        hb = {"timestamp": datetime.now(timezone.utc).isoformat(), "session": "open", "state": "success"}
        (day_dir / "heartbeat.json").write_text(json.dumps(hb), encoding="utf-8")

        resp = test_client.get("/healthz")
        data = resp.json()
        assert data["checks"]["heartbeat"]["ok"] is True
        assert data["checks"]["heartbeat"]["age_seconds"] < 10

    def test_elapsed_ms_present(self, test_client):
        resp = test_client.get("/healthz")
        assert "elapsed_ms" in resp.json()

    def test_db_unreachable_returns_503(self, tmp_path):
        bad_db = tmp_path / "nonexistent" / "state.sqlite3"
        artifact_root = tmp_path / "artifacts"
        with (
            patch("backend.app.api.healthz._DB_PATH", bad_db),
            patch("backend.app.api.healthz._ARTIFACT_ROOT", artifact_root),
        ):
            from backend.app.api.main import app
            client = TestClient(app)
            resp = client.get("/healthz")
            assert resp.status_code == 503
            assert resp.json()["checks"]["state_db"]["ok"] is False
