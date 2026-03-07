"""Tests for control-route provider behavior on the default runtime path."""
from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _reset_control_provider_registry():
    from backend.app.orchestrator import control_state_provider as csp

    csp._PROVIDER_REGISTRY.clear()
    yield
    csp._PROVIDER_REGISTRY.clear()


@pytest.fixture()
def routes(tmp_path: Path):
    db_path = tmp_path / "state" / "state.sqlite3"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    from backend.app.orchestrator.migrations import apply_migrations

    conn = sqlite3.connect(db_path)
    apply_migrations(conn)
    conn.close()

    from backend.app.api import control_routes as r

    with (
        patch.object(r, "_DB_PATH", db_path),
        patch.object(r, "bridge_control_snapshot", lambda *_args, **_kwargs: None),
        patch.object(r, "bridge_control_mutation", lambda *_args, **_kwargs: None),
    ):
        yield r


class TestControlStateProviderPath:
    def test_state_read_comes_from_provider(self, routes):
        state = routes.get_state()
        assert state["schema_version"] == "control_state.v1"
        assert state["execution_mode"] == "paper"

    def test_mutation_writes_cache_and_db(self, routes):
        before = routes.get_state()
        resp = routes.set_execution_mode(routes.ExecutionModeRequest(mode="noop"))
        assert resp["ok"] is True

        after = routes.get_state()
        assert after["execution_mode"] == "noop"
        assert after["snapshot_id"] != before["snapshot_id"]

        conn = sqlite3.connect(routes._DB_PATH)
        row = conn.execute(
            "SELECT execution_mode FROM app_control_state WHERE id=1"
        ).fetchone()
        conn.close()
        assert row is not None
        assert row[0] == "noop"

    def test_db_write_failure_returns_explicit_500_and_rolls_back_cache(self, routes):
        provider = routes._provider()
        before = provider.snapshot(consumer="test")

        with patch.object(provider, "_write_through", side_effect=sqlite3.OperationalError("disk I/O error")):
            with pytest.raises(routes.HTTPException) as exc_info:
                routes.set_pause(routes.PauseRequest(paused=True))

        err = exc_info.value
        assert err.status_code == 500
        assert err.detail["error"] == "control_state_write_through_failed"

        after = provider.snapshot(consumer="test")
        assert after["paused"] == before["paused"]
        assert after["snapshot_id"] == before["snapshot_id"]
