from __future__ import annotations

import json
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


def test_guardrails_status_shape_and_rows(routes, tmp_path: Path):
    asof = "2026-01-01"
    root = tmp_path / "artifacts"
    day = root / asof
    day.mkdir(parents=True, exist_ok=True)
    risk = {
        "checked_at": "2026-01-01T12:00:00+00:00",
        "metrics": {"ece": 0.03, "mmd": 0.01, "max_drawdown": 0.004},
        "limits": {"ece_max": 0.10, "mmd_max": 0.05, "max_drawdown": 0.02},
        "violations": [],
    }
    (day / "risk_checks.json").write_text(json.dumps(risk), encoding="utf-8")

    with patch.object(routes, "_ARTIFACT_ROOT", root):
        payload = routes.get_guardrails_status()

    assert payload["schema_version"] == "guardrails_status.v1"
    assert isinstance(payload["guardrails"], list)
    assert len(payload["guardrails"]) == 5
    row_ids = {r["id"] for r in payload["guardrails"]}
    assert row_ids == {
        "ece_tracker",
        "mmd_liveguard",
        "max_drawdown",
        "gap_risk_filter",
        "slippage_monitor",
    }


def test_guardrails_status_malformed_payload_degrades_to_unknown(routes, tmp_path: Path):
    asof = "2026-01-01"
    root = tmp_path / "artifacts"
    day = root / asof
    day.mkdir(parents=True, exist_ok=True)
    (day / "risk_checks.json").write_text("{bad", encoding="utf-8")

    with patch.object(routes, "_ARTIFACT_ROOT", root):
        payload = routes.get_guardrails_status()

    assert payload["backend_reachable"] is False
    assert len(payload["guardrails"]) == 5
    assert all(row["status"] == "unknown" for row in payload["guardrails"])
    assert all(row["reason"] == "unwired" for row in payload["guardrails"])
