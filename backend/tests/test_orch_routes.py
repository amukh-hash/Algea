from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from backend.app.orchestrator.migrations import apply_migrations


def _seed_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    apply_migrations(conn)
    conn.execute(
        "INSERT INTO runs(run_id, asof_date, session, started_at, ended_at, status, meta_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("run-1", "2026-02-18", "open", "2026-02-18T10:00:00", "2026-02-18T10:01:00", "success", json.dumps({"dry_run": False})),
    )
    conn.execute(
        """INSERT INTO jobs(run_id, asof_date, session, job_name, status, started_at, ended_at, exit_code, error_summary, last_success_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            "run-1",
            "2026-02-18",
            "open",
            "risk_checks_global",
            "success",
            "2026-02-18T10:00:02",
            "2026-02-18T10:00:10",
            0,
            None,
            "2026-02-18T10:00:10",
        ),
    )
    conn.execute(
        """INSERT INTO jobs(run_id, asof_date, session, job_name, status, started_at, ended_at, exit_code, error_summary)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            "run-1",
            "2026-02-18",
            "open",
            "order_build_and_route",
            "failed",
            "2026-02-18T10:00:11",
            "2026-02-18T10:00:15",
            1,
            "missing quotes",
        ),
    )
    conn.commit()
    conn.close()


def _seed_day(root: Path, day: str, *, legacy_risk: bool = False) -> None:
    day_dir = root / day
    (day_dir / "fills").mkdir(parents=True, exist_ok=True)
    (day_dir / "targets").mkdir(parents=True, exist_ok=True)
    (day_dir / "reports").mkdir(parents=True, exist_ok=True)

    (day_dir / "heartbeat.json").write_text(
        json.dumps({"timestamp": f"{day}T10:01:00", "session": "open", "state": "success", "mode": "paper"}),
        encoding="utf-8",
    )
    (day_dir / "fills" / "positions.json").write_text(
        json.dumps({"positions": [{"symbol": "SPY", "qty": 3, "avg_cost": 500.0}]}),
        encoding="utf-8",
    )
    (day_dir / "fills" / "fills.json").write_text(
        json.dumps({"fills": [{"symbol": "SPY", "side": "buy", "qty": 3, "price": 500.0}]}),
        encoding="utf-8",
    )
    (day_dir / "targets" / "core_targets.json").write_text(
        json.dumps({"targets": [{"symbol": "AAPL", "target_weight": 0.05}]}),
        encoding="utf-8",
    )
    (day_dir / "instance.json").write_text(
        json.dumps({"sleeves": {"core": {"status": "live"}}, "asof_date": day}),
        encoding="utf-8",
    )
    (day_dir / "summary.json").write_text(json.dumps({"ok": True}), encoding="utf-8")

    if legacy_risk:
        risk_payload = {"status": "failed", "missing_sleeves": ["vrp"], "nan_or_inf": True}
        (day_dir / "risk_checks.json").write_text(json.dumps(risk_payload), encoding="utf-8")
    else:
        risk_payload = {
            "status": "ok",
            "checked_at": f"{day}T10:01:10",
            "asof_date": day,
            "session": "open",
            "reason": None,
            "missing_sleeves": [],
            "inputs": {"target_paths": {}},
            "metrics": {"gross_exposure": 0.1, "per_sleeve": {"core": {"gross": 0.1}}},
            "limits": {"max_gross": 1.5},
            "violations": [],
            "per_sleeve": {"core": {"gross": 0.1}},
        }
        (day_dir / "reports" / "risk_checks.json").write_text(json.dumps(risk_payload), encoding="utf-8")

    (day_dir / "reports" / "portfolio_equity_curve.csv").write_text(
        "date,cum_net_unscaled,cum_net_volscaled,drawdown,rolling_vol,rolling_sharpe,turnover,cost\n"
        f"{day},0.01,0.008,-0.001,0.12,1.1,0.03,0.0005\n"
        f"{day},N/A,0.01,N/A,inf,nan,N/A,none\n",
        encoding="utf-8",
    )


@pytest.fixture()
def test_client(tmp_path: Path):
    db_path = tmp_path / "state" / "state.sqlite3"
    artifact_root = tmp_path / "artifacts"
    _seed_db(db_path)
    _seed_day(artifact_root, "2026-02-17", legacy_risk=True)
    _seed_day(artifact_root, "2026-02-18", legacy_risk=False)

    with (
        patch("backend.app.api.orch_routes._DB_PATH", db_path),
        patch("backend.app.api.orch_routes._ARTIFACT_ROOT", artifact_root),
        patch("backend.app.api.orch_routes._today", return_value="2026-02-18"),
    ):
        from backend.app.api.main import app

        yield TestClient(app, raise_server_exceptions=False)


def test_instance_success_and_missing_404(test_client):
    ok = test_client.get("/api/orchestrator/instance", params={"asof": "2026-02-18"})
    assert ok.status_code == 200
    assert ok.json()["instance"]["sleeves"]["core"]["status"] == "live"

    missing = test_client.get("/api/orchestrator/instance", params={"asof": "1999-01-01"})
    assert missing.status_code == 404
    detail = missing.json()["detail"]
    assert detail["error_code"] == "instance_not_found"
    assert "expected_paths" in detail and "found_paths" in detail


def test_risk_canonical_and_legacy_and_missing(test_client):
    canonical = test_client.get("/api/orchestrator/risk-checks", params={"asof": "2026-02-18"})
    assert canonical.status_code == 200
    assert canonical.json()["risk_checks"]["schema_version"] == "canonical"

    legacy = test_client.get("/api/orchestrator/risk-checks", params={"asof": "2026-02-17"})
    assert legacy.status_code == 200
    assert legacy.json()["risk_checks"]["schema_version"] == "legacy_normalized"

    missing = test_client.get("/api/orchestrator/risk-checks", params={"asof": "1999-01-01"})
    assert missing.status_code == 404
    assert missing.json()["detail"]["error_code"] == "risk_checks_not_found"


def test_equity_series_columns_and_missing(test_client):
    ok = test_client.get("/api/orchestrator/equity-series", params={"asof": "2026-02-18"})
    assert ok.status_code == 200
    row = ok.json()["series"][0]
    assert "t" in row
    assert "cum_net_unscaled" in row
    assert "cum_net_volscaled" in row

    missing = test_client.get("/api/orchestrator/equity-series", params={"asof": "1999-01-01"})
    assert missing.status_code == 404
    assert missing.json()["detail"]["error_code"] == "equity_series_not_found"


def test_jobs_registry_and_history_keys(test_client):
    registry = test_client.get("/api/orchestrator/jobs")
    assert registry.status_code == 200
    item = registry.json()["items"][0]
    for k in ["name", "min_interval_s", "last_success_at", "last_status", "last_error", "last_duration_s", "next_eligible_at"]:
        assert k in item

    history = test_client.get("/api/orchestrator/jobs/history", params={"limit": 10, "asof": "2026-02-18"})
    assert history.status_code == 200
    h = history.json()["items"][0]
    for k in ["name", "last_status", "last_error", "last_duration_s", "next_eligible_at"]:
        assert k in h


def test_artifacts_listing_order_and_download_url(test_client):
    resp = test_client.get("/api/orchestrator/artifacts", params={"asof": "2026-02-18"})
    assert resp.status_code == 200
    items = resp.json()["items"]
    assert items == sorted(items, key=lambda x: (x["asof"], x["relative_path"]))
    assert all(i["download_url"].startswith("/api/orchestrator/artifacts/2026-02-18/") for i in items)


def test_dates_list(test_client):
    resp = test_client.get("/api/orchestrator/dates")
    assert resp.status_code == 200
    assert resp.json()["items"] == ["2026-02-17", "2026-02-18"]


def test_risk_session_mismatch_structured_404(test_client):
    resp = test_client.get("/api/orchestrator/risk-checks", params={"asof": "2026-02-18", "session": "close"})
    assert resp.status_code == 404
    detail = resp.json()["detail"]
    assert detail["error_code"] == "session_mismatch"
    assert "expected_paths" in detail and "found_paths" in detail


def test_artifact_path_traversal_rejected(test_client):
    resp = test_client.get("/api/orchestrator/artifacts/2026-02-18/..%2F..%2Fetc%2Fpasswd")
    assert resp.status_code == 404
    assert resp.json()["detail"]["error_code"] == "artifact_path_invalid"


def test_equity_series_safe_float_handles_na(test_client):
    resp = test_client.get("/api/orchestrator/equity-series", params={"asof": "2026-02-18"})
    assert resp.status_code == 200
    rows = resp.json()["series"]
    assert len(rows) >= 2
    assert rows[1]["cum_net_unscaled"] == 0.0
    assert rows[1]["rolling_vol"] == 0.0

