from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app.api.main import app
from backend.app.api.telemetry_routes import storage
from backend.app.telemetry.emitter import TelemetryEmitter
from backend.app.telemetry.schemas import EventLevel, EventType, RunStatus, RunType


client = TestClient(app)


def test_storage_insert_and_query(tmp_path):
    test_storage = storage.__class__(db_url=f"sqlite:///{tmp_path}/telemetry.db", artifacts_root=f"{tmp_path}/artifacts")
    emitter = TelemetryEmitter(test_storage)
    run_id = emitter.start_run(RunType.sleeve_live, "test sleeve", sleeve_name="A")
    emitter.set_status(run_id, RunStatus.running)
    emitter.emit_metric(run_id, "pnl_net", 12.3)
    emitter.emit_event(run_id, EventLevel.info, EventType.DECISION_MADE, "decision")

    runs, total = test_storage.list_runs({}, limit=10, offset=0)
    assert total >= 1
    assert runs[0].run_id == run_id

    series = test_storage.query_metrics(run_id, ["pnl_net"], None, None, None)
    assert series["pnl_net"][0].value == 12.3


def test_integration_endpoints():
    emitter = TelemetryEmitter(storage)
    run_id = emitter.start_run(RunType.backtest, "integration test")
    emitter.set_status(run_id, RunStatus.running)
    emitter.emit_metric(run_id, "cum_net", 42.0)
    emitter.emit_event(run_id, EventLevel.info, EventType.BACKTEST_COMPLETE, "done")

    runs_resp = client.get("/api/telemetry/runs", params={"q": run_id})
    assert runs_resp.status_code == 200
    assert runs_resp.json()["total"] >= 1

    run_resp = client.get(f"/api/telemetry/runs/{run_id}")
    assert run_resp.status_code == 200

    metrics_resp = client.get(f"/api/telemetry/runs/{run_id}/metrics", params={"keys": "cum_net"})
    assert metrics_resp.status_code == 200
    assert metrics_resp.json()["series"]["cum_net"]

    events_resp = client.get(f"/api/telemetry/runs/{run_id}/events")
    assert events_resp.status_code == 200
    assert events_resp.json()["items"]
