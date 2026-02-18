from __future__ import annotations

from backend.app.api.telemetry_routes import storage


def test_last_event_id_replay_or_snapshot_fallback():
    run_id = "test-run-last-event"
    storage.publish(run_id, {"type": "status", "data": {"status": "running"}})
    storage.publish(run_id, {"type": "metric", "data": {"key": "equity", "value": 100}})
    replay = storage.replay_since(run_id, 0)
    assert len(replay) >= 2

    empty_replay = storage.replay_since(run_id, 999999)
    assert empty_replay == []
    snapshot = storage.stream_snapshot(run_id)
    assert "status" in snapshot
    assert "last_event_id" in snapshot
