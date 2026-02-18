from __future__ import annotations

import json

from backend.app.api.telemetry_routes import _sse, storage


def test_sse_frame_includes_monotonic_id_lines():
    run_id = "test-run-sse-ids"
    storage.publish(run_id, {"type": "metric", "data": {"value": 1}})
    storage.publish(run_id, {"type": "metric", "data": {"value": 2}})
    replayed = storage.replay_since(run_id, 0)
    assert len(replayed) >= 2
    ids = [item["id"] for item in replayed[-2:]]
    assert ids[1] > ids[0]

    frame = _sse("metric", json.dumps({"value": 2}), ids[1])
    assert frame.startswith(f"id: {ids[1]}\n")
    assert "event: metric\n" in frame
