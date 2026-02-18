from __future__ import annotations

from backend.app.api.telemetry_routes import _sse


def test_sse_formatter_includes_event_id_and_type():
    frame = _sse("metric", '{"x":1}', 7)
    assert frame.startswith("id: 7\nevent: metric")


def test_sse_snapshot_frame_shape():
    frame = _sse("snapshot", '{"ack_last_event_id":"11"}', 12)
    assert "event: snapshot" in frame
    assert "ack_last_event_id" in frame
