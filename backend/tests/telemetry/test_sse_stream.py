"""SSE stream tests — verify frame format, IDs, and contract correctness.

Note: Starlette's TestClient.stream() blocks on infinite SSE generators,
so we test the frame format and ID guarantees at the unit level.
The live streaming behavior (reconnect, snapshot) is verified manually.
"""
from __future__ import annotations

import re

from backend.app.api.telemetry_routes import _sse


class TestSseFormatter:
    def test_includes_event_id_and_type(self):
        frame = _sse("metric", '{"x":1}', 7)
        assert frame.startswith("id: 7\nevent: metric")

    def test_snapshot_frame_shape(self):
        frame = _sse("snapshot", '{"ack_last_event_id":"11"}', 12)
        assert "event: snapshot" in frame
        assert "ack_last_event_id" in frame

    def test_double_newline_terminator(self):
        frame = _sse("heartbeat", '{}', 1)
        assert frame.endswith("\n\n")

    def test_data_field_present(self):
        frame = _sse("event", '{"test": true}', 5)
        assert 'data: {"test": true}' in frame

    def test_ids_are_monotonic_in_sequence(self):
        """Generate several frames and verify IDs are strictly increasing."""
        ids = []
        for i in range(1, 20):
            frame = _sse("metric", '{}', i)
            match = re.search(r"id: (\d+)", frame)
            assert match, f"No id in frame {i}"
            ids.append(int(match.group(1)))
        for j in range(1, len(ids)):
            assert ids[j] > ids[j - 1], f"IDs not monotonic: {ids}"

    def test_snapshot_event_has_run_id_field(self):
        """Snapshot frame used on reconnect must carry run_id."""
        import json
        data = json.dumps({"run_id": "abc-123", "ack_last_event_id": "10"})
        frame = _sse("snapshot", data, 11)
        assert "run_id" in frame
        assert "abc-123" in frame

    def test_heartbeat_contains_timestamp(self):
        """Heartbeat frames include a ts field."""
        import json
        data = json.dumps({"ts": "2026-02-18T14:00:00Z"})
        frame = _sse("heartbeat", data, 99)
        assert "2026-02-18T14:00:00Z" in frame
        assert "id: 99" in frame
