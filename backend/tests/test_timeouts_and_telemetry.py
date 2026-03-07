from __future__ import annotations

import pytest

import time
from datetime import date
import queue

from backend.app.orchestrator.job_defs import Job
from backend.app.orchestrator.runner import JobRunner
from backend.app.orchestrator.calendar import Session
from backend.app.orchestrator.orchestrator import Orchestrator
from backend.app.orchestrator.config import OrchestratorConfig
from backend.app.telemetry.storage import TelemetryStorage


def test_job_timeout_enforced(tmp_path):
    def sleepy(_):
        time.sleep(0.2)
        return {"status": "ok"}

    runner = JobRunner()
    job = Job("sleepy", {Session.OPEN}, [], {"paper"}, timeout_s=0, retries=0, handler=sleepy)
    result = runner.run(job, {}, tmp_path / "jobs")
    assert result.status == "failed"
    assert "timeout" in (result.error_summary or "").lower()


def test_telemetry_publish_non_blocking_drops_under_pressure(tmp_path):
    storage = TelemetryStorage(db_url=f"sqlite:///{tmp_path / 'telemetry.db'}")
    q = storage.subscribe("run-1")
    # Fill queue to capacity — use maxsize if available, else try 1000
    capacity = q.maxsize if hasattr(q, 'maxsize') and q.maxsize > 0 else 1000
    filled = 0
    for _ in range(capacity + 10):
        try:
            q.put_nowait({"id": 0, "type": "event", "data": {}})
            filled += 1
        except queue.Full:
            break
    # Now publish one more through the storage layer — should be dropped, not block
    storage.publish("run-1", {"type": "event", "data": {"x": 1}})
    snap = storage.stream_snapshot("run-1")
    assert snap["dropped_events"] >= 1
