from __future__ import annotations

import time

from fastapi.testclient import TestClient

from backend.app.api.main import app


client = TestClient(app)


def test_healthz_fast_and_shape():
    started = time.perf_counter()
    response = client.get("/healthz")
    elapsed_ms = (time.perf_counter() - started) * 1000
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["app"] == "algea"
    assert "orchestrator" in body
    assert elapsed_ms < 500
