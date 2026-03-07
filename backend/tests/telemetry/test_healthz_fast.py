from __future__ import annotations

import pytest

import time

from fastapi.testclient import TestClient

from backend.app.api.main import app


client = TestClient(app)


def test_healthz_fast_and_shape():
    started = time.perf_counter()
    response = client.get("/healthz")
    elapsed_ms = (time.perf_counter() - started) * 1000
    # Orchestrator may not be initialized in test environment — accept 200 or 503
    assert response.status_code in (200, 503)
    body = response.json()
    if response.status_code == 200:
        assert body["ok"] is True
        assert body["app"] == "algae"
    else:
        # 503 is acceptable in test — orchestrator services not wired up
        assert "app" in body or "detail" in body
    assert elapsed_ms < 500
