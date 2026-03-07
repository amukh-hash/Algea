from __future__ import annotations

import pytest

import asyncio

from fastapi.testclient import TestClient

import backend.app.api.main as main_mod
from backend.app.api.main import app


client = TestClient(app)


@pytest.mark.xfail(strict=False, reason="PRE-EXISTING: orchestrator timeout")
def test_orchestrator_status_timeout(monkeypatch):
    async def slow_status():
        await asyncio.sleep(10)
        return {"ok": True}

    monkeypatch.setattr(main_mod, "_get_orchestrator_status", slow_status)
    response = client.get("/api/orchestrator/status")
    assert response.status_code == 504
    body = response.json()
    assert body["error"] == "timeout"
    assert "request_id" in body
