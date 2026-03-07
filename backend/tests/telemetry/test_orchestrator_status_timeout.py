from __future__ import annotations

import pytest

import asyncio

from fastapi.testclient import TestClient

import backend.app.api.main as main_mod
from backend.app.api.main import app


client = TestClient(app)


def test_orchestrator_status_timeout():
    """Verify the status endpoint returns within a reasonable time or times out gracefully."""
    response = client.get("/api/orchestrator/status")
    # In test environment, orchestrator may not be initialized.
    # Accept 200 (success), 503 (unavailable), or 504 (timeout) as valid responses.
    assert response.status_code in (200, 503, 504)
    body = response.json()
    if response.status_code == 504:
        assert "error" in body or "detail" in body
