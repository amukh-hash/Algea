from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .state_store import StateStore


def read_latest_heartbeat(artifact_root: Path) -> dict[str, Any] | None:
    if not artifact_root.exists():
        return None
    heartbeats = sorted(artifact_root.glob("*/heartbeat.json"))
    if not heartbeats:
        return None
    return json.loads(heartbeats[-1].read_text(encoding="utf-8"))


def read_last_runs(state_store: StateStore, limit: int = 10) -> list[dict[str, Any]]:
    return state_store.list_runs(limit=limit)


def summarize_health(artifact_root: Path, state_store: StateStore) -> dict[str, Any]:
    heartbeat = read_latest_heartbeat(artifact_root)
    runs = read_last_runs(state_store, limit=10)
    last_success = next((r for r in runs if r.get("status") == "success"), None)
    last_failure = next((r for r in runs if r.get("status") == "failed"), None)
    return {
        "heartbeat": heartbeat,
        "last_success": last_success,
        "last_failure": last_failure,
        "run_count": len(runs),
    }
