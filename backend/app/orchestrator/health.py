from __future__ import annotations

import json
from datetime import datetime
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


def last_routed_summary(artifact_root: Path) -> dict[str, Any] | None:
    """Read the newest orders/routed.json or orders/orders.json."""
    if not artifact_root.exists():
        return None
    # Search latest day folder
    day_dirs = sorted(artifact_root.glob("*/orders"), reverse=True)
    for orders_dir in day_dirs:
        routed = orders_dir / "routed.json"
        if routed.exists():
            return json.loads(routed.read_text(encoding="utf-8"))
        orders = orders_dir / "orders.json"
        if orders.exists():
            data = json.loads(orders.read_text(encoding="utf-8"))
            return {
                "source": "orders.json",
                "order_count": data.get("summary", {}).get("order_count", len(data.get("orders", []))),
                "total_abs_notional": data.get("summary", {}).get("total_abs_notional", 0),
                "dry_run": data.get("dry_run", True),
            }
    return None


def last_job_failure(state_store: StateStore) -> dict[str, Any] | None:
    """Return the most recent failed job from the jobs table."""
    with state_store._connect() as conn:
        row = conn.execute(
            "SELECT job_name, error_summary, ended_at FROM jobs WHERE status='failed' ORDER BY ended_at DESC LIMIT 1"
        ).fetchone()
    if row is None:
        return None
    return {"job_name": row["job_name"], "error_summary": row["error_summary"], "ended_at": row["ended_at"]}


def heartbeat_age_seconds(heartbeat: dict[str, Any] | None) -> float | None:
    """Return seconds since last heartbeat."""
    if heartbeat is None:
        return None
    ts = heartbeat.get("timestamp")
    if ts is None:
        return None
    try:
        hb_dt = datetime.fromisoformat(ts)
        return (datetime.now(hb_dt.tzinfo) - hb_dt).total_seconds()
    except Exception:
        return None


def summarize_health(artifact_root: Path, state_store: StateStore) -> dict[str, Any]:
    heartbeat = read_latest_heartbeat(artifact_root)
    runs = read_last_runs(state_store, limit=10)
    last_success = next((r for r in runs if r.get("status") == "success"), None)
    last_failure = next((r for r in runs if r.get("status") == "failed"), None)
    routed = last_routed_summary(artifact_root)
    job_fail = last_job_failure(state_store)
    return {
        "heartbeat": heartbeat,
        "heartbeat_age_s": heartbeat_age_seconds(heartbeat),
        "last_success": last_success,
        "last_failure": last_failure,
        "last_job_failure": job_fail,
        "last_routed": routed,
        "run_count": len(runs),
    }
