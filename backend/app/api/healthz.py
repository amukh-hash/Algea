"""Lightweight health-check endpoint — must respond within 50ms."""
from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter

router = APIRouter(tags=["ops"])

_ARTIFACT_ROOT = Path("backend/artifacts/orchestrator")
_DB_PATH = Path("backend/artifacts/orchestrator_state/state.sqlite3")


@router.get("/healthz")
async def healthz() -> dict:
    t0 = time.monotonic()
    checks: dict[str, dict] = {}

    # 1. Event-loop responsiveness — should be near-zero
    loop_start = time.monotonic()
    await asyncio.sleep(0)
    loop_ms = (time.monotonic() - loop_start) * 1000
    checks["event_loop"] = {"ok": loop_ms < 100, "latency_ms": round(loop_ms, 2)}

    # 2. Orchestrator DB reachable
    try:
        conn = sqlite3.connect(_DB_PATH, timeout=1)
        row = conn.execute("SELECT COUNT(*) FROM runs").fetchone()
        conn.close()
        checks["state_db"] = {"ok": True, "run_count": row[0]}
    except Exception as exc:
        checks["state_db"] = {"ok": False, "error": str(exc)}

    # 3. Heartbeat freshness
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    hb_path = _ARTIFACT_ROOT / today / "heartbeat.json"
    if hb_path.exists():
        try:
            hb = json.loads(hb_path.read_text(encoding="utf-8"))
            hb_ts = datetime.fromisoformat(hb["timestamp"])
            age_s = (datetime.now(timezone.utc) - hb_ts.astimezone(timezone.utc)).total_seconds()
            checks["heartbeat"] = {
                "ok": age_s < 300,
                "age_seconds": round(age_s, 1),
                "session": hb.get("session"),
                "state": hb.get("state"),
            }
        except Exception as exc:
            checks["heartbeat"] = {"ok": False, "error": str(exc)}
    else:
        checks["heartbeat"] = {"ok": False, "error": "no heartbeat today"}

    all_ok = all(c["ok"] for c in checks.values())
    elapsed_ms = round((time.monotonic() - t0) * 1000, 2)

    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=200 if all_ok else 503,
        content={"ok": all_ok, "elapsed_ms": elapsed_ms, "checks": checks},
    )
