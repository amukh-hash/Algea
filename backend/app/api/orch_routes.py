"""Orchestrator status API — reads from orchestrator SQLite + artifact files.

All endpoints enforce a 5-second timeout to prevent indefinite hangs.
On timeout, returns HTTP 504 with a clear error JSON.
"""
from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from datetime import date, datetime
from functools import wraps
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse

router = APIRouter(prefix="/api/orchestrator", tags=["orchestrator"])
logger = logging.getLogger("algaie.api.orchestrator")

# Default paths — match OrchestratorConfig defaults
_ARTIFACT_ROOT = Path("backend/artifacts/orchestrator")
_DB_PATH = Path("backend/artifacts/orchestrator_state/state.sqlite3")

# Timeout for all sync operations (seconds)
_TIMEOUT_S = 5.0


def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(_DB_PATH, timeout=2)
    conn.row_factory = sqlite3.Row
    return conn


def _today() -> str:
    return date.today().isoformat()


async def _with_timeout(fn, *args, timeout: float = _TIMEOUT_S) -> Any:
    """Run a sync function in a thread with an upper-bound timeout.

    Returns the result or raises an HTTPException(504) on timeout.
    """
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(fn, *args),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        logger.error("timeout after %.1fs calling %s", timeout, fn.__name__)
        raise HTTPException(
            status_code=504,
            detail={"error": "timeout", "detail": f"{fn.__name__} did not complete within {timeout}s"},
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("error in %s: %s", fn.__name__, exc)
        raise HTTPException(
            status_code=500,
            detail={"error": "internal", "detail": str(exc)},
        )


# ── sync helpers (run inside threads) ────────────────────────────────

def _get_status_sync() -> dict[str, Any]:
    today = _today()
    hb_path = _ARTIFACT_ROOT / today / "heartbeat.json"
    heartbeat = json.loads(hb_path.read_text(encoding="utf-8")) if hb_path.exists() else None

    try:
        with _db() as conn:
            row = conn.execute(
                "SELECT * FROM runs ORDER BY started_at DESC LIMIT 1"
            ).fetchone()
        last_run = dict(row) if row else None
    except Exception:
        last_run = None

    return {"asof_date": today, "heartbeat": heartbeat, "last_run": last_run}


def _list_runs_sync(limit: int) -> dict[str, Any]:
    try:
        with _db() as conn:
            rows = conn.execute(
                "SELECT * FROM runs ORDER BY started_at DESC LIMIT ?", (limit,)
            ).fetchall()
        runs = []
        for r in rows:
            d = dict(r)
            if d.get("meta_json"):
                d["meta"] = json.loads(d.pop("meta_json"))
            runs.append(d)
        return {"items": runs, "total": len(runs)}
    except Exception as exc:
        return {"items": [], "total": 0, "error": str(exc)}


def _get_run_jobs_sync(run_id: str) -> dict[str, Any]:
    try:
        with _db() as conn:
            rows = conn.execute(
                "SELECT * FROM jobs WHERE run_id=? ORDER BY started_at", (run_id,)
            ).fetchall()
        return {"items": [dict(r) for r in rows]}
    except Exception as exc:
        return {"items": [], "error": str(exc)}


def _get_positions_sync(day: str) -> dict[str, Any]:
    path = _ARTIFACT_ROOT / day / "fills" / "positions.json"
    if not path.exists():
        return {"positions": [], "asof_date": day, "source": "none"}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {**data, "asof_date": day, "source": str(path)}


def _get_targets_sync(day: str) -> dict[str, Any]:
    targets_dir = _ARTIFACT_ROOT / day / "targets"
    result: dict[str, Any] = {"asof_date": day, "sleeves": {}}
    if targets_dir.exists():
        for f in sorted(targets_dir.glob("*.json")):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                sleeve = f.stem.replace("_targets", "")
                result["sleeves"][sleeve] = data
            except Exception:
                pass
    return result


def _get_fills_sync(day: str) -> dict[str, Any]:
    path = _ARTIFACT_ROOT / day / "fills" / "fills.json"
    if not path.exists():
        return {"fills": [], "asof_date": day}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {**data, "asof_date": day}


# ── async endpoints with timeout ─────────────────────────────────────

@router.get("/status")
async def get_status() -> dict[str, Any]:
    return await _with_timeout(_get_status_sync)


@router.get("/runs")
async def list_runs(limit: int = Query(20, ge=1, le=100)) -> dict[str, Any]:
    return await _with_timeout(_list_runs_sync, limit)


@router.get("/runs/{run_id}/jobs")
async def get_run_jobs(run_id: str) -> dict[str, Any]:
    return await _with_timeout(_get_run_jobs_sync, run_id)


@router.get("/positions")
async def get_positions(asof: str | None = None) -> dict[str, Any]:
    return await _with_timeout(_get_positions_sync, asof or _today())


@router.get("/targets")
async def get_targets(asof: str | None = None) -> dict[str, Any]:
    return await _with_timeout(_get_targets_sync, asof or _today())


@router.get("/fills")
async def get_fills(asof: str | None = None) -> dict[str, Any]:
    return await _with_timeout(_get_fills_sync, asof or _today())


# ── Serve any artifact file ──────────────────────────────────────────
@router.get("/artifacts/{day}/{path:path}")
async def get_artifact(day: str, path: str) -> FileResponse:
    root = (_ARTIFACT_ROOT / day).resolve()
    full = (root / path).resolve()
    if not str(full).startswith(str(root)):
        raise HTTPException(400, detail="invalid artifact path")
    if not full.exists() or not full.is_file() or full.is_symlink():
        raise HTTPException(404, detail="Artifact not found")
    mime = "application/json" if full.suffix == ".json" else "text/plain"
    return FileResponse(full, media_type=mime)
