"""Orchestrator API routes backed by SQLite state + emitted artifacts."""
from __future__ import annotations

import asyncio
import csv
import json
import logging
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
import uuid
from dataclasses import dataclass, field
from enum import IntEnum

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from fastapi.responses import FileResponse

router = APIRouter(prefix="/api/orchestrator", tags=["orchestrator"])
logger = logging.getLogger("algea.api.orchestrator")

_ARTIFACT_ROOT = Path("backend/artifacts/orchestrator")
_DB_PATH = Path("backend/artifacts/orchestrator_state/state.sqlite3")
_TIMEOUT_S = 5.0

class TaskPriority(IntEnum):
    URGENT = 1
    HIGH = 2
    BACKGROUND = 5

@dataclass(order=True)
class AsyncJob:
    priority: int
    submit_time: float
    job_id: str = field(compare=False)
    payload: dict[str, Any] = field(compare=False)

_job_queue: asyncio.PriorityQueue[AsyncJob] = asyncio.PriorityQueue()
_job_status: dict[str, dict[str, Any]] = {}

async def _job_worker():
    while True:
        try:
            job = await _job_queue.get()
            _job_status[job.job_id]["status"] = "running"
            try:
                # Emulate task execution matching target priority GPU
                # i.e background_heavy -> cuda:0, critical_realtime -> cuda:1
                await asyncio.sleep(0.1)
                _job_status[job.job_id]["status"] = "completed"
            except Exception as e:
                _job_status[job.job_id]["status"] = "failed"
                _job_status[job.job_id]["error"] = str(e)
            finally:
                _job_queue.task_done()
        except asyncio.CancelledError:
            break

@router.on_event("startup")
async def startup_job_worker():
    asyncio.create_task(_job_worker())


def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(_DB_PATH, timeout=2)
    conn.row_factory = sqlite3.Row
    return conn


def _today() -> str:
    return date.today().isoformat()


def _found_paths(day_root: Path, names: tuple[str, ...]) -> list[str]:
    if not day_root.exists():
        return []
    out: list[str] = []
    for name in names:
        for p in day_root.rglob(name):
            if p.is_file():
                out.append(str(p))
    return sorted(set(out))


def _artifact_404(*, error_code: str, message: str, asof: str, expected_paths: list[Path], found_paths: list[str], hint: str) -> HTTPException:
    return HTTPException(
        status_code=404,
        detail={
            "error_code": error_code,
            "message": message,
            "asof": asof,
            "expected_paths": [str(p) for p in expected_paths],
            "found_paths": found_paths,
            "hint": hint,
        },
    )


async def _with_timeout(fn, *args, timeout: float = _TIMEOUT_S) -> Any:
    try:
        return await asyncio.wait_for(asyncio.to_thread(fn, *args), timeout=timeout)
    except asyncio.TimeoutError as exc:
        raise HTTPException(status_code=504, detail={"error": "timeout", "detail": f"{fn.__name__} exceeded {timeout}s"}) from exc


def _get_status_sync() -> dict[str, Any]:
    asof = _today()
    hb_path = _ARTIFACT_ROOT / asof / "heartbeat.json"
    heartbeat = json.loads(hb_path.read_text(encoding="utf-8")) if hb_path.exists() else None
    try:
        with _db() as conn:
            row = conn.execute("SELECT * FROM runs ORDER BY started_at DESC LIMIT 1").fetchone()
    except sqlite3.Error:
        row = None
    return {"asof_date": asof, "heartbeat": heartbeat, "last_run": dict(row) if row else None}


def _list_runs_sync(limit: int) -> dict[str, Any]:
    with _db() as conn:
        rows = conn.execute("SELECT * FROM runs ORDER BY started_at DESC LIMIT ?", (limit,)).fetchall()
    items: list[dict[str, Any]] = []
    for row in rows:
        payload = dict(row)
        meta = payload.pop("meta_json", None)
        if meta:
            payload["meta"] = json.loads(meta)
        items.append(payload)
    return {"items": items, "total": len(items)}


def _get_run_jobs_sync(run_id: str) -> dict[str, Any]:
    with _db() as conn:
        rows = conn.execute("SELECT * FROM jobs WHERE run_id=? ORDER BY started_at", (run_id,)).fetchall()
    return {"items": [dict(r) for r in rows]}


def _find_latest_day_sync() -> str | None:
    if not _ARTIFACT_ROOT.exists():
        return None
    days = sorted([p.name for p in _ARTIFACT_ROOT.iterdir() if p.is_dir()])
    return days[-1] if days else None


def _resolve_instance_sync(asof: str) -> tuple[Path, dict[str, Any]]:
    day_root = _ARTIFACT_ROOT / asof
    candidates = [
        day_root / "instance.json",
        day_root / "reports" / "instance.json",
        day_root / "eod_summary.json",
    ]
    for p in candidates:
        if not p.exists():
            continue
        payload = json.loads(p.read_text(encoding="utf-8"))
        if p.name == "eod_summary.json" and isinstance(payload, dict) and isinstance(payload.get("instance"), dict):
            payload = payload["instance"]
        return p, payload
    raise _artifact_404(
        error_code="instance_not_found",
        message="No instance artifact found for requested asof date.",
        asof=asof,
        expected_paths=candidates,
        found_paths=_found_paths(day_root, ("instance.json", "eod_summary.json")),
        hint="Run orchestrator instance/report generation for the requested date.",
    )


def _normalize_legacy_risk(raw: dict[str, Any], asof: str) -> dict[str, Any]:
    missing = raw.get("missing_sleeves") if isinstance(raw.get("missing_sleeves"), list) else []
    violations = list(raw.get("violations", [])) if isinstance(raw.get("violations"), list) else []
    if not violations and missing:
        violations.append({"code": "MISSING_TARGETS", "message": "missing targets", "details": {"missing_sleeves": missing}})
    if raw.get("nan_or_inf"):
        violations.append({"code": "NAN_INF", "message": "NaN/Inf detected", "details": {}})
    return {
        "schema_version": "legacy_normalized",
        "status": str(raw.get("status", "failed")),
        "checked_at": raw.get("checked_at"),
        "asof_date": raw.get("asof_date") or asof,
        "session": raw.get("session"),
        "reason": raw.get("reason") or ("legacy risk report failure" if raw.get("status") != "ok" else None),
        "missing_sleeves": missing,
        "inputs": raw.get("inputs", {}),
        "metrics": raw.get("metrics", {}),
        "limits": raw.get("limits", {}),
        "violations": violations,
        "per_sleeve": raw.get("per_sleeve", {}),
        "raw": raw,
    }


def _resolve_risk_sync(asof: str) -> tuple[Path, dict[str, Any]]:
    day_root = _ARTIFACT_ROOT / asof
    candidates = [
        day_root / "risk_checks.json",
        day_root / "reports" / "risk_checks.json",
        day_root / "risk" / "risk_checks.json",
    ]
    for p in candidates:
        if not p.exists():
            continue
        raw = json.loads(p.read_text(encoding="utf-8"))
        if all(k in raw for k in ("metrics", "limits", "violations")):
            payload = {"schema_version": "canonical", **raw, "raw": raw}
            payload.setdefault("asof_date", asof)
            payload.setdefault("per_sleeve", payload.get("metrics", {}).get("per_sleeve", {}))
            return p, payload
        return p, _normalize_legacy_risk(raw, asof)
    raise _artifact_404(
        error_code="risk_checks_not_found",
        message="No risk_checks artifact found for requested asof date.",
        asof=asof,
        expected_paths=candidates,
        found_paths=_found_paths(day_root, ("risk_checks.json",)),
        hint="Run risk_checks_global/eod_reports for the requested date.",
    )


def _csv_has_required_cols(path: Path, required: set[str]) -> bool:
    try:
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return bool(reader.fieldnames and required.issubset(set(reader.fieldnames)))
    except Exception:
        return False


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in {"", "nan", "+nan", "-nan", "inf", "+inf", "-inf", "n/a", "na", "none", "null"}:
            return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if parsed != parsed or parsed in (float("inf"), float("-inf")):
        return default
    return parsed


def _read_curve_csv(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = row.get("date") or row.get("timestamp") or row.get("time")
            if not t:
                continue
            out.append(
                {
                    "t": t,
                    "cum_net_unscaled": _safe_float(row.get("cum_net_unscaled") or row.get("cum_net")),
                    "cum_net_volscaled": _safe_float(row.get("cum_net_volscaled") or row.get("cum_net_unscaled")),
                    "drawdown": _safe_float(row.get("drawdown")),
                    "rolling_vol": _safe_float(row.get("rolling_vol") or row.get("realized_vol")),
                    "rolling_sharpe": _safe_float(row.get("rolling_sharpe") or row.get("sharpe")),
                    "turnover": _safe_float(row.get("turnover")),
                    "cost": _safe_float(row.get("cost") or row.get("cost_scaled")),
                }
            )
    return out


def _resolve_equity_series_sync(asof: str, sleeve: str | None) -> tuple[Path, list[dict[str, Any]]]:
    day_root = _ARTIFACT_ROOT / asof
    required = {"cum_net_unscaled", "cum_net_volscaled"}
    csvs = sorted([p for p in day_root.rglob("*.csv") if p.is_file()])
    preferred_names = (
        f"{sleeve}_equity_curve.csv" if sleeve else "portfolio_equity_curve.csv",
        "equity_series.csv",
        "equity_curve.csv",
    )

    candidates: list[Path] = []
    for name in preferred_names:
        candidates.extend([p for p in csvs if p.name == name])
    if sleeve:
        candidates.extend([p for p in csvs if sleeve in p.name.lower() and "equity" in p.name.lower()])
    candidates.extend(csvs)

    deduped: list[Path] = []
    seen: set[str] = set()
    for p in candidates:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(p)

    with_required = [p for p in deduped if _csv_has_required_cols(p, required)]
    if with_required:
        chosen = sorted(with_required, key=lambda p: p.stat().st_mtime)[-1]
        return chosen, _read_curve_csv(chosen)

    error_code = "sleeve_equity_not_found" if sleeve else "equity_series_not_found"
    message = "Sleeve equity series not found for requested date." if sleeve else "No equity series found for requested date."
    hint = "Ensure sleeve-level equity artifacts are emitted for this sleeve/date." if sleeve else "Ensure portfolio equity CSV with required columns is emitted."
    raise _artifact_404(
        error_code=error_code,
        message=message,
        asof=asof,
        expected_paths=[day_root / n for n in preferred_names],
        found_paths=[str(p) for p in csvs],
        hint=hint,
    )


def _job_registry_sync() -> dict[str, Any]:
    from backend.app.orchestrator.job_defs import default_jobs

    defaults = default_jobs()
    by_name = {j.name: j for j in defaults}

    with _db() as conn:
        rows = conn.execute(
            """
            SELECT job_name,
                   status AS last_status,
                   error_summary AS last_error,
                   last_success_at,
                   CAST((julianday(ended_at)-julianday(started_at))*86400 AS INTEGER) AS last_duration_s
            FROM jobs
            WHERE id IN (
                SELECT MAX(id) FROM jobs GROUP BY job_name
            )
            """
        ).fetchall()

    last_by_name = {str(r["job_name"]): dict(r) for r in rows}
    items: list[dict[str, Any]] = []
    for name, job in sorted(by_name.items()):
        last = last_by_name.get(name, {})
        min_interval_s = int(job.min_interval_s)
        last_success = last.get("last_success_at")
        next_eligible_at = None
        if last_success:
            try:
                dt = datetime.fromisoformat(str(last_success).replace("Z", "+00:00"))
                next_eligible_at = (dt + timedelta(seconds=min_interval_s)).isoformat()
            except ValueError:
                next_eligible_at = None
        items.append(
            {
                "name": name,
                "sessions": sorted([s.value for s in job.sessions]),
                "depends_on": list(job.deps),
                "min_interval_s": min_interval_s,
                "last_success_at": last_success,
                "last_status": last.get("last_status"),
                "last_error": last.get("last_error"),
                "last_duration_s": last.get("last_duration_s"),
                "next_eligible_at": next_eligible_at,
            }
        )
    return {"items": items, "total": len(items)}


def _job_history_sync(limit: int, asof: str | None) -> dict[str, Any]:
    where = "WHERE asof_date=?" if asof else ""
    params: tuple[Any, ...] = (asof, limit) if asof else (limit,)
    with _db() as conn:
        rows = conn.execute(
            f"""
            SELECT run_id, asof_date, session, job_name,
                   status AS last_status,
                   started_at,
                   ended_at,
                   error_summary AS last_error,
                   last_success_at,
                   CAST((julianday(ended_at)-julianday(started_at))*86400 AS INTEGER) AS last_duration_s
            FROM jobs
            {where}
            ORDER BY COALESCE(started_at, asof_date) DESC
            LIMIT ?
            """,
            params,
        ).fetchall()

    registry = _job_registry_sync()["items"]
    min_interval_map = {j["name"]: int(j["min_interval_s"]) for j in registry}

    items: list[dict[str, Any]] = []
    for row in rows:
        payload = dict(row)
        payload["name"] = payload.pop("job_name")
        mi = min_interval_map.get(payload["name"], 0)
        payload["min_interval_s"] = mi
        payload["next_eligible_at"] = None
        if payload.get("last_success_at"):
            try:
                dt = datetime.fromisoformat(str(payload["last_success_at"]).replace("Z", "+00:00"))
                payload["next_eligible_at"] = (dt + timedelta(seconds=mi)).isoformat()
            except ValueError:
                payload["next_eligible_at"] = None
        items.append(payload)
    return {"items": items, "total": len(items)}


def _list_artifacts_sync(asof: str | None) -> dict[str, Any]:
    roots = [_ARTIFACT_ROOT / asof] if asof else sorted([p for p in _ARTIFACT_ROOT.iterdir() if p.is_dir()])
    items: list[dict[str, Any]] = []
    for root in roots:
        if not root.exists() or not root.is_dir():
            continue
        for p in sorted(root.rglob("*")):
            if not p.is_file():
                continue
            rel = str(p.relative_to(root))
            stat = p.stat()
            items.append(
                {
                    "asof": root.name,
                    "name": p.name,
                    "relative_path": rel,
                    "size_bytes": stat.st_size,
                    "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "download_url": f"/api/orchestrator/artifacts/{root.name}/{rel}",
                }
            )
    items.sort(key=lambda x: (x["asof"], x["relative_path"]))
    return {"items": items, "total": len(items)}


def _list_dates_sync() -> dict[str, Any]:
    if not _ARTIFACT_ROOT.exists():
        return {"items": []}
    return {"items": sorted([p.name for p in _ARTIFACT_ROOT.iterdir() if p.is_dir()])}


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
    day = asof or _today()
    path = _ARTIFACT_ROOT / day / "fills" / "positions.json"
    if not path.exists():
        return {"positions": [], "asof_date": day, "source": "none"}
    return {**json.loads(path.read_text(encoding="utf-8")), "asof_date": day, "source": str(path)}


@router.get("/targets")
async def get_targets(asof: str | None = None) -> dict[str, Any]:
    day = asof or _today()
    targets_dir = _ARTIFACT_ROOT / day / "targets"
    out: dict[str, Any] = {"asof_date": day, "sleeves": {}}
    if targets_dir.exists():
        for p in sorted(targets_dir.glob("*.json")):
            out["sleeves"][p.stem.replace("_targets", "")] = json.loads(p.read_text(encoding="utf-8"))
    return out


@router.get("/fills")
async def get_fills(asof: str | None = None) -> dict[str, Any]:
    day = asof or _today()
    path = _ARTIFACT_ROOT / day / "fills" / "fills.json"
    if not path.exists():
        return {"fills": [], "asof_date": day, "source": "none"}
    return {**json.loads(path.read_text(encoding="utf-8")), "asof_date": day, "source": str(path)}


@router.get("/instance")
async def get_instance(asof: str = Query(...)) -> dict[str, Any]:
    path, instance = await _with_timeout(_resolve_instance_sync, asof)
    return {"asof": asof, "asof_date": asof, "instance": instance, "source": str(path), "sleeves": instance.get("sleeves") if isinstance(instance, dict) else None}


@router.get("/instance/latest")
async def get_instance_latest() -> dict[str, Any]:
    latest = await _with_timeout(_find_latest_day_sync)
    if not latest:
        raise _artifact_404(
            error_code="instance_not_found",
            message="No orchestrator asof directories found.",
            asof="latest",
            expected_paths=[_ARTIFACT_ROOT],
            found_paths=[],
            hint="Run orchestrator to emit artifacts.",
        )
    path, instance = await _with_timeout(_resolve_instance_sync, latest)
    return {"asof": latest, "asof_date": latest, "instance": instance, "source": str(path), "sleeves": instance.get("sleeves") if isinstance(instance, dict) else None}


@router.get("/instance/{day}")
async def get_instance_day(day: str) -> dict[str, Any]:
    return await get_instance(day)


@router.get("/risk-checks")
async def get_risk_checks(asof: str = Query(...), session: str | None = None) -> dict[str, Any]:
    path, payload = await _with_timeout(_resolve_risk_sync, asof)
    if session and payload.get("session") and payload.get("session") != session:
        raise _artifact_404(
            error_code="session_mismatch",
            message="Requested session does not match artifact session.",
            asof=asof,
            expected_paths=[path],
            found_paths=[str(path)],
            hint="Use the correct session value or omit the session parameter.",
        )
    return {"asof": asof, "asof_date": asof, "risk_checks": payload, "source": str(path)}


@router.get("/risk-checks/latest")
async def get_risk_checks_latest() -> dict[str, Any]:
    latest = await _with_timeout(_find_latest_day_sync)
    if not latest:
        raise _artifact_404(
            error_code="risk_checks_not_found",
            message="No orchestrator asof directories found.",
            asof="latest",
            expected_paths=[_ARTIFACT_ROOT],
            found_paths=[],
            hint="Run orchestrator risk checks.",
        )
    path, payload = await _with_timeout(_resolve_risk_sync, latest)
    return {"asof": latest, "asof_date": latest, "risk_checks": payload, "source": str(path)}


@router.get("/risk-checks/{day}")
async def get_risk_checks_day(day: str) -> dict[str, Any]:
    return await get_risk_checks(day)


@router.get("/equity-series")
async def get_equity_series(asof: str = Query(...), sleeve: str | None = None) -> dict[str, Any]:
    path, series = await _with_timeout(_resolve_equity_series_sync, asof, sleeve)
    return {"asof": asof, "asof_date": asof, "scope": sleeve or "portfolio", "series": series, "source": str(path)}


@router.get("/timeseries")
async def get_timeseries(asof: str | None = None, sleeve: str | None = None) -> dict[str, Any]:
    day = asof or _today()
    return await get_equity_series(day, sleeve)


@router.get("/jobs")
async def get_jobs() -> dict[str, Any]:
    return await _with_timeout(_job_registry_sync)


@router.get("/jobs/registry")
async def get_jobs_registry() -> dict[str, Any]:
    return await _with_timeout(_job_registry_sync)


@router.post("/jobs/enqueue")
async def enqueue_job(payload: dict[str, Any], priority: int = TaskPriority.BACKGROUND) -> dict[str, Any]:
    job_id = str(uuid.uuid4())
    job = AsyncJob(priority=priority, submit_time=datetime.now().timestamp(), job_id=job_id, payload=payload)
    _job_status[job_id] = {
        "status": "pending",
        "priority": priority,
        "submitted_at": job.submit_time
    }
    await _job_queue.put(job)
    return {"job_id": job_id, "status": "pending"}


@router.get("/jobs/status/{job_id}")
async def get_job_status(job_id: str) -> dict[str, Any]:
    if job_id not in _job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    return _job_status[job_id]


@router.get("/jobs/history")
async def get_jobs_history(limit: int = Query(100, ge=1, le=500), asof: str | None = None) -> dict[str, Any]:
    return await _with_timeout(_job_history_sync, limit, asof)


@router.get("/dates")
async def get_dates() -> dict[str, Any]:
    return await _with_timeout(_list_dates_sync)


@router.get("/artifacts")
async def get_artifacts(asof: str | None = None) -> dict[str, Any]:
    return await _with_timeout(_list_artifacts_sync, asof)


@router.get("/artifacts/{day}/{path:path}")
async def get_artifact(day: str, path: str) -> FileResponse:
    root = (_ARTIFACT_ROOT / day).resolve()
    full = (root / path).resolve()

    if root not in full.parents and full != root:
        raise _artifact_404(
            error_code="artifact_path_invalid",
            message="Requested artifact path is invalid.",
            asof=day,
            expected_paths=[root],
            found_paths=[],
            hint="Use /api/orchestrator/artifacts to discover valid download_url paths.",
        )

    if not full.exists() or not full.is_file():
        raise _artifact_404(
            error_code="artifact_file_not_found",
            message="Requested artifact file was not found.",
            asof=day,
            expected_paths=[full],
            found_paths=[],
            hint="Check /api/orchestrator/artifacts for available files.",
        )

    if full.suffix == ".json":
        mime = "application/json"
    elif full.suffix == ".csv":
        mime = "text/csv"
    else:
        mime = "text/plain"
    return FileResponse(full, media_type=mime)
