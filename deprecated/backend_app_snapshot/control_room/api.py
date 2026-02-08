from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse

from backend.app.ops import run_paths

router = APIRouter()


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def _run_dirs() -> List[Path]:
    root = run_paths.get_runs_root()
    return [p for p in root.iterdir() if p.is_dir()]


def _run_summary(run_dir: Path) -> Optional[Dict[str, Any]]:
    manifest = _load_json(run_dir / "run_manifest.json")
    status = _load_json(run_dir / "status.json")
    if not manifest and not status:
        return None
    return {"manifest": manifest, "status": status}


def _filter_runs(
    runs: List[Dict[str, Any]],
    status: Optional[str],
    pipeline_type: Optional[str],
    start_from: Optional[datetime],
    start_to: Optional[datetime],
    search: Optional[str],
) -> List[Dict[str, Any]]:
    filtered = []
    for run in runs:
        manifest = run.get("manifest") or {}
        status_obj = run.get("status") or {}
        run_status = status_obj.get("status") or manifest.get("status")
        run_pipeline = manifest.get("pipeline_type")
        if status and run_status != status:
            continue
        if pipeline_type and run_pipeline != pipeline_type:
            continue
        start_time = _parse_iso(manifest.get("start_time"))
        if start_from and start_time and start_time < start_from:
            continue
        if start_to and start_time and start_time > start_to:
            continue
        if search:
            haystack = f"{manifest.get('run_id','')} {run_pipeline} {manifest.get('tags','')}"
            if search.lower() not in haystack.lower():
                continue
        filtered.append(run)
    return filtered


@router.get("/runs")
def list_runs(
    status: Optional[str] = Query(None),
    pipeline_type: Optional[str] = Query(None),
    from_: Optional[str] = Query(None, alias="from"),
    to: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    search: Optional[str] = Query(None),
) -> List[Dict[str, Any]]:
    runs = []
    for run_dir in _run_dirs():
        summary = _run_summary(run_dir)
        if summary:
            runs.append(summary)
    runs.sort(key=lambda r: r.get("manifest", {}).get("start_time") or "", reverse=True)
    filtered = _filter_runs(runs, status, pipeline_type, _parse_iso(from_), _parse_iso(to), search)
    return filtered[:limit]


@router.get("/runs/{run_id}")
def get_run(run_id: str) -> Dict[str, Any]:
    run_dir = run_paths.get_run_dir(run_id)
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Run not found")
    manifest = _load_json(run_dir / "run_manifest.json")
    status = _load_json(run_dir / "status.json")
    if not manifest:
        raise HTTPException(status_code=404, detail="Run manifest not found")
    return {"manifest": manifest, "status": status}


@router.get("/runs/{run_id}/events")
def get_events(run_id: str, tail: int = Query(500, ge=1, le=5000)) -> List[Dict[str, Any]]:
    run_dir = run_paths.get_run_dir(run_id)
    path = run_dir / "events.ndjson"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Events not found")
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()[-tail:]
    return [json.loads(line) for line in lines if line.strip()]


@router.get("/runs/{run_id}/metrics")
def get_metrics(run_id: str) -> List[Dict[str, Any]]:
    run_dir = run_paths.get_run_dir(run_id)
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@router.get("/runs/{run_id}/artifacts")
def get_artifacts(run_id: str) -> Dict[str, Any]:
    run_dir = run_paths.get_run_dir(run_id)
    index = _load_json(run_dir / "artifacts_index.json")
    if not index:
        raise HTTPException(status_code=404, detail="Artifacts index not found")
    return index


@router.get("/runs/{run_id}/reports/preflight")
def get_preflight_report(run_id: str) -> Dict[str, Any]:
    run_dir = run_paths.get_run_dir(run_id)
    report = _load_json(run_dir / "reports" / "preflight_report.json")
    if not report:
        raise HTTPException(status_code=404, detail="Preflight report not found")
    return report


@router.get("/runs/{run_id}/reports/gate")
def get_gate_report(run_id: str) -> Dict[str, Any]:
    run_dir = run_paths.get_run_dir(run_id)
    report = _load_json(run_dir / "reports" / "gate_report.json")
    if not report:
        raise HTTPException(status_code=404, detail="Gate report not found")
    return report


@router.get("/runs/{run_id}/checkpoints")
def get_checkpoints(run_id: str) -> Dict[str, Any]:
    run_dir = run_paths.get_run_dir(run_id)
    index = _load_json(run_dir / "checkpoints" / "checkpoint_index.json")
    if not index:
        raise HTTPException(status_code=404, detail="Checkpoint index not found")
    return index


def _allowed_paths(run_dir: Path) -> Dict[str, Path]:
    allowed: Dict[str, Path] = {}
    artifacts = _load_json(run_dir / "artifacts_index.json")
    if artifacts:
        for artifact in artifacts.get("artifacts", []):
            rel = artifact.get("path")
            if rel:
                allowed[rel] = run_dir / rel
    reports_dir = run_dir / "reports"
    for report_name in ("preflight_report.json", "gate_report.json"):
        report_path = reports_dir / report_name
        if report_path.exists():
            rel = report_path.relative_to(run_dir).as_posix()
            allowed[rel] = report_path
    checkpoints = _load_json(run_dir / "checkpoints" / "checkpoint_index.json")
    if checkpoints:
        for checkpoint in checkpoints.get("checkpoints", []):
            rel = checkpoint.get("path")
            if rel:
                allowed[rel] = run_dir / rel
    return allowed


@router.get("/runs/{run_id}/file")
def get_file(run_id: str, path: str = Query(...)):
    if ".." in path or path.startswith("/"):
        raise HTTPException(status_code=400, detail="Invalid path")
    run_dir = run_paths.get_run_dir(run_id)
    allowed = _allowed_paths(run_dir)
    if path not in allowed:
        raise HTTPException(status_code=404, detail="File not allowed")
    file_path = allowed[path]
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    suffix = file_path.suffix.lower()
    media_type = {
        ".json": "application/json",
        ".html": "text/html",
        ".parquet": "application/octet-stream",
        ".pt": "application/octet-stream",
    }.get(suffix, "application/octet-stream")
    if suffix == ".json":
        return JSONResponse(content=_load_json(file_path))
    return FileResponse(path=file_path, media_type=media_type, filename=file_path.name)


@router.get("/artifacts")
def search_artifacts(search: Optional[str] = Query(None)) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for run_dir in _run_dirs():
        run_id = run_dir.name
        artifacts = _load_json(run_dir / "artifacts_index.json")
        if not artifacts:
            continue
        for artifact in artifacts.get("artifacts", []):
            if search:
                haystack = f"{artifact.get('name','')} {artifact.get('type','')} {artifact.get('path','')}"
                if search.lower() not in haystack.lower():
                    continue
            results.append({"run_id": run_id, **artifact})
    return results
