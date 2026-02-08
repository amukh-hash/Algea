from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.app.ops import run_paths


REQUIRED_REPORTS = {"preflight": "preflight_report.json", "gate": "gate_report.json"}


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_hash(obj: Dict[str, Any]) -> str:
    payload = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _atomic_write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent)) as tmp:
        json.dump(obj, tmp, indent=2)
        tmp_path = Path(tmp.name)
    import time
    max_retries = 5
    for i in range(max_retries):
        try:
            tmp_path.replace(path)
            return
        except PermissionError:
            if i == max_retries - 1:
                raise
            time.sleep(0.1)
        except OSError:
            # Handle other OS errors if needed, but PermissionError is the main Windows locking one
            if i == max_retries - 1:
                raise
            time.sleep(0.1)


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_run_dirs(run_dir: Path) -> None:
    (run_dir / "reports").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "outputs").mkdir(parents=True, exist_ok=True)


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_dir(path: Path) -> str:
    h = hashlib.sha256()
    for root, _, files in os.walk(path):
        for name in sorted(files):
            fpath = Path(root) / name
            rel = fpath.relative_to(path).as_posix()
            h.update(rel.encode("utf-8"))
            h.update(_hash_file(fpath).encode("utf-8"))
    return h.hexdigest()


def _hash_path(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    if path.is_dir():
        return _hash_dir(path)
    return _hash_file(path)


def _git_version() -> Dict[str, Any]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.call(["git", "diff", "--quiet"]) != 0
        return {"git_commit": commit, "dirty": dirty}
    except Exception:
        return {"git_commit": "unknown", "dirty": True}


def _next_run_id() -> str:
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    prefix = f"RUN-{date_str}-"
    runs_root = run_paths.get_runs_root()
    existing = []
    for entry in runs_root.iterdir():
        if not entry.is_dir():
            continue
        name = entry.name
        if name.startswith(prefix):
            suffix = name.replace(prefix, "")
            if suffix.isdigit():
                existing.append(int(suffix))
    next_idx = max(existing, default=0) + 1
    return f"{prefix}{next_idx:03d}"


def _relative_to_run(run_dir: Path, path: Path) -> str:
    try:
        return path.relative_to(run_dir).as_posix()
    except ValueError:
        return path.as_posix()


def init_run(
    pipeline_type: str,
    trigger: str,
    config: Dict[str, Any],
    data_versions: Dict[str, Any],
    tags: Optional[List[str]] = None,
) -> str:
    run_id = _next_run_id()
    run_dir = run_paths.get_run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=False)
    _ensure_run_dirs(run_dir)

    hardware = {
        "device": config.get("device", "cpu"),
        "dtype": config.get("dtype", "fp32"),
        "batch_size": int(config.get("batch_size", 0) or 0),
    }

    manifest = {
        "run_id": run_id,
        "pipeline_type": pipeline_type,
        "trigger": trigger,
        "start_time": _iso_now(),
        "end_time": None,
        "status": "RUNNING",
        "stage": None,
        "step": None,
        "code_version": _git_version(),
        "config": config,
        "config_hash": _stable_hash(config),
        "data_versions": data_versions,
        "hardware": hardware,
        "artifacts_root": str(run_dir),
    }
    if tags:
        manifest["tags"] = tags

    write_manifest(run_id, manifest)
    status = {
        "run_id": run_id,
        "status": "RUNNING",
        "stage": None,
        "step": None,
        "updated_at": _iso_now(),
        "error": None,
    }
    _atomic_write_json(run_dir / "status.json", status)

    (run_dir / "events.ndjson").touch()
    _atomic_write_json(run_dir / "artifacts_index.json", {"run_id": run_id, "artifacts": []})
    _atomic_write_json(run_dir / "checkpoints" / "checkpoint_index.json", {"run_id": run_id, "checkpoints": []})
    return run_id


def write_manifest(run_id: str, manifest: Dict[str, Any]) -> None:
    run_dir = run_paths.get_run_dir(run_id)
    _atomic_write_json(run_dir / "run_manifest.json", manifest)


def set_status(
    run_id: str,
    status: str,
    stage: Optional[str] = None,
    step: Optional[str] = None,
    error: Optional[Dict[str, Any]] = None,
) -> None:
    run_dir = run_paths.get_run_dir(run_id)
    status_path = run_dir / "status.json"
    status_obj = _read_json(status_path)
    status_obj.update(
        {
            "run_id": run_id,
            "status": status,
            "stage": stage,
            "step": step,
            "updated_at": _iso_now(),
            "error": error,
        }
    )
    _atomic_write_json(status_path, status_obj)

    manifest_path = run_dir / "run_manifest.json"
    manifest = _read_json(manifest_path)
    if manifest:
        manifest["status"] = status
        manifest["stage"] = stage
        manifest["step"] = step
        _atomic_write_json(manifest_path, manifest)


def emit_event(
    run_id: str,
    stage: str,
    step: str,
    level: str,
    message: str,
    payload: Optional[Dict[str, Any]] = None,
) -> None:
    run_dir = run_paths.get_run_dir(run_id)
    event = {
        "timestamp": _iso_now(),
        "run_id": run_id,
        "stage": stage,
        "step": step,
        "level": level,
        "message": message,
        "payload": payload or {},
    }
    with open(run_dir / "events.ndjson", "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


def emit_metric(run_id: str, step: int, epoch: Optional[float], metrics: Dict[str, Any]) -> None:
    run_dir = run_paths.get_run_dir(run_id)
    entry = {
        "timestamp": _iso_now(),
        "run_id": run_id,
        "step": step,
        "epoch": epoch,
        "metrics": metrics,
    }
    with open(run_dir / "metrics.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def register_artifact(
    run_id: str,
    name: str,
    type: str,
    path: str,
    tags: Optional[List[str]] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    run_dir = run_paths.get_run_dir(run_id)
    abs_path = Path(path)
    if not abs_path.is_absolute():
        abs_path = run_dir / path
    rel_path = _relative_to_run(run_dir, abs_path)
    entry = {
        "name": name,
        "type": type,
        "path": rel_path,
        "created_at": _iso_now(),
        "hash": _hash_path(abs_path),
        "tags": tags or [],
        "meta": meta or {},
    }
    index_path = run_dir / "artifacts_index.json"
    index = _read_json(index_path)
    artifacts = index.get("artifacts", [])
    artifacts.append(entry)
    index["run_id"] = run_id
    index["artifacts"] = artifacts
    _atomic_write_json(index_path, index)


def write_report(run_id: str, report_type: str, obj: Dict[str, Any]) -> Path:
    if report_type not in REQUIRED_REPORTS:
        raise ValueError(f"Unknown report_type: {report_type}")
    run_dir = run_paths.get_run_dir(run_id)
    report_path = run_dir / "reports" / REQUIRED_REPORTS[report_type]
    _atomic_write_json(report_path, obj)
    register_artifact(
        run_id,
        name=report_type,
        type="json",
        path=report_path,
        tags=["report", report_type],
    )
    return report_path


def register_checkpoint(
    run_id: str,
    checkpoint_path: str,
    step: int,
    epoch: float,
    component: str,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    run_dir = run_paths.get_run_dir(run_id)
    abs_path = Path(checkpoint_path)
    if not abs_path.is_absolute():
        abs_path = run_dir / checkpoint_path
    rel_path = _relative_to_run(run_dir, abs_path)
    entry = {
        "path": rel_path,
        "step": step,
        "epoch": epoch,
        "component": component,
        "created_at": _iso_now(),
        "hash": _hash_path(abs_path),
        "meta": meta or {},
    }
    index_path = run_dir / "checkpoints" / "checkpoint_index.json"
    index = _read_json(index_path)
    checkpoints = index.get("checkpoints", [])
    checkpoints.append(entry)
    index["run_id"] = run_id
    index["checkpoints"] = checkpoints
    _atomic_write_json(index_path, index)


def finalize_run(run_id: str, status: str) -> None:
    run_dir = run_paths.get_run_dir(run_id)
    manifest_path = run_dir / "run_manifest.json"
    manifest = _read_json(manifest_path)
    manifest["end_time"] = _iso_now()
    manifest["status"] = status
    _atomic_write_json(manifest_path, manifest)
    set_status(run_id, status, stage=manifest.get("stage"), step=manifest.get("step"))
