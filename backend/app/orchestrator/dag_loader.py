"""DAG topology loader — YAML → Job bridge.

Reads ``dag_topology.yaml`` and produces ``Job`` objects that dispatch
inference work to ``GPUProcessSupervisor`` instead of importing ML modules
directly.  This is the F9 dependency-inversion layer.

**Backward Compatible**: ``load_yaml_jobs()`` returns standard ``Job``
objects that plug directly into the existing ``Orchestrator.run_once()``
loop — no orchestrator changes needed.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import yaml

from .calendar import Session
from .job_defs import Job

logger = logging.getLogger(__name__)

# Session name → Session enum lookup
_SESSION_MAP = {
    "premarket": Session.PREMARKET,
    "open": Session.OPEN,
    "intraday": Session.INTRADAY,
    "preclose": Session.PRECLOSE,
    "close": Session.CLOSE,
    "overnight": Session.OVERNIGHT,
}


def _make_gpu_handler(
    plugin_path: str,
    target_gpu: int,
    timeout_s: int,
    optimize_ada: bool = False,
) -> Any:
    """Create a closure that dispatches to GPUProcessSupervisor.

    Returns a standard ``JobHandler`` (callable(ctx) → dict) that can
    replace any hardcoded handler in ``default_jobs()``.  The closure
    lazily bootstraps the supervisor on first call.
    """
    # Supervisor cache — one per device, created lazily
    _supervisor_cache: dict[int, Any] = {}

    def handler(ctx: dict[str, Any]) -> dict[str, Any]:
        from backend.app.core.gpu_supervisor import GPUProcessSupervisor

        if target_gpu not in _supervisor_cache:
            _supervisor_cache[target_gpu] = GPUProcessSupervisor(
                device_id=target_gpu, timeout_s=timeout_s,
            )
        supervisor = _supervisor_cache[target_gpu]

        # Build context payload for the plugin
        artifact_dir = Path(ctx.get("artifact_root", ".")) / "intents"
        artifact_dir.mkdir(parents=True, exist_ok=True)

        plugin_context = {
            "asof_date": ctx["asof_date"],
            "session": ctx["session"],
            "artifact_dir": str(artifact_dir),
            "mode": ctx.get("mode", "paper"),
            "tick_id": ctx.get("tick_id", ""),
            "dry_run": ctx.get("dry_run", True),
        }

        supervisor.execute_job(
            plugin_path=plugin_path,
            context=plugin_context,
            optimize_ada=optimize_ada,
        )

        # Read generated intents if the plugin wrote them
        intent_file = artifact_dir / f"{plugin_path.rsplit('.', 1)[-1]}_intents.json"
        intents: list[dict] = []
        if intent_file.exists():
            intents = json.loads(intent_file.read_text(encoding="utf-8"))

        return {
            "status": "ok",
            "artifacts": {"intents": str(intent_file)},
            "metrics": {"n_intents": len(intents), "gpu": target_gpu},
        }

    return handler


def _make_cpu_handler(plugin_path: str) -> Any:
    """Create a closure that runs a CPU-bound plugin directly.

    Unlike GPU handlers, these do not dispatch to ``GPUProcessSupervisor``.
    Used for data ingest, risk checks, or other non-ML jobs.
    """

    def handler(ctx: dict[str, Any]) -> dict[str, Any]:
        plugin = importlib.import_module(plugin_path)

        artifact_dir = Path(ctx.get("artifact_root", ".")) / "intents"
        artifact_dir.mkdir(parents=True, exist_ok=True)

        plugin_context = {
            "asof_date": ctx["asof_date"],
            "session": ctx["session"],
            "artifact_dir": str(artifact_dir),
            "mode": ctx.get("mode", "paper"),
            "tick_id": ctx.get("tick_id", ""),
            "dry_run": ctx.get("dry_run", True),
        }

        plugin.execute(plugin_context, {})

        return {
            "status": "ok",
            "artifacts": {},
            "metrics": {"gpu": None},
        }

    return handler


def load_yaml_jobs(
    yaml_path: Path | str | None = None,
) -> list[Job]:
    """Parse ``dag_topology.yaml`` into standard ``Job`` objects.

    Parameters
    ----------
    yaml_path : Path, optional
        Defaults to ``configs/dag_topology.yaml`` relative to project root.

    Returns
    -------
    list[Job]
        Standard ``Job`` objects compatible with ``JobRunner.run()`` and
        ``filtered_jobs()``.
    """
    if yaml_path is None:
        yaml_path = Path("backend/configs/dag_topology.yaml")
    yaml_path = Path(yaml_path)

    if not yaml_path.exists():
        logger.warning("dag_topology.yaml not found at %s — returning empty", yaml_path)
        return []

    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    job_defs = raw.get("jobs", [])

    jobs: list[Job] = []
    for entry in job_defs:
        job_id = entry["id"]
        plugin_path = entry["plugin_path"]
        target_gpu_raw = entry.get("target_gpu", 0)
        target_gpu = int(target_gpu_raw) if target_gpu_raw is not None else None
        timeout_s = int(entry.get("timeout_s", 120))
        optimize_ada = bool(entry.get("optimize_ada", False))
        sessions = {_SESSION_MAP[s] for s in entry.get("sessions", ["premarket"])}
        deps = list(entry.get("deps", []))
        modes = set(entry.get("modes", ["paper", "live", "noop"]))

        if target_gpu is not None:
            handler = _make_gpu_handler(plugin_path, target_gpu, timeout_s, optimize_ada)
        else:
            # CPU-bound job — run directly without GPU dispatch
            handler = _make_cpu_handler(plugin_path)

        jobs.append(
            Job(
                name=job_id,
                sessions=sessions,
                deps=deps,
                mode_allow=modes,
                timeout_s=timeout_s,
                retries=0,
                handler=handler,
            )
        )
        logger.debug(
            "YAML job loaded: %s → gpu=%d plugin=%s",
            job_id, target_gpu, plugin_path,
        )

    logger.info("Loaded %d jobs from %s", len(jobs), yaml_path)
    return jobs
