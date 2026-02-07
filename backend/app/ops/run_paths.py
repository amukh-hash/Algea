from __future__ import annotations

from pathlib import Path

DEFAULT_RUNS_ROOT = Path("backend/data/runs")


def get_runs_root() -> Path:
    runs_root = DEFAULT_RUNS_ROOT
    runs_root.mkdir(parents=True, exist_ok=True)
    return runs_root


def get_run_dir(run_id: str) -> Path:
    return get_runs_root() / run_id
