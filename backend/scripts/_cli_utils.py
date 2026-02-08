from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import List

from algaie.core.config import PipelineConfig, ensure_run_id, load_config
from algaie.core.logging import setup_logging
from algaie.core.paths import ArtifactPaths, ensure_artifact_dirs
from algaie.core.artifacts.registry import ArtifactRegistry


def load_pipeline_config(config_path: str) -> PipelineConfig:
    return load_config(config_path)


def prepare_run(config: PipelineConfig) -> tuple[ArtifactPaths, ArtifactRegistry, Path]:
    paths = ArtifactPaths(config.artifact_root)
    ensure_artifact_dirs(paths)
    run_id = ensure_run_id(config)
    run_dir = paths.runs / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(run_dir / "run.log")
    logger.info("run_id=%s", run_id)
    registry = ArtifactRegistry(paths.root)
    config_snapshot = run_dir / "config.json"
    payload = asdict(config)
    payload["artifact_root"] = str(config.artifact_root)
    config_snapshot.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return paths, registry, run_dir


def write_artifact_log(registry: ArtifactRegistry, run_dir: Path) -> None:
    registry.dump(run_dir / "artifacts.json")
