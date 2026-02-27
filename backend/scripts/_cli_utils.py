from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import List

from algea.core.config import PipelineConfig, ensure_run_id, load_config
from algea.core.logging import setup_logging
from algea.core.paths import ArtifactPaths, ensure_artifact_dirs
from algea.core.artifacts.registry import ArtifactRegistry


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


def detect_ffn_modules(model) -> list[str]:
    """Detect FFN projection module name suffixes for LoRA targeting.

    Shared across training and inference-verification scripts to avoid
    duplicating the detection heuristic.
    """
    candidates = {
        "wi", "wo", "wi_0", "wi_1", "dense", "fc1", "fc2",
        "gate_proj", "up_proj", "down_proj", "mlp",
    }
    found: set[str] = set()
    for name, mod in model.named_modules():
        if hasattr(mod, "weight") and mod.weight is not None:
            short = name.split(".")[-1]
            if short in candidates or any(c in short for c in candidates):
                found.add(short)
    return sorted(found)


def normalise_ohlcv_columns(df):
    """Rename 'ticker' -> 'symbol' if needed (common across data scripts)."""
    if "ticker" in df.columns and "symbol" not in df.columns:
        df = df.rename(columns={"ticker": "symbol"})
    return df
