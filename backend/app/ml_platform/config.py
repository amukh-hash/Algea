from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os


@dataclass(frozen=True)
class MLPlatformConfig:
    train_device: str = field(default_factory=lambda: os.getenv("TRAIN_DEVICE", "cuda:0"))
    infer_device: str = field(default_factory=lambda: os.getenv("INFER_DEVICE", "cuda:1"))
    model_root: Path = field(default_factory=lambda: Path(os.getenv("MODEL_ARTIFACT_ROOT", "backend/artifacts/models")))
    registry_db_path: Path = field(
        default_factory=lambda: Path(os.getenv("MODEL_REGISTRY_DB", "backend/artifacts/model_registry.sqlite"))
    )
    trace_root: Path = field(default_factory=lambda: Path(os.getenv("TRACE_ROOT", "backend/artifacts/traces")))
    hf_cache_root: Path = field(default_factory=lambda: Path(os.getenv("HF_CACHE_ROOT", "backend/artifacts/hf_cache")))
    min_train_vram_gb: float = field(default_factory=lambda: float(os.getenv("MIN_TRAIN_VRAM_GB", "8")))
    min_infer_vram_gb: float = field(default_factory=lambda: float(os.getenv("MIN_INFER_VRAM_GB", "6")))
    selector_model_alias: str = field(default_factory=lambda: os.getenv("SELECTOR_MODEL_ALIAS", "prod"))
    vrp_model_alias: str = field(default_factory=lambda: os.getenv("VRP_MODEL_ALIAS", "prod"))
    itransformer_model_alias: str = field(default_factory=lambda: os.getenv("ITRANSFORMER_MODEL_ALIAS", "prod"))
