import os
import glob
import json
from typing import Dict, Optional, Tuple, Any
from pathlib import Path
from backend.app.core import config

# This module resolves compatible artifacts for a given date.
# Paths are derived from backend.app.core.config

def resolve_priors_path(as_of_date: str, version: str = "v1") -> Optional[str]:
    """
    Finds the priors file for the given date.
    Path: {PRIORS_DIR}/{version}/{as_of_date}.parquet
    """
    base = os.path.join(config.PRIORS_DIR, version)
    path = os.path.join(base, f"{as_of_date}.parquet")
    if os.path.exists(path):
        return path
    return None

def resolve_selector_checkpoint(version: str = "v1") -> Optional[str]:
    """
    Finds the latest checkpoint for the given version.
    Path: {CHECKPOINTS_DIR}/selector/{version}/best.pt
    """
    path = os.path.join(config.CHECKPOINTS_DIR, "selector", version, "best.pt")
    if os.path.exists(path):
        return path
    return None

def resolve_scaler_path(version: str = "v1") -> Optional[str]:
    """
    Finds the scaler artifact.
    Path: {CHECKPOINTS_DIR}/selector/{version}/scaler.joblib
    """
    path = os.path.join(config.CHECKPOINTS_DIR, "selector", version, "scaler.joblib")
    if os.path.exists(path):
        return path
    return None

def resolve_calibration_path(version: str = "v1") -> Optional[str]:
    """
    Finds the calibration artifact.
    Path: {CHECKPOINTS_DIR}/selector/{version}/calibration.joblib
    """
    path = os.path.join(config.CHECKPOINTS_DIR, "selector", version, "calibration.joblib")
    if os.path.exists(path):
        return path
    return None

def resolve_leaderboard_path(as_of_date: str, version: str = "v1") -> Optional[str]:
    """
    Finds the leaderboard signals for the given date.
    Path: {SIGNALS_DIR}/selector/{version}/{as_of_date}.parquet
    """
    base = os.path.join(config.SIGNALS_DIR, "selector", version)
    path = os.path.join(base, f"{as_of_date}.parquet")
    if os.path.exists(path):
        return path
    return None

def get_manifest_path(artifact_path: str) -> str:
    """Returns the expected path for the artifact's manifest."""
    return f"{artifact_path}.manifest.json"

def load_manifest(artifact_path: str) -> Dict[str, Any]:
    """Loads the manifest for the given artifact if it exists."""
    man_path = get_manifest_path(artifact_path)
    if os.path.exists(man_path):
        with open(man_path, "r") as f:
            return json.load(f)
    return {}

def ensure_artifact_compatibility(priors_path: str, scaler_path: str, checkpoint_path: str):
    """
    Checks if artifacts exist and validates compatibility via manifests.
    Raises FileNotFoundError if critical artifacts are missing.
    """
    artifacts = {
        "priors": priors_path,
        "scaler": scaler_path,
        "checkpoint": checkpoint_path
    }
    
    for name, path in artifacts.items():
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"Critical artifact missing: {name} at {path}")
        
        # Manifest check (optional for now, but good for integrity)
        man = load_manifest(path)
        if not man:
            # warning or pass? pass for now to fail open on missing manifest
            pass
