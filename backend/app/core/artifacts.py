import os
import glob
from typing import Dict, Optional, Tuple
from pathlib import Path

# This module resolves compatible artifacts for a given date.
# In a real system, this might query a database or registry.
# Here we use filesystem conventions:
# backend/data/priors/chronos2/v1/YYYY-MM-DD.parquet
# backend/data/signals/selector/v1/YYYY-MM-DD.parquet
# backend/data/checkpoints/selector/v1/...

def resolve_priors_path(as_of_date: str, version: str = "v1") -> Optional[str]:
    """
    Finds the priors file for the given date.
    Path: backend/data/priors/{version}/{as_of_date}.parquet
    """
    base = os.path.join("backend/data/priors", version)
    path = os.path.join(base, f"{as_of_date}.parquet")
    if os.path.exists(path):
        return path
    return None

def resolve_selector_checkpoint(version: str = "v1") -> Optional[str]:
    """
    Finds the latest checkpoint for the given version.
    Path: backend/data/checkpoints/selector/{version}/best.pt
    """
    path = os.path.join("backend/data/checkpoints/selector", version, "best.pt")
    if os.path.exists(path):
        return path
    return None

def resolve_scaler_path(version: str = "v1") -> Optional[str]:
    """
    Finds the scaler artifact.
    Path: backend/data/checkpoints/selector/{version}/scaler.joblib
    """
    path = os.path.join("backend/data/checkpoints/selector", version, "scaler.joblib")
    if os.path.exists(path):
        return path
    return None

def resolve_calibration_path(version: str = "v1") -> Optional[str]:
    """
    Finds the calibration artifact.
    Path: backend/data/checkpoints/selector/{version}/calibration.joblib
    """
    path = os.path.join("backend/data/checkpoints/selector", version, "calibration.joblib")
    if os.path.exists(path):
        return path
    return None

def resolve_leaderboard_path(as_of_date: str, version: str = "v1") -> Optional[str]:
    """
    Finds the leaderboard signals for the given date.
    Path: backend/data/signals/selector/{version}/{as_of_date}.parquet
    """
    base = os.path.join("backend/data/signals/selector", version)
    path = os.path.join(base, f"{as_of_date}.parquet")
    if os.path.exists(path):
        return path
    return None

def ensure_artifact_compatibility(priors_path: str, scaler_path: str, checkpoint_path: str):
    """
    Checks if artifacts are compatible (e.g. metadata matching).
    For now, we rely on them being in the same 'version' folder structure or
    having matching version IDs in metadata if loaded.
    """
    # Placeholder for strict hash checks
    pass
