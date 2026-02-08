import hashlib
import json
from typing import Dict, Any

def compute_hash(data: Dict[str, Any]) -> str:
    """Computes SHA256 hash of a dictionary (sorted keys)."""
    # Handle non-serializable types if needed (e.g. paths)
    s = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(s.encode()).hexdigest()

def get_schema_hash(columns: list) -> str:
    """Computes hash of column names/order."""
    return hashlib.md5(json.dumps(sorted(columns)).encode()).hexdigest()

def check_compatibility(required_id: str, current_id: str):
    if required_id != current_id:
        raise ValueError(f"Artifact Version ID mismatch: Required {required_id}, got {current_id}")

def get_priors_hash(priors_meta: Dict[str, Any]) -> str:
    """
    Computes hash of priors metadata (teacher version, adapter, etc.)
    """
    return compute_hash(priors_meta)

def get_selector_hash(selector_config: Dict[str, Any]) -> str:
    """
    Computes hash of selector architecture/hyperparams.
    """
    return compute_hash(selector_config)
