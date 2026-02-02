import hashlib
import json
from typing import Dict, Any

def compute_hash(data: Dict[str, Any]) -> str:
    """Computes SHA256 hash of a dictionary (sorted keys)."""
    s = json.dumps(data, sort_keys=True)
    return hashlib.sha256(s.encode()).hexdigest()

def get_schema_hash(columns: list) -> str:
    """Computes hash of column names/order."""
    return hashlib.md5(json.dumps(sorted(columns)).encode()).hexdigest()

def check_compatibility(required_id: str, current_id: str):
    if required_id != current_id:
        raise ValueError(f"Preproc ID mismatch: Required {required_id}, got {current_id}")
