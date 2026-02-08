
import hashlib
import json
import os
from typing import Dict, Any

def stable_hash(obj: Dict[str, Any]) -> str:
    """
    Computes stable SHA256 hash of a dictionary.
    """
    s = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12] # Short hash

def compute_feature_version(feature_spec: Dict, code_version: str) -> str:
    combined = {"spec": feature_spec, "code": code_version}
    return stable_hash(combined)

def compute_prior_version(prior_spec: Dict, code_version: str) -> str:
    combined = {"spec": prior_spec, "code": code_version}
    return stable_hash(combined)

def compute_universe_version(universe_rules: Dict, code_version: str) -> str:
    combined = {"rules": universe_rules, "code": code_version}
    return stable_hash(combined)

def write_metadata(sidecar_path: str, metadata: Dict) -> None:
    with open(sidecar_path, "w") as f:
        json.dump(metadata, f, indent=2)

def attach_provenance(df, provenance: Dict):
    # Pandas attrs are ephemeral if not saved carefully (parquet handles it? sometimes)
    # Better to verify columns exist if schema requires it.
    # This might just be a helper to verify columns match.
    return df
