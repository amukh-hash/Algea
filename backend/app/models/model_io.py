import torch
import json
import os
from dataclasses import asdict
from typing import Any, Tuple
from backend.app.models.signal_types import ModelMetadata

def save_model(model_state: Any, metadata: ModelMetadata, path: str):
    """
    Saves model state and metadata to a single file (using torch.save).
    """
    # Create directory if needed
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    payload = {
        "state_dict": model_state,
        "metadata": asdict(metadata)
    }
    torch.save(payload, path)

def load_model(path: str, device: str = "cpu") -> Tuple[Any, ModelMetadata]:
    """
    Loads model state and metadata.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model artifact not found: {path}")
        
    payload = torch.load(path, map_location=device)
    
    meta_dict = payload["metadata"]
    metadata = ModelMetadata(**meta_dict)
    
    return payload["state_dict"], metadata

def verify_preproc_compatibility(model_meta: ModelMetadata, active_preproc_id: str):
    if model_meta.preproc_id != active_preproc_id:
        raise ValueError(
            f"Model requires preproc {model_meta.preproc_id}, but active is {active_preproc_id}"
        )
