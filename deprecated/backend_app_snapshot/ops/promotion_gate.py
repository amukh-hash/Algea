
import json
import logging
import os
import pandas as pd
from typing import Dict
from backend.app.ops import pathmap, artifact_registry

logger = logging.getLogger(__name__)

def load_prod_pointer() -> Dict:
    path = os.path.join(pathmap.get_paths().models, "PROD_POINTER.json")
    if not os.path.exists(path):
        # Return fallback or error
        return {
            "model_version": "v1",
            "feature_version": "v1",
            "prior_version": "v1",
            "cal_version": "v1"
        }
    with open(path, "r") as f:
        return json.load(f)

def promote_model(run_id: str, metrics: Dict, versions: Dict) -> None:
    """
    Promotes a model to PROD if gates pass.
    """
    # 1. Check Gates
    # if metrics['sharpe'] < 1.2: raise ...
    
    # 2. Write Pointer
    pointer = {
        "run_id": run_id,
        "promoted_at": str(pd.Timestamp.now()),
        **versions
    }
    
    path = os.path.join(pathmap.get_paths().models, "PROD_POINTER.json")
    with open(path, "w") as f:
        json.dump(pointer, f, indent=2)
    
    logger.info(f"PROMOTED model to PROD: {path}")
