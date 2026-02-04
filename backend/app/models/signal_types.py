from dataclasses import dataclass
from typing import List, Dict, Optional, Union
import numpy as np

@dataclass
class ModelMetadata:
    model_version: str
    preproc_id: str
    training_start: str
    training_end: str
    # Add other fields as needed

@dataclass
class ModelSignal:
    horizons: List[str] # ["1D", "3D"]
    
    # Probabilistic outputs
    # Quantiles: e.g., {"1D": {"0.05": -0.02, "0.5": 0.01, ...}}
    quantiles: Optional[Dict[str, Dict[str, float]]] = None
    
    # Logits/Distribution (for distillation or advanced risk)
    logits: Optional[Dict[str, List[float]]] = None
    bin_edges: Optional[Dict[str, List[float]]] = None
    
    # Point estimates / Direction
    direction_probs: Optional[Dict[str, float]] = None # Prob(Up)
    uncertainty: Optional[Dict[str, float]] = None # Entropy or Width
    
    # Metadata context
    metadata: Optional[ModelMetadata] = None
