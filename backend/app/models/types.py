from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union

@dataclass
class DistributionForecast:
    """
    Standard output for probabilistic models (Teacher_O, etc).
    """
    model_name: str
    horizons: List[str] # ["1D", "3D"]

    # Quantiles: {"1D": {"0.05": -0.01, ...}}
    quantiles: Dict[str, Dict[str, float]]

    # Optional raw logits or bin edges for reconstruction
    logits: Optional[Dict[str, List[float]]] = None
    bin_edges: Optional[Dict[str, List[float]]] = None

    # Tail metrics (e.g. Expected Shortfall)
    tail_risk: Optional[Dict[str, float]] = None

    # Uncertainty scalar (e.g. entropy, width)
    uncertainty_scalar: Optional[float] = None

    metadata: Dict = field(default_factory=dict)
