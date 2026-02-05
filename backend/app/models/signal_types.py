from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import datetime

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

@dataclass
class RankScoreSignal:
    score: float
    rank: int
    rank_pct: float
    confidence: float # e.g. from uncertainty head or logic

@dataclass
class ChronosPriors:
    drift_20d: float
    vol_20d: float
    downside_q10_20d: float
    trend_conf_20d: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SelectorOutputs:
    score: float
    rank_score_signal: RankScoreSignal
    priors: ChronosPriors
    # Optional heads
    p_up: Optional[float] = None
    mu_ret: Optional[float] = None

    # Attached calibration
    expected_value: Optional[float] = None

    metadata: Dict[str, Any] = field(default_factory=dict)

# Canonical Leaderboard Schema
LEADERBOARD_SCHEMA = {
    # Keys
    "as_of_date": "str",   # YYYY-MM-DD
    "symbol": "str",

    # Core Ranking
    "score": "float",
    "rank": "int",
    "rank_pct": "float",

    # Calibration / Value
    "ev_10d": "float",  # Expected Value over 10 days

    # Teacher Priors (Attached)
    "prior_drift_20d": "float",
    "prior_vol_20d": "float",
    "prior_downside_q10_20d": "float",
    "prior_trend_conf_20d": "float",

    # Metadata columns (can be uniform across file)
    "selector_checkpoint_id": "str",
    "selector_version": "str",
    "selector_scaler_version": "str",
    "calibration_version": "str",
    "teacher_model_id": "str",
    "teacher_adapter_id": "str",
    "teacher_codec_version": "str",
    "feature_contract_hash": "str",
    "preproc_version": "str"
}
