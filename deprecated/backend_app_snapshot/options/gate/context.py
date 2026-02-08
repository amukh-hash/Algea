from dataclasses import dataclass, field
from typing import Dict, Optional, Any
from datetime import datetime
from backend.app.models.signal_types import ModelSignal, ChronosPriors
from backend.app.options.data.types import IVSnapshot, OptionChainSnapshot

@dataclass
class OptionsContext:
    ticker: str
    timestamp: datetime
    
    # Inputs
    underlying_price: float
    student_signal: ModelSignal
    iv_snapshot: Optional[IVSnapshot] = None
    chain_snapshot: Optional[OptionChainSnapshot] = None
    
    # Overlay Inputs
    teacher_priors: Optional[ChronosPriors] = None
    in_equity_selection: bool = False
    
    # Context
    breadth: Dict[str, float] = field(default_factory=dict) # {"ad_line": ..., "bpi": ...}
    portfolio_state: Dict[str, Any] = field(default_factory=dict) # Positions, risk usage
    posture: str = "NORMAL" # NORMAL, CAUTIOUS, DEFENSIVE
