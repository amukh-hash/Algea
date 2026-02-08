from dataclasses import dataclass
from typing import Optional
from enum import Enum

class ActionType(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    NO_NEW_RISK = "NO_NEW_RISK"
    REDUCE = "REDUCE"
    LIQUIDATE = "LIQUIDATE"

@dataclass
class RiskDecision:
    ticker: str
    action: ActionType
    quantity: float # Signed? Or use Action to determine sign? Usually absolute qty + Action.
    reason: str
    confidence: float = 1.0
    max_position_size: Optional[float] = None # Cap
    target_weight: Optional[float] = None # For HRP/Portfolio sizing
