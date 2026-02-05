from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from datetime import datetime

class OptionsMode(str, Enum):
    OFF = "off"
    MONITOR = "monitor"
    PAPER = "paper"
    LIVE = "live"

class GateReasonCode(str, Enum):
    PASS = "PASS"
    REJECT_REGIME = "REJECT_REGIME"   # Broad market/Student regime bad
    REJECT_IV = "REJECT_IV"           # IV rank too low/high
    REJECT_LIQUIDITY = "REJECT_LIQUIDITY"
    REJECT_MODEL = "REJECT_MODEL"     # Teacher/Lag-Llama veto
    REJECT_STRIKES = "REJECT_STRIKES" # No valid strikes found
    UNCERTAINTY_HIGH = "UNCERTAINTY_HIGH" # High uncertainty from Teacher
    TREND_WEAK = "TREND_WEAK" # Prior trend weak

@dataclass
class GateDecision:
    should_trade: bool
    reason_code: GateReasonCode
    reason_desc: str
    metadata: Dict = field(default_factory=dict)

@dataclass
class SpreadCandidate:
    underlying_ticker: str
    expiry_date: str # YYYY-MM-DD
    dte: int
    strategy_type: str = "put_credit_spread"
    
    # Legs
    short_strike: float = 0.0
    long_strike: float = 0.0
    
    # Pricing
    short_price: float = 0.0
    long_price: float = 0.0
    net_credit: float = 0.0
    
    # Greeks/Risk
    width: float = 0.0
    max_loss: float = 0.0
    max_profit: float = 0.0
    risk_reward_ratio: float = 0.0
    
    # Model
    prob_profit: float = 0.0
    expected_value: float = 0.0

@dataclass
class OptionsDecision:
    """The final output from the OptionsPod before execution."""
    action: str # "OPEN", "CLOSE", "ADJUST", "NO_OP"
    candidate: Optional[SpreadCandidate] = None
    quantity: int = 0
    reason: str = ""
    timestamp: Optional[datetime] = None

@dataclass
class OptionsPosition:
    ticker: str
    expiry: str
    short_strike: float
    long_strike: float
    quantity: int
    entry_price: float
    current_price: float = 0.0
    pnl: float = 0.0
    open_timestamp: Optional[datetime] = None
