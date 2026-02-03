from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime

@dataclass
class IVSnapshot:
    ticker: str
    timestamp: datetime
    dte: int
    atm_iv: float
    iv_rank: Optional[float] = None
    iv_percentile: Optional[float] = None
    metadata: dict = field(default_factory=dict)

@dataclass
class LiquidityMetrics:
    ticker: str
    timestamp: datetime
    volume_24h: float = 0.0
    oi_total: float = 0.0
    spread_width_pct: float = 0.0

@dataclass
class OptionRow:
    strike: float
    option_type: str # "call" or "put"
    bid: float
    ask: float
    iv: float
    delta: Optional[float] = None
    oi: Optional[int] = None
    volume: Optional[int] = None

@dataclass
class OptionChainSnapshot:
    ticker: str
    timestamp: datetime
    expiry: str # YYYY-MM-DD
    dte: int
    rows: List[OptionRow] = field(default_factory=list)
