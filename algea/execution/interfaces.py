from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal, Optional

import pandas as pd

Action = Literal["open", "close", "hold"]


@dataclass(frozen=True)
class SignalFrame:
    frame: pd.DataFrame


@dataclass(frozen=True)
class ExecutionDecision:
    action: Action
    instrument: str
    quantity: float
    reason: str


@dataclass(frozen=True)
class ExecutionContext:
    asof: date
    prices: pd.DataFrame
    iv_snapshot: Optional[pd.DataFrame] = None
