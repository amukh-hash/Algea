"""Universal data contracts for the Algaie orchestration layer.

All sleeves must yield ``TargetIntent`` objects to disk as JSON.  No sleeve
is permitted to make direct network calls to a broker.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ExecutionPhase(str, Enum):
    """Market micro-structure phases that determine when orders are routed.

    Asset-class-aware: equities route to NYSE auctions; futures route
    at cash open/close via MKT orders; commodities route at their
    exchange-specific settlement windows.
    """

    PREMARKET = "premarket"
    AUCTION_OPEN = "auction_open"      # Equity MOO — route at 09:26 (before 09:28 cutoff)
    FUTURES_OPEN = "futures_open"      # Futures MKT — route at 09:30:00 (exact cash open)
    INTRADAY = "intraday"              # Standard continuous routing
    COMMODITY_CLOSE = "commodity_close" # GC/SI/HG/CL at settlement-specific times
    AUCTION_CLOSE = "auction_close"    # Equity MOC — route at 15:48 (before 15:50 cutoff)
    FUTURES_CLOSE = "futures_close"    # Futures MKT — route at 15:59:50


class TargetIntent(BaseModel):
    """Immutable declaration of a sleeve's desired position.

    Parameters
    ----------
    asof_date : str
        Trading date in strict ``YYYY-MM-DD`` format.
    sleeve : str
        Originating sleeve name (e.g. ``"kronos"``, ``"mera"``, ``"vrp"``).
    symbol : str
        Instrument ticker (e.g. ``"ES"``, ``"SPY"``).
    asset_class : ``"EQUITY"`` | ``"FUTURE"`` | ``"OPTION"``
        Determines how exposure is measured (cash vs. notional × multiplier).
    target_weight : float
        Desired weight relative to total account equity, range ``[-2.0, 2.0]``.
    execution_phase : ExecutionPhase
        Which market phase this intent should be routed in.
    multiplier : float
        Contract multiplier (e.g. ``50.0`` for ``/ES`` futures, ``1.0`` for
        cash equities).  Must be positive.
    dte : int
        Days to expiration for options.  ``-1`` means not applicable.
    """

    asof_date: str
    sleeve: str
    symbol: str
    asset_class: Literal["EQUITY", "FUTURE", "OPTION"]
    target_weight: float = Field(..., ge=-2.0, le=2.0)
    execution_phase: ExecutionPhase
    multiplier: float = Field(default=1.0, gt=0.0)
    dte: int = Field(default=-1)

    @field_validator("asof_date")
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        """Enforce strict YYYY-MM-DD format."""
        datetime.strptime(v, "%Y-%m-%d")
        return v
