"""
Derivatives position data structures — OptionLeg, DerivativesPosition, DerivativesPositionFrame.

v2: adds entry_iv, entry_mid, entry_underlying, entry marks,
    exit_reason, exit_dt, risk_state_at_entry.
"""
from __future__ import annotations

import enum
import warnings
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, List, Optional

import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════════════

class StructureType(enum.Enum):
    PUT_CREDIT_SPREAD = "PUT_CREDIT_SPREAD"
    CALL_CREDIT_SPREAD = "CALL_CREDIT_SPREAD"
    IRON_CONDOR = "IRON_CONDOR"


class OptionSide(enum.Enum):
    BUY = "buy"
    SELL = "sell"


# ═══════════════════════════════════════════════════════════════════════════
# Leg (v2 — includes entry IV + marks)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class OptionLeg:
    option_type: str       # "call" | "put"
    strike: float
    qty: int               # positive = long, negative = short
    side: str              # "buy" | "sell"
    entry_price_mid: float
    delta: float = 0.0
    gamma: float = 0.0
    vega: float = 0.0
    theta: float = 0.0
    # v2 fields ────────────────────────────────────────────────────────────
    entry_iv: float = 0.0            # implied vol at entry (for scenario repricing)
    entry_mid: float = 0.0           # same as entry_price_mid; explicit alias
    entry_underlying: float = 0.0    # underlying price at trade time
    entry_bid: float = 0.0
    entry_ask: float = 0.0
    entry_dt: Optional[datetime] = None


# ═══════════════════════════════════════════════════════════════════════════
# Position (v2 — exit reason + risk state snapshot)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DerivativesPosition:
    underlying: str
    structure_type: StructureType
    expiry: date
    legs: List[OptionLeg]
    premium_collected: float             # net credit received
    max_loss: float                      # worst-case loss (always positive)
    multiplier: int = 100

    # Aggregate greeks at entry
    delta: float = 0.0
    gamma: float = 0.0
    vega: float = 0.0
    theta: float = 0.0

    # Sizing metadata
    risk_budget_used: float = 0.0
    target_vol_scale: float = 1.0

    # Timestamps
    entry_dt: Optional[datetime] = None
    planned_exit_dt: Optional[datetime] = None

    # Strategy tags
    strategy_tags: Dict[str, Any] = field(default_factory=dict)

    # Tracking
    position_id: str = ""
    is_open: bool = True
    realized_pnl: float = 0.0
    current_mark: float = 0.0

    # v2 fields ────────────────────────────────────────────────────────────
    exit_reason: str = ""
    exit_dt: Optional[datetime] = None
    risk_state_at_entry: Dict[str, Any] = field(default_factory=dict)

    @property
    def dte(self) -> int:
        """Days to expiry from today (or entry)."""
        ref = self.entry_dt.date() if self.entry_dt else date.today()
        return (self.expiry - ref).days

    def net_greeks(self) -> Dict[str, float]:
        """Aggregate greeks across all legs."""
        d = g = v = t = 0.0
        for leg in self.legs:
            d += leg.delta * leg.qty * self.multiplier
            g += leg.gamma * leg.qty * self.multiplier
            v += leg.vega * leg.qty * self.multiplier
            t += leg.theta * leg.qty * self.multiplier
        return {"delta": d, "gamma": g, "vega": v, "theta": t}


# ═══════════════════════════════════════════════════════════════════════════
# Migration helper (v1 → v2)
# ═══════════════════════════════════════════════════════════════════════════

def migrate_positions_v1_to_v2(
    positions: List[DerivativesPosition],
    audit_warnings: Optional[List[str]] = None,
) -> List[DerivativesPosition]:
    """Ensure v1 positions are compatible with v2 schema.

    Sets deterministic defaults for any missing v2 fields and appends
    warnings to *audit_warnings* for downstream audit artifacts.
    """
    if audit_warnings is None:
        audit_warnings = []
    for pos in positions:
        if not pos.exit_reason:
            pass  # already default ""
        if not pos.risk_state_at_entry:
            pos.risk_state_at_entry = {"migrated_from_v1": True}
            audit_warnings.append(
                f"Position {pos.position_id}: risk_state_at_entry missing; set v1 migration default."
            )
        for leg in pos.legs:
            if leg.entry_iv == 0.0 and leg.entry_price_mid > 0.0:
                # Cannot recover IV; emit warning
                audit_warnings.append(
                    f"Position {pos.position_id} leg K={leg.strike}: entry_iv=0; "
                    "scenario repricing will use 0.20 fallback."
                )
    return positions


# ═══════════════════════════════════════════════════════════════════════════
# Position Frame (collection)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DerivativesPositionFrame:
    """Container for multiple DerivativesPosition objects with summary helpers."""

    positions: List[DerivativesPosition] = field(default_factory=list)

    def add(self, position: DerivativesPosition) -> None:
        self.positions.append(position)

    @property
    def open_positions(self) -> List[DerivativesPosition]:
        return [p for p in self.positions if p.is_open]

    def total_max_loss(self) -> float:
        return sum(p.max_loss * p.multiplier for p in self.open_positions)

    def total_premium_collected(self) -> float:
        return sum(p.premium_collected * p.multiplier for p in self.open_positions)

    def aggregate_greeks(self) -> Dict[str, float]:
        agg = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}
        for p in self.open_positions:
            g = p.net_greeks()
            for k in agg:
                agg[k] += g[k]
        return agg

    def positions_by_underlying(self) -> Dict[str, List[DerivativesPosition]]:
        by_und: Dict[str, List[DerivativesPosition]] = {}
        for p in self.open_positions:
            by_und.setdefault(p.underlying, []).append(p)
        return by_und

    def positions_by_expiry(self) -> Dict[date, List[DerivativesPosition]]:
        by_exp: Dict[date, List[DerivativesPosition]] = {}
        for p in self.open_positions:
            by_exp.setdefault(p.expiry, []).append(p)
        return by_exp

    def to_summary_df(self) -> pd.DataFrame:
        """Flatten open positions into a summary DataFrame."""
        rows = []
        for p in self.open_positions:
            rows.append({
                "position_id": p.position_id,
                "underlying": p.underlying,
                "structure_type": p.structure_type.value,
                "expiry": p.expiry,
                "premium_collected": p.premium_collected,
                "max_loss": p.max_loss,
                "delta": p.delta,
                "gamma": p.gamma,
                "vega": p.vega,
                "theta": p.theta,
                "risk_budget_used": p.risk_budget_used,
                "entry_dt": p.entry_dt,
                "is_open": p.is_open,
                "current_mark": p.current_mark,
                "realized_pnl": p.realized_pnl,
                "exit_reason": p.exit_reason,
            })
        return pd.DataFrame(rows)
