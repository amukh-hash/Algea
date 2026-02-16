from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Tuple


RollCycle = Literal["quarterly", "monthly", "custom"]

_INDEX_QUARTERS: Tuple[str, ...] = ("H", "M", "U", "Z")
_ALL_MONTHS: Tuple[str, ...] = ("F", "G", "H", "J", "K", "M", "N", "Q", "U", "V", "X", "Z")
_METALS_CUSTOM: Tuple[str, ...] = ("H", "K", "N", "U", "Z")   # SI, HG
_GOLD_CUSTOM: Tuple[str, ...] = ("G", "J", "M", "Q", "V", "Z")  # GC


@dataclass(frozen=True)
class ContractSpec:
    symbol: str
    multiplier: float
    tick_size: float
    ib_symbol: str
    # --- extended fields (defaults preserve backward compat) ---
    tick_value: float = 0.0                        # multiplier * tick_size
    currency: str = "USD"
    exchange: str = ""                             # expected IBKR routing exchange
    roll_cycle: RollCycle = "quarterly"
    roll_month_codes: Tuple[str, ...] = _INDEX_QUARTERS
    first_notice_offset_days: int = 0              # days before expiry
    last_trade_offset_days: int = 0


def _spec(
    symbol: str,
    multiplier: float,
    tick_size: float,
    ib_symbol: str,
    *,
    exchange: str = "",
    roll_cycle: RollCycle = "quarterly",
    roll_months: Tuple[str, ...] = _INDEX_QUARTERS,
    first_notice: int = 0,
    last_trade: int = 0,
) -> ContractSpec:
    return ContractSpec(
        symbol=symbol,
        multiplier=multiplier,
        tick_size=tick_size,
        ib_symbol=ib_symbol,
        tick_value=multiplier * tick_size,
        currency="USD",
        exchange=exchange,
        roll_cycle=roll_cycle,
        roll_month_codes=roll_months,
        first_notice_offset_days=first_notice,
        last_trade_offset_days=last_trade,
    )


CONTRACT_MASTER = {
    # ── Equity index futures ────────────────────────────────────────────
    "ES":  _spec("ES",  50.0,   0.25,  "ES",  exchange="CME",   roll_cycle="quarterly"),
    "NQ":  _spec("NQ",  20.0,   0.25,  "NQ",  exchange="CME",   roll_cycle="quarterly"),
    "YM":  _spec("YM",  5.0,    1.0,   "YM",  exchange="ECBOT", roll_cycle="quarterly"),
    "RTY": _spec("RTY", 50.0,   0.1,   "RTY", exchange="CME",   roll_cycle="quarterly"),
    # ── Equity index micros ─────────────────────────────────────────────
    "MES": _spec("MES", 5.0,    0.25,  "MES", exchange="CME",   roll_cycle="quarterly"),
    "MNQ": _spec("MNQ", 2.0,    0.25,  "MNQ", exchange="CME",   roll_cycle="quarterly"),
    "MYM": _spec("MYM", 0.5,    1.0,   "MYM", exchange="ECBOT", roll_cycle="quarterly"),
    "M2K": _spec("M2K", 5.0,    0.1,   "M2K", exchange="CME",   roll_cycle="quarterly"),
    # ── Energy ──────────────────────────────────────────────────────────
    "CL":  _spec("CL",  1000.0, 0.01,  "CL",  exchange="NYMEX", roll_cycle="monthly",
                 roll_months=_ALL_MONTHS, first_notice=1, last_trade=3),
    # ── Metals ──────────────────────────────────────────────────────────
    "GC":  _spec("GC",  100.0,  0.10,  "GC",  exchange="COMEX", roll_cycle="custom",
                 roll_months=_GOLD_CUSTOM, first_notice=2, last_trade=3),
    "SI":  _spec("SI",  5000.0, 0.005, "SI",  exchange="COMEX", roll_cycle="custom",
                 roll_months=_METALS_CUSTOM, first_notice=2, last_trade=3),
    "HG":  _spec("HG",  25000.0, 0.0005, "HG", exchange="COMEX", roll_cycle="custom",
                 roll_months=_METALS_CUSTOM, first_notice=2, last_trade=3),
    # ── Rates ───────────────────────────────────────────────────────────
    "ZN":  _spec("ZN",  1000.0,  0.015625, "ZN", exchange="ECBOT", roll_cycle="quarterly"),
    "ZB":  _spec("ZB",  1000.0,  0.03125,  "ZB", exchange="ECBOT", roll_cycle="quarterly"),
    # ── FX ──────────────────────────────────────────────────────────────
    "6E":  _spec("6E",  125000.0,    0.00005,    "6E", exchange="CME", roll_cycle="quarterly"),
    "6J":  _spec("6J",  12500000.0,  0.0000005,  "6J", exchange="CME", roll_cycle="quarterly"),
    "6B":  _spec("6B",  62500.0,     0.0001,     "6B", exchange="CME", roll_cycle="quarterly"),
    "6A":  _spec("6A",  100000.0,    0.0001,     "6A", exchange="CME", roll_cycle="quarterly"),
}
