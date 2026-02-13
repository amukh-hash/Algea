from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContractSpec:
    symbol: str
    multiplier: float
    tick_size: float
    ib_symbol: str


CONTRACT_MASTER = {
    "ES": ContractSpec("ES", 50.0, 0.25, "ES"),
    "NQ": ContractSpec("NQ", 20.0, 0.25, "NQ"),
    "YM": ContractSpec("YM", 5.0, 1.0, "YM"),
    "RTY": ContractSpec("RTY", 50.0, 0.1, "RTY"),
    "MES": ContractSpec("MES", 5.0, 0.25, "MES"),
    "MNQ": ContractSpec("MNQ", 2.0, 0.25, "MNQ"),
    "MYM": ContractSpec("MYM", 0.5, 1.0, "MYM"),
    "M2K": ContractSpec("M2K", 5.0, 0.1, "M2K"),
}
