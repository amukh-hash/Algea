from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from backend.app.core.schemas import ExecutionPhase


@dataclass(frozen=True)
class SleeveDerivationPolicy:
    sleeve: str
    asset_class: str
    default_execution_phase: ExecutionPhase
    default_multiplier: float = 1.0


POLICY_VERSION = "intent-derivation-policy.v1"


# Central policy owner for target->intent derivation defaults.
SLEEVE_POLICIES: dict[str, SleeveDerivationPolicy] = {
    "core": SleeveDerivationPolicy("core", "EQUITY", ExecutionPhase.INTRADAY, 1.0),
    "vrp": SleeveDerivationPolicy("vrp", "EQUITY", ExecutionPhase.INTRADAY, 1.0),
    "selector": SleeveDerivationPolicy("selector", "EQUITY", ExecutionPhase.INTRADAY, 1.0),
    "futures_overnight": SleeveDerivationPolicy("futures_overnight", "FUTURE", ExecutionPhase.FUTURES_OPEN, 1.0),
    "statarb": SleeveDerivationPolicy("statarb", "EQUITY", ExecutionPhase.INTRADAY, 1.0),
}


# Explicit multiplier source for futures symbols.
FUTURES_MULTIPLIER_MAP: dict[str, float] = {
    "ES": 50.0,
    "NQ": 20.0,
    "YM": 5.0,
    "RTY": 50.0,
    "CL": 1000.0,
    "GC": 100.0,
    "SI": 5000.0,
    "HG": 25000.0,
}


def resolve_policy(sleeve: str, row: dict[str, Any]) -> SleeveDerivationPolicy:
    key = str(sleeve).strip().lower()
    if key not in SLEEVE_POLICIES:
        raise KeyError(f"No derivation policy configured for sleeve='{sleeve}'")
    return SLEEVE_POLICIES[key]


def resolve_multiplier(sleeve: str, symbol: str, row: dict[str, Any]) -> float:
    policy = resolve_policy(sleeve, row)

    # Override precedence:
    # 1) explicit row multiplier
    # 2) futures symbol mapping for futures_overnight
    # 3) sleeve default
    if row.get("multiplier") is not None:
        return float(row.get("multiplier"))

    if policy.sleeve == "futures_overnight":
        sym = str(symbol).strip().upper()
        if sym not in FUTURES_MULTIPLIER_MAP:
            raise RuntimeError(
                f"UNTRANSLATABLE_TARGET: missing multiplier mapping for futures symbol '{sym}'"
            )
        return float(FUTURES_MULTIPLIER_MAP[sym])

    return float(policy.default_multiplier)
