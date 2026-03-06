"""Static SPY-relative beta map for cross-sleeve risk netting.

Commodities (GC, SI, HG, CL) and FX (6E, 6J, 6B, 6A) carry near-zero
equity beta and are excluded from the equity-beta netting engine.

V1: static map.  V2 (future): rolling 60-day beta from daily returns.
"""
from __future__ import annotations

# Beta-to-SPY for futures and ETFs used across sleeves.
# Zero-beta instruments are explicitly mapped to 0.0 so the netting
# engine never accidentally assigns them equity correlation.
_SPY_BETA: dict[str, float] = {
    # ── Equity index futures ─────────────────────────────────────────
    "ES":  1.0,
    "NQ":  1.2,
    "YM":  1.0,
    "RTY": 1.1,
    "MES": 1.0,
    "MNQ": 1.2,
    "MYM": 1.0,
    "M2K": 1.1,
    # ── Equity ETFs ──────────────────────────────────────────────────
    "SPY": 1.0,
    "QQQ": 1.2,
    "IWM": 1.0,
    "XLF": 1.0,
    "XLK": 1.1,
    "XLE": 0.8,
    "XLI": 1.0,
    "XLY": 1.0,
    "XLP": 0.6,
    "SMH": 1.3,
    "KRE": 1.0,
    "VNQ": 0.7,
    "XRT": 1.0,
    "ITB": 1.0,
    "XBI": 0.9,
    "ARKK": 1.4,
    # ── Zero-beta: Commodities ───────────────────────────────────────
    "GC":  0.0,
    "SI":  0.0,
    "HG":  0.0,
    "CL":  0.0,
    "GLD": 0.0,
    "USO": 0.0,
    "GDXJ": 0.3,   # Gold miners have *some* equity beta
    "XOP": 0.5,    # Oil & gas equities have moderate equity beta
    "TAN": 0.8,    # Solar equities - moderate equity beta
    # ── Zero-beta: FX ────────────────────────────────────────────────
    "6E":  0.0,
    "6J":  0.0,
    "6B":  0.0,
    "6A":  0.0,
    # ── Zero-beta: Rates ─────────────────────────────────────────────
    "ZN":  0.0,
    "ZB":  0.0,
    "TLT": 0.0,
    "JNK": 0.4,    # High-yield has moderate equity beta
    # ── Volatility ───────────────────────────────────────────────────
    "VIX": -1.0,   # Inverse equity beta
}

# Default beta for unknown symbols (conservative estimate)
_DEFAULT_BETA: float = 1.0


def get_spy_beta(symbol: str) -> float:
    """Return the SPY-relative beta for a given symbol.

    Unknown symbols default to 1.0 (conservative assumption —
    treat as fully correlated with equities).
    """
    return _SPY_BETA.get(symbol, _DEFAULT_BETA)
