from __future__ import annotations

import numpy as np


def round_toward_zero(x: float) -> int:
    return int(np.trunc(x))


def contracts_from_weights(weights: dict[str, float], capital: float, prices: dict[str, float], multipliers: dict[str, float], max_contracts: int) -> dict[str, int]:
    out = {}
    for k, w in weights.items():
        notional = w * capital
        c_notional = multipliers[k] * prices[k]
        qty = round_toward_zero(notional / c_notional)
        out[k] = int(max(-max_contracts, min(max_contracts, qty)))
    return out
