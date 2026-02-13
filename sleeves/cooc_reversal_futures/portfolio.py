from __future__ import annotations

import numpy as np


def utility_signal(mu: np.ndarray, sigma: np.ndarray, cost_hat: np.ndarray | None = None) -> np.ndarray:
    cost = np.zeros_like(mu) if cost_hat is None else cost_hat
    return (mu - cost) / np.clip(sigma, 1e-6, None) ** 2


def market_neutral_weights(signal: np.ndarray, gross: float, net_cap: float = 0.05) -> np.ndarray:
    centered = signal - np.nanmean(signal)
    denom = np.sum(np.abs(centered))
    w = np.zeros_like(centered) if denom == 0 else centered / denom * gross
    net = np.sum(w)
    if abs(net) > net_cap:
        w -= net / len(w)
    return w
