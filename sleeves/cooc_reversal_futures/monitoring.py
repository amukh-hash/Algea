from __future__ import annotations


def calibration_error(pred_mu: float, realized: float) -> float:
    return realized - pred_mu


def slippage_bps(expected: float, realized: float) -> float:
    return (realized - expected) * 1e4
