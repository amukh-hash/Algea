from __future__ import annotations


def pinball_loss(y_true: list[float], y_pred: list[float], q: float) -> float:
    if not y_true:
        return 0.0
    total = 0.0
    for yt, yp in zip(y_true, y_pred):
        diff = yt - yp
        total += max(q * diff, (q - 1) * diff)
    return total / len(y_true)
