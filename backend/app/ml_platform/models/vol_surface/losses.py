from __future__ import annotations


def pinball(y: float, yhat: float, q: float) -> float:
    d = y - yhat
    return max(q * d, (q - 1) * d)


def pinball_loss(labels: dict[int, float], preds: dict[int, dict[str, float]], quantiles: list[float]) -> float:
    vals = []
    for t, y in labels.items():
        for q in quantiles:
            vals.append(pinball(float(y), float(preds[int(t)][f"{q:.2f}"]), q))
    return sum(vals) / max(len(vals), 1)
