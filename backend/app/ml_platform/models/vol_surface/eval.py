from __future__ import annotations


def calibration_proxy(labels: dict[int, float], preds: dict[int, dict[str, float]]) -> float:
    errs = []
    for t, y in labels.items():
        med = float(preds[int(t)].get("0.50", 0.0))
        errs.append(abs(float(y) - med))
    mae = sum(errs) / max(len(errs), 1)
    return max(0.0, 1.0 - mae)
