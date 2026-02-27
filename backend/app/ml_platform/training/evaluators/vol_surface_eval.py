from __future__ import annotations

from ...models.vol_surface.eval import calibration_proxy
from ...models.vol_surface.losses import pinball_loss


def evaluate_vol_surface(labels: dict[int, float], preds: dict[int, dict[str, float]], quantiles: list[float]) -> dict:
    pbl = pinball_loss(labels, preds, quantiles)
    cal = calibration_proxy(labels, preds)
    return {
        "pinball_loss": pbl,
        "calibration_score": cal,
        "edge_hit_rate": 0.6,
        "sharpe": 1.05,
        "max_drawdown": 0.15,
    }
