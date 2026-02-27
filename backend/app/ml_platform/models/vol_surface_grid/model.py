from __future__ import annotations


class VolSurfaceGridForecaster:
    def __init__(self, scale: float = 0.05):
        self.scale = scale

    def forecast(self, grid_history: list[dict]) -> tuple[dict[str, float], float, float]:
        if not grid_history:
            return {}, 1.0, 1.0
        last = grid_history[-1]
        iv = {k: float(v) for k, v in last.get("iv", {}).items()}
        liq = {k: float(v) for k, v in last.get("liq", {}).items()}
        pred = {k: v + self.scale * (liq.get(k, 0.0) - 0.5) for k, v in sorted(iv.items())}
        unc = sum(abs(v - iv.get(k, 0.0)) for k, v in pred.items()) / max(len(pred), 1)
        drift = sum(abs(float(x.get("ret", 0.0))) for x in grid_history[-5:]) / max(len(grid_history[-5:]), 1)
        return pred, unc, drift
