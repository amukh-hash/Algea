from __future__ import annotations

from .patches import patch_sequence


class VolSurfaceForecaster:
    def __init__(self, hidden_size: int = 16):
        self.hidden_size = hidden_size

    def forecast(self, history: dict[int, list[dict]], quantiles: list[float]) -> dict[int, dict[str, float]]:
        out: dict[int, dict[str, float]] = {}
        for tenor, rows in history.items():
            seq = [float(r.get("rv_hist_20", 0.0)) for r in rows]
            patches = patch_sequence(seq, patch_len=4)
            base = sum(sum(p) / max(len(p), 1) for p in patches) / max(len(patches), 1)
            out[int(tenor)] = {f"{q:.2f}": float(base * (1 + (q - 0.5) * 0.3)) for q in quantiles}
        return out
