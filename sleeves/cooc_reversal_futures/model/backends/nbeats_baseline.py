from __future__ import annotations

import numpy as np


class Nbeats_baselineBackend:
    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self.mu_ = float(np.nanmean(y)) if len(y) else 0.0
        self.sig_ = float(np.nanstd(y) + 1e-6)

    def predict(self, x: np.ndarray) -> dict[str, np.ndarray]:
        n = x.shape[0]
        return {
            "mu": np.full(n, getattr(self, "mu_", 0.0)),
            "sigma": np.full(n, getattr(self, "sig_", 1.0)),
        }
