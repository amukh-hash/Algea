from __future__ import annotations

import numpy as np

from .backends.mlp_baseline import Mlp_baselineBackend


def train_model(x: np.ndarray, y: np.ndarray, backend: str = "mlp_baseline") -> object:
    model = Mlp_baselineBackend()
    model.fit(x, y)
    return model
