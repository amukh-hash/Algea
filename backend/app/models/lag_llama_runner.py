import random
import numpy as np
from typing import Dict
from backend.app.models.types import DistributionForecast
from backend.app.core.config import OPTIONS_SEED

class LagLlamaRunner:
    def __init__(self, model_path: str = None, device: str = "cpu", seed: int = OPTIONS_SEED):
        self.seed = seed
        self.device = device

    def predict(self, input_features: np.ndarray) -> DistributionForecast:
        # Mock uncertainty
        rng = random.Random(self.seed + int(input_features.sum() * 100))

        # Determine uncertainty level (0 to 1)
        # Randomly high or low
        uncertainty = rng.random()

        # If high uncertainty, wide quantiles
        width = 0.01 + 0.05 * uncertainty
        mu = 0.0

        quantiles = {}
        for h in ["1D", "3D"]:
            quantiles[h] = {
                "0.05": mu - 2*width,
                "0.50": mu,
                "0.95": mu + 2*width
            }

        return DistributionForecast(
            model_name="LagLlama_Mock",
            horizons=["1D", "3D"],
            quantiles=quantiles,
            uncertainty_scalar=uncertainty,
            metadata={"seed": self.seed, "mock": True}
        )
