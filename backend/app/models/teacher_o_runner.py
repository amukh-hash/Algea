import random
import numpy as np
from typing import List, Dict
from backend.app.models.types import DistributionForecast
from backend.app.core.config import OPTIONS_SEED

class TeacherORunner:
    def __init__(self, model_path: str = None, device: str = "cpu", seed: int = OPTIONS_SEED):
        self.seed = seed
        self.device = device
        self.model_path = model_path
        # Mock initialization

    def predict_distribution(self, input_features: np.ndarray) -> DistributionForecast:
        # Input: (batch, time, feats) or similar.
        # We assume batch size 1 for now or handle batch.
        # Mock logic: produce a distribution centered slightly positive or negative based on seed + random.

        # Use input features sum to "condition" the output deterministically if needed
        # But simplistic mock:

        seed_offset = 0
        if input_features is not None:
            seed_offset = int(input_features.sum() * 100)

        rng = random.Random(self.seed + seed_offset)

        horizons = ["1D", "3D"]
        quantiles = {}
        q_levels = ["0.05", "0.25", "0.50", "0.75", "0.95"]

        for h in horizons:
            # Generate a normal distribution
            mu = rng.gauss(0.001, 0.005) # Slight upward drift
            sigma = rng.gauss(0.015, 0.002) # Volatility

            # Simple inverse CDF approximation (or just sorted samples)
            samples = [rng.gauss(mu, sigma) for _ in range(100)]
            samples.sort()

            q_dict = {}
            for q in q_levels:
                idx = int(float(q) * 100)
                q_dict[q] = samples[idx]

            quantiles[h] = q_dict

        return DistributionForecast(
            model_name="Teacher_O_Mock",
            horizons=horizons,
            quantiles=quantiles,
            metadata={"seed": self.seed, "mock": True}
        )

    def verify_tokens(self, input_features: np.ndarray, tokens: List[int]) -> List[bool]:
        # Mock verification: accept if token is within reasonable range (e.g. 10-40)
        # In real spec decode, we check P(token) > threshold or sample match
        rng = random.Random(self.seed + int(input_features.sum() * 100))

        # Simulate an accept mask
        # e.g. 90% acceptance rate
        results = []
        for t in tokens:
            accept = rng.random() < 0.9
            results.append(accept)
        return results
