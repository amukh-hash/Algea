import random
import numpy as np
from typing import List
from backend.app.core.config import OPTIONS_SEED

class TinyORunner:
    def __init__(self, model_path: str = None, device: str = "cpu", seed: int = OPTIONS_SEED):
        self.seed = seed
        self.device = device

    def predict_tokens(self, input_features: np.ndarray, steps: int = 5) -> List[int]:
        # Suggest 'steps' future tokens.
        # Assume tokens 0-50 represent return bins.

        rng = random.Random(self.seed + int(input_features.sum() * 100))

        tokens = []
        current_token = 25 # Center
        for _ in range(steps):
            # Random walk on tokens
            step = rng.choice([-1, 0, 1])
            current_token = max(0, min(50, current_token + step))
            tokens.append(current_token)

        return tokens
