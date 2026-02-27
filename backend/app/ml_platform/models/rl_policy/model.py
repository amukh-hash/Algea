from __future__ import annotations

import math


class RLPolicyModel:
    """Lightweight deterministic inference wrapper (transformer+MLP stub)."""

    def __init__(self, hidden_size: int = 32):
        self.hidden_size = hidden_size

    def act(self, state_vector: list[float]) -> tuple[float, bool, float]:
        if not state_vector:
            return 0.0, True, 0.0
        mean_signal = sum(state_vector) / len(state_vector)
        logit = max(-10.0, min(10.0, mean_signal))
        multiplier = 1.0 / (1.0 + math.exp(-logit))
        confidence = min(1.0, abs(mean_signal))
        veto = multiplier < 0.05
        return float(multiplier), veto, confidence
