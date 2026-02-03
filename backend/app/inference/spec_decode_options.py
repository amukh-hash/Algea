from typing import List, Dict
import numpy as np
from backend.app.models.teacher_o_runner import TeacherORunner
from backend.app.models.tiny_o_runner import TinyORunner

class SpecDecodeEngine:
    def __init__(self, tiny: TinyORunner, teacher: TeacherORunner):
        self.tiny = tiny
        self.teacher = teacher

    def generate(self, input_features: np.ndarray, steps: int = 5) -> Dict:
        """
        Run speculative decoding generation.
        """
        # 1. Propose
        proposed_tokens = self.tiny.predict_tokens(input_features, steps=steps)

        # 2. Verify
        # In real spec decode, we pass the prefix + proposal to teacher
        valid_mask = self.teacher.verify_tokens(input_features, proposed_tokens)

        # 3. Accept/Cut
        final_tokens = []
        for token, valid in zip(proposed_tokens, valid_mask):
            if valid:
                final_tokens.append(token)
            else:
                # Fallback: Teacher would regenerate from here.
                # For mock, we just stop or fill with dummy.
                break

        # Metrics
        accept_rate = len(final_tokens) / len(proposed_tokens) if proposed_tokens else 0.0

        return {
            "tokens": final_tokens,
            "proposed": proposed_tokens,
            "accept_rate": accept_rate,
            "fallback_triggered": len(final_tokens) < len(proposed_tokens)
        }
