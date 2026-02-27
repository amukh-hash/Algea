from __future__ import annotations

import math
import torch
from algea.models.rl.td3 import TD3Actor


class RLPolicyModel:
    """TD3 Execution Agent Wrapper"""

    def __init__(self, hidden_size: int = 32):
        self.actor = TD3Actor(state_dim=256, action_dim=2)
        self.actor.eval()

    def act(self, state_embedding: torch.Tensor | list[float]) -> tuple[float, bool, float]:
        """
        Receives dense embeddings directly from the Spatial-Temporal Transformer.
        """
        if isinstance(state_embedding, list):
            # Fallback legacy mock mapping
            state_tensor = torch.zeros(1, 256)
        else:
            state_tensor = state_embedding

        if state_tensor is None or state_tensor.numel() == 0:
            return 0.0, True, 0.0
            
        with torch.no_grad():
            action = self.actor(state_tensor)
        
        act_val = action[0].numpy()
        
        # Continuous action space natively dictates proxy targets
        multiplier = max(0.01, min(math.exp(float(act_val[0])), 1.0))
        confidence = min(1.0, float(abs(act_val[1])))
        veto = multiplier < 0.05
        
        return multiplier, veto, confidence
