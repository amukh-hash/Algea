from __future__ import annotations

import torch


def mean_pooling(hidden_states: torch.Tensor) -> torch.Tensor:
    if hidden_states.ndim != 3:
        raise ValueError("hidden_states must be [B, L, D]")
    return hidden_states.mean(dim=1)
