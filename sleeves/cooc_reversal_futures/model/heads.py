from __future__ import annotations

import torch
from torch import nn


class MuSigmaHead(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.mu = nn.Linear(d_model, 1)
        self.log_sigma = nn.Linear(d_model, 1)
        self.p_profit = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "mu": self.mu(x).squeeze(-1),
            "log_sigma": self.log_sigma(x).squeeze(-1),
            "p_profit": self.p_profit(x).squeeze(-1),
        }
