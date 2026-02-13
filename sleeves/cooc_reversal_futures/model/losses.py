from __future__ import annotations

import torch
import torch.nn.functional as F


def gaussian_nll(mu: torch.Tensor, log_sigma: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    sigma2 = torch.exp(2 * log_sigma).clamp_min(1e-8)
    return (0.5 * torch.log(sigma2) + 0.5 * (y - mu) ** 2 / sigma2).mean()


def huber(y_hat: torch.Tensor, y: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    return F.huber_loss(y_hat, y, delta=delta)


def sizing_utility(mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    return mu / sigma.clamp_min(1e-6) ** 2
