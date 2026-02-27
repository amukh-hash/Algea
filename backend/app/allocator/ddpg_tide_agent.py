"""DDPG Meta-Allocator with TiDE encoder and Deterministic Bounding.

Replaces Grid-Search MVO with a deep RL-based hierarchy:
- TiDE (Time-series Dense Encoder) for regime-aware state encoding
- DDPG actor with a ``DeterministicBoundingLayer`` that forces
  allocations to sum to 1.0 and clamps the VRP sleeve  ≤ 0.25
- DDPG critic optimising Dynamic CVaR
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# TiDE Encoder
# ======================================================================


class TiDEEncoder(nn.Module):
    """Time-series Dense Encoder for regime-aware state representation.

    Encodes a multi-variate return/feature time-series into a fixed-
    length state vector using dense projections with residual
    connections.

    Parameters
    ----------
    input_dim : int
        Features per timestep.
    seq_len : int
        Lookback length.
    hidden_dim : int
        Internal projection dimension.
    output_dim : int
        Final state-vector dimension.
    """

    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        hidden_dim: int = 128,
        output_dim: int = 64,
    ):
        super().__init__()
        flat_dim = input_dim * seq_len

        self.flatten_proj = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.residual = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.out_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode time-series to fixed-length vector.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch, seq_len, input_dim)``.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, output_dim)``.
        """
        flat = x.reshape(x.size(0), -1)  # (B, seq_len * input_dim)
        h = self.flatten_proj(flat)
        h = h + self.residual(h)  # residual connection
        return self.out_proj(h)


# ======================================================================
# Deterministic Bounding Layer
# ======================================================================


class DeterministicBoundingLayer(nn.Module):
    """Post-processing layer that enforces hard allocation constraints.

    1. Clamps VRP-sleeve weight to ``[0, max_vrp]``.
    2. Applies softmax to ensure all weights are non-negative and
       sum to 1.0.

    Parameters
    ----------
    n_sleeves : int
        Number of allocation outputs (sleeves).
    vrp_index : int
        Index of the VRP sleeve in the output vector.
    max_vrp : float
        Maximum allocation to VRP sleeve (default 0.25).
    """

    def __init__(self, n_sleeves: int, vrp_index: int = 2, max_vrp: float = 0.25):
        super().__init__()
        self.n_sleeves = n_sleeves
        self.vrp_index = vrp_index
        self.max_vrp = max_vrp

    def forward(self, raw_logits: torch.Tensor) -> torch.Tensor:
        """Apply allocation constraints.

        Parameters
        ----------
        raw_logits : torch.Tensor
            Shape ``(batch, n_sleeves)`` — unbounded actor output.

        Returns
        -------
        weights : torch.Tensor
            Shape ``(batch, n_sleeves)`` — constrained allocations
            summing to 1.0 with VRP ≤ max_vrp.
        """
        # Softmax gives non-negative weights summing to 1
        weights = F.softmax(raw_logits, dim=-1)

        vrp_w = weights[:, self.vrp_index]  # (B,)
        excess = (vrp_w - self.max_vrp).clamp(min=0.0)  # amount over cap

        # Build mask for non-VRP sleeves
        other_mask = torch.ones(self.n_sleeves, device=raw_logits.device, dtype=torch.bool)
        other_mask[self.vrp_index] = False

        other_w = weights[:, other_mask]  # (B, n_sleeves-1)
        other_sum = other_w.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        # Redistribute excess proportionally to other sleeves
        redistribution = excess.unsqueeze(-1) * (other_w / other_sum)

        result = weights.clone()
        result[:, self.vrp_index] = vrp_w - excess
        result[:, other_mask] = other_w + redistribution
        return result


# ======================================================================
# Actor
# ======================================================================


class DDPGAllocatorActor(nn.Module):
    """DDPG actor for sleeve allocation with deterministic bounding.

    Parameters
    ----------
    state_dim : int
        Dimension of the regime-aware state vector (TiDE output).
    n_sleeves : int
        Number of sleeves to allocate across.
    vrp_index : int
        Index of VRP sleeve for the hard cap.
    max_vrp : float
        Maximum VRP allocation.
    """

    def __init__(
        self,
        state_dim: int,
        n_sleeves: int = 4,
        vrp_index: int = 2,
        max_vrp: float = 0.25,
    ):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_sleeves),
        )
        self.bounding = DeterministicBoundingLayer(n_sleeves, vrp_index, max_vrp)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Produce bounded allocation weights.

        Parameters
        ----------
        state : torch.Tensor
            Shape ``(batch, state_dim)``.

        Returns
        -------
        weights : torch.Tensor
            Shape ``(batch, n_sleeves)`` — constrained allocations.
        """
        raw = self.backbone(state)
        return self.bounding(raw)


# ======================================================================
# Critic
# ======================================================================


class DDPGAllocatorCritic(nn.Module):
    """Q-network for allocation quality (optimises Dynamic CVaR).

    Parameters
    ----------
    state_dim : int
        State-vector dimension.
    n_sleeves : int
        Number of allocation weights (action dimension).
    """

    def __init__(self, state_dim: int, n_sleeves: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + n_sleeves, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Evaluate state-action pair.

        Parameters
        ----------
        state : torch.Tensor
            Shape ``(batch, state_dim)``.
        action : torch.Tensor
            Shape ``(batch, n_sleeves)``.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, 1)`` — Q-value.
        """
        return self.net(torch.cat([state, action], dim=1))
