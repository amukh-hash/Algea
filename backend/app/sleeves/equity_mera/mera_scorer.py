"""MERA Equity Scorer with Sparse Mixture-of-Experts (SMoE).

Fuses real-time feature embeddings with historical context through a
top-K routed SMoE gate. Incorporates MASTER Market Status Vector
for noisy top-K gating and standard routing loss.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SMoEGateNet(nn.Module):
    """Sparse Mixture-of-Experts gate with top-K routing and GRIP loss.

    Parameters
    ----------
    input_dim : int
        Dimension of the input features.
    n_experts : int
        Number of expert sub-networks.
    top_k : int
        Number of experts activated per sample.
    expert_dim : int
        Hidden dimension inside each expert.
    output_dim : int
        Output dimension of each expert.
    """

    def __init__(
        self,
        input_dim: int,
        n_experts: int = 8,
        top_k: int = 2,
        expert_dim: int = 64,
        output_dim: int = 32,
        market_dim: int = 16,
        noise_std: float = 0.1,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.noise_std = noise_std

        # Gating network based on MASTER Market Status
        self.gate = nn.Linear(input_dim + market_dim, n_experts)

        # Expert sub-networks
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, expert_dim),
                    nn.GELU(),
                    nn.Linear(expert_dim, output_dim),
                )
                for _ in range(n_experts)
            ]
        )

    def forward(self, x: torch.Tensor, market_status: torch.Tensor):
        """Forward pass with noisy top-K sparse routing.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch, input_dim)``.
        market_status : torch.Tensor
            Shape ``(batch, market_dim)``.

        Returns
        -------
        output : torch.Tensor
            Shape ``(batch, output_dim)``.
        gate_probs : torch.Tensor
            Shape ``(batch, n_experts)`` full gate probabilities.
        """
        # Form gating input combining specific feature and systemic market state
        gate_in = torch.cat([x, market_status], dim=-1)
        
        # Compute gate logits with noisy gating
        gate_logits = self.gate(gate_in)  # (B, n_experts)
        if self.training:
            gate_logits = gate_logits + torch.randn_like(gate_logits) * self.noise_std
            
        gate_probs = F.softmax(gate_logits, dim=-1)

        # Top-K selection
        top_k_vals, top_k_idx = torch.topk(gate_probs, self.top_k, dim=-1)
        # Re-normalise selected weights
        top_k_weights = top_k_vals / (top_k_vals.sum(dim=-1, keepdim=True) + 1e-8)

        # Compute expert outputs and combine
        batch_size = x.size(0)
        output = torch.zeros(batch_size, self.experts[0][-1].out_features, device=x.device)

        for i in range(self.top_k):
            expert_idx = top_k_idx[:, i]  # (B,)
            weight = top_k_weights[:, i].unsqueeze(-1)  # (B, 1)

            # Gather expert outputs — iterate since expert count is small
            expert_out = torch.zeros_like(output)
            for e in range(self.n_experts):
                mask = expert_idx == e
                if mask.any():
                    expert_out[mask] = self.experts[e](x[mask])

            output = output + weight * expert_out

        return output, gate_probs

    @staticmethod
    def standard_routing_loss(gate_probs: torch.Tensor) -> torch.Tensor:
        """Standard sink-aware routing loss for load balancing.

        Encourages both sum of probabilities (load) to be balanced across batch.

        Parameters
        ----------
        gate_probs : torch.Tensor
            Shape ``(batch, n_experts)`` softmax probabilities.

        Returns
        -------
        loss : torch.Tensor
        """
        n_experts = gate_probs.size(1)
        # f takes mean of gate probs over batch
        f = gate_probs.mean(dim=0)
        # importance is boolean selection fraction (approximated here by f)
        importance = f
        # Standard cv squared loss: n_experts * sum(f * importance) - 1
        loss = n_experts * torch.sum(f * importance) - 1.0
        return loss


class MERAEquityScorer(nn.Module):
    """Multi-scale Embedding Retrieval-Augmented equity scorer.

    Combines a real-time feature branch with a historical context branch
    through the SMoE gate to produce per-stock alpha scores.

    Parameters
    ----------
    realtime_dim : int
        Dimension of real-time features per stock.
    historical_dim : int
        Dimension of historical context embeddings.
    n_experts : int
        Number of SMoE experts.
    top_k : int
        Number of active experts per sample.
    """

    def __init__(
        self,
        realtime_dim: int = 32,
        historical_dim: int = 64,
        market_dim: int = 16,
        n_experts: int = 8,
        top_k: int = 2,
    ):
        super().__init__()
        combined_dim = realtime_dim + historical_dim

        self.realtime_proj = nn.Sequential(
            nn.Linear(realtime_dim, realtime_dim),
            nn.LayerNorm(realtime_dim),
            nn.GELU(),
        )

        self.historical_proj = nn.Sequential(
            nn.Linear(historical_dim, historical_dim),
            nn.LayerNorm(historical_dim),
            nn.GELU(),
        )

        self.gate = SMoEGateNet(
            input_dim=combined_dim,
            n_experts=n_experts,
            top_k=top_k,
            expert_dim=64,
            output_dim=32,
            market_dim=market_dim,
        )

        self.score_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )

    def forward(self, realtime: torch.Tensor, historical: torch.Tensor, market_status: torch.Tensor):
        """Produce per-stock alpha scores.

        Parameters
        ----------
        realtime : torch.Tensor
            Shape ``(batch, realtime_dim)``.
        historical : torch.Tensor
            Shape ``(batch, historical_dim)``.
        market_status : torch.Tensor
            Shape ``(batch, market_dim)``.

        Returns
        -------
        scores : torch.Tensor
            Shape ``(batch, 1)`` alpha scores.
        routing_loss : torch.Tensor
            Scalar standard load-balancing auxiliary loss.
        """
        rt = self.realtime_proj(realtime)
        hist = self.historical_proj(historical)
        combined = torch.cat([rt, hist], dim=-1)

        gated_out, gate_probs = self.gate(combined, market_status)
        scores = self.score_head(gated_out)
        routing_loss = SMoEGateNet.standard_routing_loss(gate_probs)

        return scores, routing_loss
