"""Spatial-Temporal Transformer for IV Surface Forecasting.

Replaces the legacy LSTM-CNN hybrid with a dual-attention Transformer:
  - Spatial attention: captures correlations across moneyness-DTE grid positions
  - Temporal attention: captures dynamics across time steps
  - ProbSparse self-attention for O(L log L) efficiency
  - Grid reconstruction head for IV surface pre-training
  - Dense context-aware embedding output for TD3 state space

No CNN, no LSTM, no manual Black-Scholes greeks.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════
# ProbSparse Self-Attention
# ═══════════════════════════════════════════════════════════════════════

class ProbSparseAttention(nn.Module):
    """ProbSparse self-attention (Informer-style) for efficiency.

    Selects the top-u queries with highest KL-divergence from uniform
    distribution, reducing from O(L²) to O(L log L).

    Parameters
    ----------
    d_model : int
        Hidden dimension.
    n_heads : int
        Number of attention heads.
    factor : int
        Sparsity factor (c in the paper, controls log L sampling).
    dropout : float
        Attention dropout rate.
    """

    def __init__(self, d_model: int, n_heads: int = 8, factor: int = 5, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.factor = factor

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _prob_sparse_scores(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select top-u queries by KL-divergence from uniform.

        Parameters
        ----------
        Q, K : torch.Tensor
            Shape ``(batch, n_heads, L, d_k)``.
        """
        B, H, L, D = Q.shape
        U = max(1, int(self.factor * math.log(L + 1)))
        U = min(U, L)  # cap at sequence length

        # Sample U keys randomly to estimate sparsity
        K_sample_idx = torch.randint(0, L, (U,), device=Q.device)
        K_sample = K[:, :, K_sample_idx, :]  # (B, H, U, D)

        # Compute sparsity measure: max(QK^T) - mean(QK^T) for each query
        Q_K_sample = torch.matmul(Q, K_sample.transpose(-2, -1))  # (B, H, L, U)
        M = Q_K_sample.max(dim=-1).values - Q_K_sample.mean(dim=-1)  # (B, H, L)

        # Select top-u queries
        u = min(U, L)
        top_u_idx = M.topk(u, dim=-1).indices  # (B, H, u)

        return top_u_idx, M

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch, seq_len, d_model)``.

        Returns
        -------
        torch.Tensor — same shape.
        """
        B, L, D = x.shape
        H = self.n_heads
        d_k = self.d_k

        Q = self.W_q(x).view(B, L, H, d_k).transpose(1, 2)  # (B, H, L, d_k)
        K = self.W_k(x).view(B, L, H, d_k).transpose(1, 2)
        V = self.W_v(x).view(B, L, H, d_k).transpose(1, 2)

        # ProbSparse: select active queries
        top_idx, _ = self._prob_sparse_scores(Q, K)
        u = top_idx.size(-1)

        # Gather active queries
        top_idx_exp = top_idx.unsqueeze(-1).expand(-1, -1, -1, d_k)  # (B, H, u, d_k)
        Q_sparse = Q.gather(2, top_idx_exp)  # (B, H, u, d_k)

        # Standard attention on sparse queries
        scale = math.sqrt(d_k)
        attn = torch.matmul(Q_sparse, K.transpose(-2, -1)) / scale  # (B, H, u, L)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        sparse_out = torch.matmul(attn, V)  # (B, H, u, d_k)

        # Fill back into full context (non-selected queries retain values)
        context = V.mean(dim=2, keepdim=True).expand_as(V).clone()
        context.scatter_(2, top_idx_exp, sparse_out)

        # Merge heads
        context = context.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(context)


# ═══════════════════════════════════════════════════════════════════════
# Dual-Attention Encoder Block
# ═══════════════════════════════════════════════════════════════════════

class SpatialTemporalBlock(nn.Module):
    """Single dual-attention block: spatial attention + temporal attention.

    Parameters
    ----------
    d_model : int
        Hidden dimension.
    n_heads : int
        Number of attention heads.
    d_ff : int
        Feed-forward inner dimension.
    dropout : float
        Dropout rate.
    """

    def __init__(self, d_model: int = 128, n_heads: int = 8, d_ff: int = 256, dropout: float = 0.1):
        super().__init__()

        # Spatial attention (across grid positions within each time step)
        self.spatial_norm = nn.LayerNorm(d_model)
        self.spatial_attn = ProbSparseAttention(d_model, n_heads, dropout=dropout)

        # Temporal attention (across time steps for each grid position)
        self.temporal_norm = nn.LayerNorm(d_model)
        self.temporal_attn = ProbSparseAttention(d_model, n_heads, dropout=dropout)

        # Feed-forward
        self.ff_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, n_grid: int, n_time: int) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch, n_time * n_grid, d_model)``.
        n_grid : int
            Number of grid positions (e.g. 100 for 10×10).
        n_time : int
            Number of time steps.

        Returns
        -------
        torch.Tensor — same shape.
        """
        B, S, D = x.shape

        # ── Spatial attention (within each time step) ────────────────
        # Reshape to (B * n_time, n_grid, D) for spatial attention per timestep
        x_spatial = x.view(B * n_time, n_grid, D)
        x_spatial = x_spatial + self.spatial_attn(self.spatial_norm(x_spatial))
        x = x_spatial.view(B, S, D)

        # ── Temporal attention (across time for each grid pos) ───────
        # Reshape to (B * n_grid, n_time, D) for temporal attention per position
        x_temp = x.view(B, n_time, n_grid, D).permute(0, 2, 1, 3).reshape(B * n_grid, n_time, D)
        x_temp = x_temp + self.temporal_attn(self.temporal_norm(x_temp))
        x = x_temp.view(B, n_grid, n_time, D).permute(0, 2, 1, 3).reshape(B, S, D)

        # ── Feed-forward ─────────────────────────────────────────────
        x = x + self.ff(self.ff_norm(x))

        return x


# ═══════════════════════════════════════════════════════════════════════
# Full Spatial-Temporal Transformer
# ═══════════════════════════════════════════════════════════════════════

class SpatialTemporalTransformer(nn.Module):
    """Dual-attention Transformer for IV surface analysis.

    Replaces the legacy LSTM-CNN ``IVSurfaceForecaster``.

    Architecture:
        Flatten 10×10 grid → token embedding + positional encoding →
        N × (Spatial Attention + Temporal Attention + FFN) →
        Dense context embedding (for TD3 state) + Grid reconstruction head

    Parameters
    ----------
    grid_h, grid_w : int
        IV grid dimensions (10×10).
    n_time_steps : int
        Number of historical time steps in the input sequence.
    d_model : int
        Transformer hidden dimension.
    n_heads : int
        Number of attention heads.
    n_layers : int
        Number of encoder blocks.
    d_ff : int
        Feed-forward inner dimension.
    dropout : float
        Dropout rate.
    embed_dim : int
        Output embedding dimension for TD3 state space.
    """

    def __init__(
        self,
        grid_h: int = 10,
        grid_w: int = 10,
        n_time_steps: int = 20,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
        embed_dim: int = 64,
    ):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.n_grid = grid_h * grid_w  # 100
        self.n_time_steps = n_time_steps

        # Token embedding: each grid position is a token
        self.token_embed = nn.Linear(1, d_model)

        # Spatial positional encoding (grid position)
        self.spatial_pos = nn.Parameter(
            torch.randn(1, self.n_grid, d_model) * 0.02
        )

        # Temporal positional encoding (time step)
        self.temporal_pos = nn.Parameter(
            torch.randn(1, n_time_steps, 1, d_model) * 0.02
        )

        self.embed_norm = nn.LayerNorm(d_model)
        self.embed_dropout = nn.Dropout(dropout)

        # Dual-attention encoder blocks
        self.blocks = nn.ModuleList([
            SpatialTemporalBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Dense context embedding head (output for TD3 state)
        total_tokens = n_time_steps * self.n_grid
        self.context_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, embed_dim),
            nn.GELU(),
        )

        # Grid reconstruction head (for pre-training)
        self.reconstruction_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(
        self, iv_grids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass producing both context embedding and grid reconstruction.

        Parameters
        ----------
        iv_grids : torch.Tensor
            Shape ``(batch, n_time_steps, grid_h, grid_w)`` — historical IV surfaces.

        Returns
        -------
        context_embedding : torch.Tensor
            Shape ``(batch, embed_dim)`` — dense state vector for TD3.
        reconstructed_grid : torch.Tensor
            Shape ``(batch, n_time_steps, grid_h, grid_w)`` — reconstructed surfaces.
        """
        B, T, H, W = iv_grids.shape
        assert H == self.grid_h and W == self.grid_w, \
            f"Expected grid ({self.grid_h}×{self.grid_w}), got ({H}×{W})"

        # Flatten grid: (B, T, H, W) → (B, T, H*W, 1)
        x = iv_grids.view(B, T, self.n_grid, 1)

        # Token embedding
        x = self.token_embed(x)  # (B, T, n_grid, d_model)

        # Add positional encodings
        x = x + self.spatial_pos.unsqueeze(1)  # spatial: broadcast over time
        x = x + self.temporal_pos[:, :T, :, :]  # temporal: broadcast over grid

        # Flatten to sequence: (B, T * n_grid, d_model)
        x = x.view(B, T * self.n_grid, -1)
        x = self.embed_dropout(self.embed_norm(x))

        # Dual-attention encoder
        for block in self.blocks:
            x = block(x, n_grid=self.n_grid, n_time=T)

        # ── Context embedding (for TD3 state) ────────────────────────
        # Pool over all tokens to produce a single dense vector
        pooled = x.mean(dim=1)  # (B, d_model)
        context_embedding = self.context_head(pooled)  # (B, embed_dim)

        # ── Grid reconstruction (for pre-training) ───────────────────
        recon = self.reconstruction_head(x)  # (B, T*n_grid, 1)
        reconstructed_grid = recon.squeeze(-1).view(B, T, self.grid_h, self.grid_w)

        return context_embedding, reconstructed_grid
