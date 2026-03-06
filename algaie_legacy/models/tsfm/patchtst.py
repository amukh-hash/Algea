"""ContinuousPatchTST — Pure continuous regression forecaster for futures.

Implements a channel-independent PatchTST architecture with:
  - Learnable linear patch embedding + positional encoding
  - Multi-head self-attention encoder with LayerNorm
  - nn.Linear regression head projecting to continuous float forecast
  - RevIN (Reversible Instance Normalization) for denorm to absolute scale

Loss: strict MSE  L = (1/n) Σ (yi - ŷi)²
No discrete tokens, no vocabulary, no HuggingFace tokenizer.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════
# RevIN Layer
# ═══════════════════════════════════════════════════════════════════════

class RevIN(nn.Module):
    """Reversible Instance Normalization (learnable affine variant).

    Normalises each instance independently, then denormalises the output
    to restore absolute price magnitude for the regression head.

    Parameters
    ----------
    num_features : int
        Number of input channels (1 for channel-independent PatchTST).
    eps : float
        Numerical stability for std division.
    affine : bool
        Whether to learn scale/shift parameters.
    """

    def __init__(self, num_features: int = 1, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))

        # Stored statistics for denormalization
        self.register_buffer("_mean", torch.zeros(1), persistent=False)
        self.register_buffer("_std", torch.ones(1), persistent=False)

    def forward(self, x: torch.Tensor, mode: str = "norm") -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch, seq_len)`` or ``(batch, seq_len, 1)``.
        mode : str
            ``'norm'`` to normalize, ``'denorm'`` to reverse.
        """
        if mode == "norm":
            self._mean = x.mean(dim=1, keepdim=True).detach()
            self._std = (x.std(dim=1, keepdim=True) + self.eps).detach()
            x = (x - self._mean) / self._std
            if self.affine:
                x = x * self.gamma + self.beta
            return x
        elif mode == "denorm":
            if self.affine:
                x = (x - self.beta) / (self.gamma + self.eps)
            return x * self._std + self._mean
        else:
            raise ValueError(f"RevIN mode must be 'norm' or 'denorm', got '{mode}'")


# ═══════════════════════════════════════════════════════════════════════
# Patch Embedding
# ═══════════════════════════════════════════════════════════════════════

class PatchEmbedding(nn.Module):
    """Learnable linear projection of patches + positional encoding.

    Parameters
    ----------
    patch_len : int
        Length of each patch (input dimension per patch token).
    d_model : int
        Transformer hidden dimension.
    max_patches : int
        Maximum number of patches (for positional encoding table).
    dropout : float
        Dropout rate on the embedding.
    """

    def __init__(
        self,
        patch_len: int = 16,
        d_model: int = 128,
        max_patches: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.projection = nn.Linear(patch_len, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_patches, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        patches : torch.Tensor
            Shape ``(batch, num_patches, patch_len)``.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, num_patches, d_model)``.
        """
        n_patches = patches.size(1)
        x = self.projection(patches)  # (B, N, d_model)
        x = x + self.pos_encoding[:, :n_patches, :]
        x = self.norm(x)
        return self.dropout(x)


# ═══════════════════════════════════════════════════════════════════════
# Transformer Encoder Block
# ═══════════════════════════════════════════════════════════════════════

class PatchTSTEncoderLayer(nn.Module):
    """Single Transformer encoder layer with pre-norm architecture.

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
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch, num_patches, d_model)``.
        """
        # Pre-norm self-attention
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out

        # Pre-norm feed-forward
        normed = self.norm2(x)
        x = x + self.ff(normed)

        return x


class PatchTSTEncoder(nn.Module):
    """Stack of Transformer encoder layers.

    Parameters
    ----------
    n_layers : int
        Number of stacked encoder layers.
    d_model : int
        Hidden dimension.
    n_heads : int
        Number of attention heads.
    d_ff : int
        Feed-forward inner dimension.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        n_layers: int = 3,
        d_model: int = 128,
        n_heads: int = 8,
        d_ff: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            PatchTSTEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch, num_patches, d_model)``.

        Returns
        -------
        torch.Tensor
            Same shape, normalized.
        """
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)


# ═══════════════════════════════════════════════════════════════════════
# Full ContinuousPatchTST Model
# ═══════════════════════════════════════════════════════════════════════

class ContinuousPatchTST(nn.Module):
    """Continuous PatchTST for time-series regression forecasting.

    Architecture flow:
        RevIN(norm) → Patch Embedding → Transformer Encoder →
        Flatten → Linear Regression Head → RevIN(denorm)

    Parameters
    ----------
    c_in : int
        Number of input channels (1 for channel-independent mode).
    seq_len : int
        Input sequence length.
    patch_len : int
        Length of each patch.
    stride : int
        Stride between patches.
    forecast_horizon : int
        Number of future time steps to predict.
    d_model : int
        Transformer hidden dimension.
    n_heads : int
        Number of attention heads.
    n_layers : int
        Number of encoder layers.
    d_ff : int
        Feed-forward inner dimension.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        c_in: int = 1,
        seq_len: int = 512,
        patch_len: int = 16,
        stride: int = 8,
        forecast_horizon: int = 1,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.c_in = c_in
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.forecast_horizon = forecast_horizon

        # Number of patches for the given sequence configuration
        self.num_patches = (seq_len - patch_len) // stride + 1

        # RevIN for input normalization / output denormalization
        self.revin = RevIN(num_features=1, affine=True)

        # Patch embedding with positional encoding
        self.patch_embed = PatchEmbedding(
            patch_len=patch_len,
            d_model=d_model,
            max_patches=self.num_patches + 16,  # safety margin
            dropout=dropout,
        )

        # Transformer encoder
        self.encoder = PatchTSTEncoder(
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
        )

        # Regression head: flatten encoder output → continuous float forecast
        self.regression_head = nn.Sequential(
            nn.Linear(self.num_patches * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, forecast_horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass producing continuous float forecast.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch, seq_len)`` — single-channel time series.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, forecast_horizon)`` — continuous float predictions.
        """
        # RevIN normalize (stores mean/std for denorm)
        x_norm = self.revin(x, mode="norm")

        # Create patches: (B, seq_len) → (B, num_patches, patch_len)
        patches = x_norm.unfold(dimension=1, size=self.patch_len, step=self.stride)

        # Patch embedding → Transformer encoder
        embedded = self.patch_embed(patches)
        enc_out = self.encoder(embedded)

        # Flatten and project to continuous output
        flat = enc_out.flatten(start_dim=1)  # (B, num_patches * d_model)
        prediction = self.regression_head(flat)  # (B, forecast_horizon)

        # CRITICAL: RevIN denormalize to restore absolute magnitude
        prediction = self.revin(prediction, mode="denorm")

        return prediction

    @staticmethod
    def mse_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Strict MSE loss: L = (1/n) Σ (yi - ŷi)²"""
        return F.mse_loss(y_pred, y_true, reduction="mean")
