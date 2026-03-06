"""
iTransformer — Inverted Transformer for Multivariate Statistical Arbitrage.

Architecture:
    Attention is calculated across *variates* (ETFs/assets) rather than time
    steps.  Each variate's full lookback window is projected into a single
    token embedding, and the Transformer Encoder models cross-variate
    dependencies via multi-head self-attention.

    Input:  [Batch, Time, Variates]
    Output: [Batch, Variates, Horizon]

Reference:
    Liu et al., "iTransformer: Inverted Transformers Are Effective for
    Time Series Forecasting", ICLR 2024.

Designed for the 3090 Ti (24 GB VRAM).  Use ``torch.bfloat16`` for
inference to halve memory and leverage Ampere tensor cores.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class iTransformer(nn.Module):
    """
    Production implementation of the iTransformer (Inverted Transformer)
    for Multivariate StatArb.

    Attention is calculated across variates (ETFs) instead of time steps.
    Each variate's entire lookback window is projected into a d_model-dim
    token, and the Transformer Encoder learns cross-variate dependencies.

    Parameters
    ----------
    num_variates : int
        Number of assets/ETFs in the cross-section (e.g. 6 sector ETFs).
    lookback_len : int
        Number of historical time steps per variate (e.g. 60 for 5-min bars).
    pred_len : int
        Forecast horizon in time steps.
    d_model : int
        Hidden dimension of the Transformer.
    n_heads : int
        Number of attention heads.
    e_layers : int
        Number of Transformer encoder layers.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        num_variates: int,
        lookback_len: int,
        pred_len: int,
        d_model: int = 256,
        n_heads: int = 8,
        e_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_variates = num_variates
        self.lookback_len = lookback_len
        self.pred_len = pred_len

        # 1. Variate Embedding: Project entire time series into d_model
        #    Each variate's [lookback_len] window → [d_model] token
        self.projector = nn.Linear(lookback_len, d_model)

        # 2. Learnable Variate Identity Embedding
        #    Encodes which variate (ETF) each token represents
        self.variate_embedding = nn.Parameter(
            torch.randn(1, num_variates, d_model) * 0.02
        )

        # 3. Transformer Encoder (Attention over Variates)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)

        # 4. Layer Norm before prediction
        self.norm = nn.LayerNorm(d_model)

        # 5. Output Projection: [d_model] → [pred_len] per variate
        self.predictor = nn.Linear(d_model, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : Tensor
            Shape ``[Batch, Time, Variates]`` — multivariate time series.

        Returns
        -------
        Tensor
            Shape ``[Batch, Variates, Horizon]`` — per-variate forecasts.

        Raises
        ------
        ValueError
            If input shape does not match ``(B, lookback_len, num_variates)``.
        """
        B, T, N = x.shape
        if T != self.lookback_len or N != self.num_variates:
            raise ValueError(
                f"Expected shape [B, {self.lookback_len}, {self.num_variates}], "
                f"got [B, {T}, {N}]"
            )

        # Invert: [Batch, Time, Variates] → [Batch, Variates, Time]
        x_inv = x.transpose(1, 2)

        # Embed temporal dynamics into a single variate token
        # [B, N, T] → [B, N, d_model]
        x_emb = self.projector(x_inv) + self.variate_embedding

        # Cross-variate attention
        enc_out = self.encoder(x_emb)
        enc_out = self.norm(enc_out)

        # Predict future steps for each variate
        # [B, N, d_model] → [B, N, pred_len]
        return self.predictor(enc_out)
