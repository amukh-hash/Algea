"""JIT-compatible Temporal Fusion Transformer for gap forecasting.

Optimized for the Ada Lovelace (RTX 4070 Super) tensor cores using
``bfloat16`` and ``torch.compile``.  No data-dependent control flow
in ``GatedResidualNetwork`` to ensure full compatibility with
``torch.compile(mode="reduce-overhead")``.

Output: Quantile predictions [P10, P50, P90] for the Open-to-Close
return of /ES, conditioned on 184 bars of overnight price action and
a set of exogenous covariates.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class GatedResidualNetwork(nn.Module):
    """Feed-forward GRN optimized for Torch compile.

    No data-dependent control flow — all paths execute unconditionally,
    making this fully compatible with ``torch.compile`` graph capture.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input features.
    hidden_dim : int
        Width of the hidden layers and output.
    dropout : float
        Dropout probability applied after the second linear layer.
    """

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.norm = nn.LayerNorm(hidden_dim)
        # Project residual to hidden_dim if dimensions mismatch
        self.residual_proj = (
            nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual_proj(x)
        x = self.fc1(x)
        x = self.elu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.gate(x) * x
        return self.norm(residual + x)


class TemporalFusionTransformer(nn.Module):
    """Conditional gap forecaster with quantile output.

    Architecture
    ------------
    1. **LSTM Temporal Encoder**: Processes the 184-bar overnight
       sequence ``[B, 184, 3]`` into hidden states.
    2. **Static VSN**: Encodes day-of-week, OpEx flag, and macro
       event ID through a GRN.
    3. **Observed VSN**: Encodes the 09:20 EST snapshot (gap proxy,
       Nikkei, EuroStoxx, /ZN drift, VIX) through a GRN.
    4. **Interpretable Multi-Head Attention**: Context (static + obs)
       attends over temporal hidden states.
    5. **Quantile Head**: Outputs [P10, P50, P90] for the OC return.

    Parameters
    ----------
    hidden_dim : int
        Width of hidden layers (default 64, optimized for 12GB VRAM).
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        # Encoders
        self.ts_encoder = nn.LSTM(
            input_size=3, hidden_size=hidden_dim, batch_first=True,
        )
        self.static_vsn = GatedResidualNetwork(3, hidden_dim)   # DoW, OpEx, Macro
        self.obs_vsn = GatedResidualNetwork(5, hidden_dim)       # Gap, Nikkei, Euro, ZN, VIX

        # Interpretable Multi-Head Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=4, batch_first=True,
        )

        # Quantile Output (0.10, 0.50, 0.90)
        self.quantile_head = nn.Linear(hidden_dim, 3)

    def forward(
        self,
        ts_data: torch.Tensor,
        static_data: torch.Tensor,
        obs_data: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        ts_data : Tensor
            Shape ``[B, 184, 3]`` — overnight 5-min bar features.
            Expected precision: ``torch.bfloat16`` for Ada Lovelace.
        static_data : Tensor
            Shape ``[B, 3]`` — day_of_week, is_opex, macro_event_id.
        obs_data : Tensor
            Shape ``[B, 5]`` — gap_proxy, nikkei, eurostoxx, zn, vix.

        Returns
        -------
        Tensor of shape ``[B, 3]`` — quantile predictions [P10, P50, P90].
        """
        lstm_out, _ = self.ts_encoder(ts_data)

        static_ctx = self.static_vsn(static_data).unsqueeze(1)
        obs_ctx = self.obs_vsn(obs_data).unsqueeze(1)

        # Context acts as Query; Temporal sequence acts as Key/Value
        context = static_ctx + obs_ctx
        attn_out, _ = self.attention(
            query=context, key=lstm_out, value=lstm_out,
        )

        return self.quantile_head(attn_out.squeeze(1))  # Shape: [B, 3]
