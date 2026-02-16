"""Cross-Sectional Set Transformer for daily instrument panels.

Architecture
------------
1. Token embedding MLP: ``F → hidden_dim``
2. TransformerEncoder with self-attention across instruments
3. Score head:  ``hidden_dim -> 1`` (per-instrument reversal score)
4. Risk  head:  ``hidden_dim -> 1`` raw log_sigma (per-instrument risk scale)

Input per day:  ``X_day [N, F]``
Output per day: ``score_day [N]``  (or ``(score, risk)`` when ``return_risk=True``)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossSectionalTransformer(nn.Module):
    """Set Transformer for daily cross-sectional instrument panels.

    Parameters
    ----------
    n_features : number of input features per instrument.
    hidden_dim : internal dimension for attention layers.
    n_heads : number of attention heads.
    n_layers : number of transformer encoder layers.
    dropout : dropout rate (set to 0 for inference).
    two_head : if True, model also predicts per-instrument risk (|r_oc| proxy).
    """

    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
        two_head: bool = False,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.two_head = two_head

        # Token embedding: per-instrument features → hidden_dim
        self.embed = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        # Self-attention across instruments
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

        # Score head: hidden_dim → 1 (per-instrument reversal score)
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Risk head: hidden_dim -> 1 (raw log_sigma — NO softplus in forward)
        # softplus + floor is applied in loss (StudentTNLL) and stabilizer
        if two_head:
            self.risk_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 1),
            )
        else:
            self.risk_head = None

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        return_risk: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        x : [B, N, F] — features per instrument per day.
        mask : [B, N] — True for valid instruments, False for padded. Optional.
        return_risk : if True and model has risk_head, return (score, risk).

        Returns
        -------
        scores : [B, N]  (default)
        (scores, risk) : [B, N], [B, N]  (when return_risk=True and two_head)
        """
        # Embed tokens
        h = self.embed(x)  # [B, N, hidden_dim]

        # Build attention mask: TransformerEncoder expects src_key_padding_mask
        # where True = ignore (padded)
        src_key_padding_mask = None
        if mask is not None:
            src_key_padding_mask = ~mask  # invert: True = padded = ignore

        # Self-attention
        h = self.encoder(h, src_key_padding_mask=src_key_padding_mask)  # [B, N, hidden_dim]

        # Per-instrument score
        scores = self.score_head(h).squeeze(-1)  # [B, N]

        if return_risk and self.risk_head is not None:
            log_sigma_raw = self.risk_head(h).squeeze(-1)  # [B, N], raw
            return scores, log_sigma_raw

        return scores

    def count_parameters(self) -> int:
        """Total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
