"""
MLPSelector — fast per-ticker scorer for cross-sectional ranking.

Replaces TransformerEncoder self-attention (O(N²) over ~2,800 tickers)
with independent per-ticker MLP scoring (O(N)), reducing epoch time from
~30 min to ~2 min on production-sized universes.

Input features are already cross-sectionally z-scored, so explicit
cross-asset attention is redundant for ranking.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


class MLPSelector(nn.Module):
    """Per-ticker MLP scorer for cross-sectional ranking.

    Architecture
    ------------
    * LayerNorm on input features
    * ``depth`` blocks of: Linear → GELU → Dropout
    * Score head: Linear(hidden → 1)
    * Optional risk head: Linear(hidden → 1)

    Parameters
    ----------
    d_input : int
        Number of input features per ticker.
    hidden : int
        Hidden dimension for MLP layers.
    depth : int
        Number of hidden layers.
    dropout : float
        Dropout rate.
    use_risk_head : bool
        If True, adds a separate risk prediction head.
    """

    def __init__(
        self,
        d_input: int,
        hidden: int = 128,
        depth: int = 3,
        dropout: float = 0.10,
        use_risk_head: bool = False,
    ):
        super().__init__()
        self.use_risk_head = use_risk_head

        # Input normalization
        self.input_norm = nn.LayerNorm(d_input)

        # MLP trunk
        layers = []
        in_dim = d_input
        for _ in range(depth):
            layers.extend([
                nn.Linear(in_dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden
        self.trunk = nn.Sequential(*layers)

        # Score head
        self.score_head = nn.Linear(hidden, 1)

        # Optional risk head
        if use_risk_head:
            self.risk_head = nn.Linear(hidden, 1)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        x : Tensor [B, N, d_input]
            Per-ticker features.
        mask : Tensor [B, N], optional
            1 = valid ticker, 0 = padding.

        Returns
        -------
        dict with keys:
            ``score`` : [B, N, 1]
            ``risk``  : [B, N, 1] (only if use_risk_head)
        """
        # LayerNorm + MLP (applied pointwise across tickers)
        h = self.input_norm(x)      # [B, N, d_input]
        h = self.trunk(h)           # [B, N, hidden]
        score = self.score_head(h)  # [B, N, 1]

        out: Dict[str, torch.Tensor] = {"score": score}

        if self.use_risk_head:
            out["risk"] = self.risk_head(h)

        return out
