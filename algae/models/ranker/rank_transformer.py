"""
Rank Transformer and SimpleRanker for cross-sectional stock ranking.

RankTransformer: encoder-only Transformer with multi-task heads (quantiles, direction, risk).
SimpleRanker: lightweight fallback that sums feature columns.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Lightweight fallback (backward-compatible stub)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RankerConfig:
    feature_columns: tuple[str, ...] = ("ret_1d", "ret_5d", "vol_20d", "dollar_vol")


class SimpleRanker:
    def __init__(self, config: RankerConfig) -> None:
        self.config = config

    def score(self, features: pd.DataFrame) -> pd.Series:
        return features[list(self.config.feature_columns)].sum(axis=1)


# ---------------------------------------------------------------------------
# Full RankTransformer (ported from deprecated/backend_app_snapshot/models/rank_transformer.py)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RankTransformerConfig:
    d_input: int = 18
    d_model: int = 128
    n_head: int = 4
    n_layers: int = 2
    dropout: float = 0.1
    max_len: int = 64
    pooling: str = "none"


class RankTransformer(nn.Module):
    """
    Encoder-only Transformer for cross-sectional ranking.

    Input : ``[B, N, F]``  (Batch=Day, N=Assets in universe, F=feature dim)
    Output: per-asset dict with keys ``quantiles``, ``score``, ``p_up``, ``risk``.

    Architecture
    ------------
    * Linear projection → LayerNorm
    * ``n_layers`` TransformerEncoder layers (``norm_first=True``, ``batch_first=True``)
    * Multi-task heads:
        - quantile_head → q10, q50, q90  (3 values)
        - direction_head → P(up) via Sigmoid
        - risk_head → scalar risk score
    """

    def __init__(
        self,
        d_input: int,
        d_model: int = 128,
        n_head: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        max_len: int = 64,
        pooling: str = "none",
        **kwargs,
    ):
        # Support aliases from Trainer / config dicts
        if "input_dim" in kwargs:
            d_input = kwargs["input_dim"]
        if "nhead" in kwargs:
            n_head = kwargs["nhead"]
        if "num_layers" in kwargs:
            n_layers = kwargs["num_layers"]

        super().__init__()
        self.d_model = d_model
        self.pooling = pooling

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(d_input, d_model),
            nn.LayerNorm(d_model),
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Heads
        self.quantile_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 3),  # q10, q50, q90
        )
        self.direction_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )
        self.risk_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    # ------------------------------------------------------------------
    def _pool_last(self, h: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            return h[:, -1, :]
        lengths = mask.sum(dim=1).clamp(min=1).to(h.device)
        idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, h.size(-1))
        return h.gather(1, idx).squeeze(1)

    def _pool_mean(self, h: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            return h.mean(dim=1)
        weights = mask.to(h.dtype).unsqueeze(-1)
        denom = weights.sum(dim=1).clamp(min=1.0)
        return (h * weights).sum(dim=1) / denom

    _POOL_FNS = {
        "none": lambda self, h, mask: h,
        "last": _pool_last,
        "mean": _pool_mean,
    }

    def _pool(self, h: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        fn = self._POOL_FNS.get(self.pooling)
        if fn is None:
            raise ValueError(f"Unsupported pooling mode: {self.pooling}")
        return fn(self, h, mask)

    # ------------------------------------------------------------------
    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        x : Tensor [B, N, F]
        mask : Tensor [B, N] — 1 = valid, 0 = padding (optional)
        """
        h = self.input_proj(x)

        key_padding_mask = None
        if mask is not None:
            key_padding_mask = mask == 0  # PyTorch convention: True = ignore

        h = self.encoder(h, src_key_padding_mask=key_padding_mask)
        pooled = self._pool(h, mask)

        quantiles = self.quantile_head(pooled)
        score = quantiles[..., 1:2]
        p_up = self.direction_head(pooled)
        risk = self.risk_head(pooled)

        return {
            "quantiles": quantiles,
            "score": score,
            "p_up": p_up,
            "risk": risk,
        }
