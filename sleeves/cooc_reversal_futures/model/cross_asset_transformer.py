from __future__ import annotations

import torch
from torch import nn

from .heads import MuSigmaHead


class CrossAssetTransformer(nn.Module):
    def __init__(self, in_dim: int, d_model: int = 64, n_heads: int = 4, n_layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.head = MuSigmaHead(d_model)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.encoder(self.proj(x))
        return self.head(h)
