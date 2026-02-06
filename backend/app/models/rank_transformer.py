import torch
import torch.nn as nn
from typing import Optional, Dict

class RankTransformer(nn.Module):
    """
    Encoder-only Transformer for cross-sectional ranking.
    Input: [B, N, F] (Batch=Day, Assets=Universe, Feature=Input)
    Output: Per-asset outputs [B, N, ...]

    Architecture:
    - Linear Projection (d_model) + LayerNorm
    - Transformer Encoder Layers
    - Multi-task Heads -> Quantiles, Direction, Risk, Score
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
        **kwargs
    ):
        # Support aliases for Trainer compatibility
        if 'input_dim' in kwargs:
            d_input = kwargs['input_dim']
        if 'nhead' in kwargs:
            n_head = kwargs['nhead']
        if 'num_layers' in kwargs:
            n_layers = kwargs['num_layers']
            
        super().__init__()
        self.d_model = d_model
        self.pooling = pooling

        # Input Projection
        self.input_proj = nn.Sequential(
            nn.Linear(d_input, d_model),
            nn.LayerNorm(d_model)
        )

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Heads
        self.quantile_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 3) # q10, q50, q90
        )

        self.direction_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid() # Probability
        )

        self.risk_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

    def _pool(self, h: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if self.pooling == "none":
            return h

        if self.pooling == "last":
            if mask is None:
                return h[:, -1, :]
            lengths = mask.sum(dim=1).clamp(min=1).to(h.device)
            idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, h.size(-1))
            return h.gather(1, idx).squeeze(1)

        if self.pooling == "mean":
            if mask is None:
                return h.mean(dim=1)
            weights = mask.to(h.dtype).unsqueeze(-1)
            denom = weights.sum(dim=1).clamp(min=1.0)
            return (h * weights).sum(dim=1) / denom

        raise ValueError(f"Unsupported pooling mode: {self.pooling}")

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        x: [B, N, F]
        mask: [B, N] (optional padding mask)
        """
        B, N, F = x.shape

        # Embed
        h = self.input_proj(x) # [B, T, d_model]

        # Transformer Encoder
        # src_key_padding_mask expects [B, N] boolean (True = masked/ignore)
        # If mask is 1 for valid, 0 for pad -> invert for PyTorch
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = (mask == 0)

        h = self.encoder(h, src_key_padding_mask=key_padding_mask) # [B, T, d_model]

        pooled = self._pool(h, mask)

        quantiles = self.quantile_head(pooled)
        score = quantiles[..., 1:2]
        p_up = self.direction_head(pooled)
        risk = self.risk_head(pooled)

        return {
            "quantiles": quantiles,
            "score": score,
            "p_up": p_up,
            "risk": risk
        }
