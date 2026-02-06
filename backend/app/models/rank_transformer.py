import torch
import torch.nn as nn
from typing import Optional, Dict

class RankTransformer(nn.Module):
    """
    Encoder-only Transformer for scoring equities cross-sectionally.
    Input: [B, T, F] (Batch=Ticker, Time=Sequence, Feature=Input)
    Output: Score [B, 1] (Scalar score per ticker)

    Architecture:
    - Linear Projection (d_model)
    - Positional Encoding
    - Transformer Encoder Layers
    - Pooling (Last Token or Mean)
    - MLP Head -> Score
    """
    def __init__(
        self,
        d_input: int,
        d_model: int = 128,
        n_head: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        max_len: int = 64,
        pooling: str = "mean",
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
        self.input_proj = nn.Linear(d_input, d_model)

        # Positional Encoding (Learned or Sinusoidal)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model))

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Heads
        self.score_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1) # Linear score (logits for ranking)
        )

        # Optional Aux Heads
        self.direction_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid() # Probability
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
        x: [B, T, F]
        mask: [B, T] (optional padding mask)
        """
        B, T, F = x.shape

        # Embed
        h = self.input_proj(x) # [B, T, d_model]

        # Add Positional Encoding (broadcast)
        # Handle if T < max_len
        if T > self.pos_embedding.shape[1]:
             # Truncate or fail? Let's truncate pos embedding or handle dynamically?
             # For now, slice pos embedding
             h = h + self.pos_embedding[:, :T, :]
        else:
             h = h + self.pos_embedding[:, :T, :]

        # Transformer Encoder
        # src_key_padding_mask expects [B, T] boolean (True = masked/ignore)
        # If mask is 1 for valid, 0 for pad -> invert for PyTorch
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = (mask == 0)

        h = self.encoder(h, src_key_padding_mask=key_padding_mask) # [B, T, d_model]

        pooled = self._pool(h, mask)

        score = self.score_head(pooled)
        p_up = self.direction_head(pooled)

        return {
            "score": score,
            "p_up": p_up
        }
