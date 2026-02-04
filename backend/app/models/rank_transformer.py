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
        max_len: int = 64
    ):
        super().__init__()
        self.d_model = d_model

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

        # Pooling: Take the last valid token
        # If no mask, just take -1
        if mask is None:
            h_pool = h[:, -1, :]
        else:
            # Gather last valid index per batch
            # mask is [B, T], sum(dim=1) -> length
            lengths = mask.sum(dim=1).long() - 1 # [B]
            # Clamp to 0 just in case
            lengths = lengths.clamp(min=0)

            # Make sure lengths within bounds [0, T-1]
            lengths = torch.min(lengths, torch.tensor(T - 1, device=x.device))

            # Gather: [B, 1, d_model]
            # h[b, length[b], :]
            # Use gather or advanced indexing
            # h[torch.arange(B), lengths]
            h_pool = h[torch.arange(B, device=x.device), lengths, :]

        # Heads
        score = self.score_head(h_pool) # [B, 1]
        p_up = self.direction_head(h_pool) # [B, 1]

        return {
            "score": score,
            "p_up": p_up
        }
