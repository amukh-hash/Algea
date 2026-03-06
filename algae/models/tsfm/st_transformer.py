import torch
import torch.nn as nn

class SpatialTemporalTransformer(nn.Module):
    """
    Pure Spatial-Temporal Transformer (dual-attention PatchTST variant) 
    treating the options grid as a global spatial-temporal sequence.
    Replaces deprecated CNN-LSTM networks.
    """
    def __init__(self, spatial_dim=25, temporal_dim=60, d_model=128, nhead=8, num_layers=3):
        super().__init__()
        # Flattened spatial-temporal grid embeddings
        self.embedding = nn.Linear(spatial_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, temporal_dim, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            batch_first=True,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Dense output embedding for TD3 State space S_t (pipe directly into RL agent)
        self.state_projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(temporal_dim * d_model, 256),
            nn.LayerNorm(256),
            nn.GELU()
        )
        
    def forward(self, grid_sequence: torch.Tensor) -> torch.Tensor:
        """
        grid_sequence: [Batch, Temporal, Spatial (e.g. Strikes/Expiries)]
        returns: [Batch, 256] dense embeddings for TD3 state space S_t.
        """
        # (B, T, S) -> (B, T, d_model)
        x = self.embedding(grid_sequence) + self.pos_encoder
        x = self.transformer(x)
        state_embed = self.state_projector(x)
        return state_embed
