import torch
import torch.nn as nn

class PatchTSTConfig:
    def __init__(self, c_in=1, context_len=512, target_len=96, patch_len=16, stride=8, d_model=128, n_heads=8, e_layers=3, d_ff=256, dropout=0.1):
        self.c_in = c_in
        self.context_len = context_len
        self.target_len = target_len
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.d_ff = d_ff
        self.dropout = dropout

class PatchTST(nn.Module):
    def __init__(self, config: PatchTSTConfig):
        super().__init__()
        self.config = config
        self.patch_num = int((config.context_len - config.patch_len) / config.stride + 1)
        
        # Channel independence formatting
        # Input shape: [Batch, Length, Channels]
        self.value_embedding = nn.Linear(config.patch_len, config.d_model, bias=False)
        self.position_embedding = nn.Parameter(torch.empty(1, self.patch_num, config.d_model))
        nn.init.uniform_(self.position_embedding, -0.1, 0.1)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model, 
            nhead=config.n_heads, 
            dim_feedforward=config.d_ff, 
            dropout=config.dropout, 
            activation="gelu", 
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.e_layers)
        
        # Head
        self.head_nf = config.d_model * self.patch_num
        self.head = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Dropout(config.dropout),
            nn.Linear(self.head_nf, config.target_len)
        )
        self.dropout = nn.Dropout(config.dropout)
        
        # Phase 1.2: Native regression loss replacing any discrete CE
        self.criterion = nn.HuberLoss()
        
    def compute_loss(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Computes regression loss optimized for continuous mathematical inputs."""
        return self.criterion(preds, targets)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, C] where C is ideally 1 for explicit Channel Independence (c_in=1)
        returns: [B, TargetLen, C]
        """
        B, L, C = x.shape
        # Channel-independent Patching: treat each channel as a separate sample in batch
        x = x.permute(0, 2, 1) # [B, C, L]
        x = x.reshape(B * C, L) # [B*C, L]
        
        # Patching
        # Unfold (B*C, L) -> (B*C, patch_num, patch_len)
        x_unfolded = x.unfold(dimension=1, size=self.config.patch_len, step=self.config.stride) 
        
        # Embed patches
        enc_in = self.value_embedding(x_unfolded) + self.position_embedding
        enc_in = self.dropout(enc_in)
        
        # Transformer
        enc_out = self.encoder(enc_in) # (B*C, patch_num, d_model)
        
        # Head
        out = self.head(enc_out) # (B*C, target_len)
        
        # Reshape back to [B, TargetLen, C]
        out = out.reshape(B, C, self.config.target_len)
        out = out.permute(0, 2, 1) # [B, TargetLen, C]
        
        return out
