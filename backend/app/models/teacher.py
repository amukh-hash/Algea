"""
TeacherTransformer - High-capacity Transformer for Classification

Used as the source model for knowledge distillation.
Produces calibrated probability distributions for Student to learn from.
"""

import torch
import torch.nn as nn


class TeacherTransformer(nn.Module):
    """
    High-capacity Transformer Encoder for classification.
    Encoder-only (like BERT) since we're classifying, not generating.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_dim = config.get('input_dim', 64)
        self.d_model = config.get('d_model', 512)
        self.nhead = config.get('nhead', 8)
        self.num_layers = config.get('num_layers', 6)
        self.num_classes = config.get('num_classes', 3)
        self.dropout = config.get('dropout', 0.1)
        
        # Input projection
        self.feature_embedding = nn.Linear(self.input_dim, self.d_model)
        
        # Learnable positional encoding
        self.pos_encoder = nn.Parameter(torch.zeros(1, 100, self.d_model))
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dropout=self.dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model // 2, self.num_classes)
        )

    def forward(self, x, return_hidden=False):
        """
        x: [Batch, Seq_Len, Features]
        Returns logits [Batch, num_classes]
        """
        bs, seq_len, _ = x.shape
        
        # Embed and add positional encoding
        x = self.feature_embedding(x) + self.pos_encoder[:, :seq_len, :]
        
        # Transformer pass
        hidden = self.transformer_encoder(x)
        
        # Pool: take last timestep's embedding
        last_hidden = hidden[:, -1, :]
        
        logits = self.classifier(last_hidden)
        
        if return_hidden:
            return logits, hidden
        return logits
    
    def get_soft_targets(self, x, temperature):
        """Generate softened probability distribution for distillation."""
        logits = self.forward(x)
        return torch.softmax(logits / temperature, dim=-1)
