"""
StudentModel - Lightweight Transformer for Classification

Distilled 'Bolt' version: 1/4th the parameters of Teacher for <50ms inference.
Designed for 3-class classification (Neutral, Buy, Sell).
"""

import torch
import torch.nn as nn


class StudentModel(nn.Module):
    """
    Lightweight Transformer Encoder for classification.
    Mimics Teacher but with reduced capacity for fast inference.
    """
    
    def __init__(self, config):
        super(StudentModel, self).__init__()
        self.input_dim = config.get('input_dim', 64)
        self.d_model = config.get('d_model', 128)      # Smaller than Teacher's 512
        self.nhead = config.get('nhead', 4)            # Fewer heads
        self.num_layers = config.get('num_layers', 2)  # Shallower (Teacher was 6)
        self.num_classes = config.get('num_classes', 3)
        self.dropout = config.get('dropout', 0.1)
        
        # Feature Projection
        self.embedding = nn.Linear(self.input_dim, self.d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 100, self.d_model))
        
        # Lightweight Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.d_model * 4,
            dropout=self.dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers
        )
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Linear(self.d_model // 2, self.num_classes)
        )

    def forward(self, x):
        """
        x: [Batch, Seq_Len, Features]
        Returns: logits [Batch, num_classes]
        """
        seq_len = x.size(1)
        
        # Embed
        x = self.embedding(x) + self.pos_encoder[:, :seq_len, :]
        
        # Transformer Pass
        x = self.transformer_encoder(x)
        
        # Pool (Take last step)
        last_step = x[:, -1, :]
        
        # Logits
        logits = self.classifier(last_step)
        return logits
