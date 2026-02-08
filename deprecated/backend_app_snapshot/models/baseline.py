import torch
import torch.nn as nn

class BaselineMLP(nn.Module):
    def __init__(self, input_dim: int, lookback: int, horizons: int = 2, output_dim: int = 3):
        """
        Simple MLP baseline.
        output_dim: e.g. 3 quantiles (0.05, 0.5, 0.95)
        """
        super().__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(input_dim * lookback, 64),
            nn.ReLU(),
            nn.Linear(64, horizons * output_dim)
        )
        self.horizons = horizons
        self.output_dim = output_dim

    def forward(self, x):
        # x: (B, Lookback, InputDim)
        B = x.shape[0]
        x = self.flatten(x)
        out = self.net(x)
        # Reshape to (B, Horizons, OutputDim)
        return out.view(B, self.horizons, self.output_dim)
