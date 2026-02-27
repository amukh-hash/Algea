"""LSTM-CNN hybrid IV Surface Forecaster.

Treats temporal macro/underlying dynamics via LSTM and the
strike/expiry grid spatially via CNN, fusing both to forecast a 10×10
implied-volatility surface.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class IVSurfaceForecaster(nn.Module):
    """Hybrid LSTM-CNN model for implied-volatility surface forecasting.

    Parameters
    ----------
    temporal_input_dim : int
        Number of features per timestep in the temporal sequence
        (e.g. OHLCV + VIX = 6).
    grid_channels : int
        Number of channels in the spatial IV grid input
        (e.g. current IV, historical IV, volume = 3).
    hidden_dim : int
        LSTM hidden-state dimension.
    grid_h : int
        Height of the output IV surface grid (moneyness bins).
    grid_w : int
        Width of the output IV surface grid (DTE bins).
    """

    def __init__(
        self,
        temporal_input_dim: int,
        grid_channels: int,
        hidden_dim: int = 128,
        grid_h: int = 10,
        grid_w: int = 10,
    ):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w

        # ── Temporal branch (LSTM) ────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=temporal_input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
        )

        # ── Spatial branch (CNN) ──────────────────────────────────────
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=grid_channels, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # → (B, 32, 4, 4)
        )

        # ── Fusion ────────────────────────────────────────────────────
        cnn_flat = 32 * 4 * 4
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + cnn_flat, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, grid_h * grid_w),
        )

    def forward(
        self, temporal_seq: torch.Tensor, spatial_grid: torch.Tensor
    ) -> torch.Tensor:
        """Forecast the IV surface.

        Parameters
        ----------
        temporal_seq : torch.Tensor
            Shape ``(batch, seq_len, temporal_input_dim)``.
        spatial_grid : torch.Tensor
            Shape ``(batch, grid_channels, moneyness_bins, dte_bins)``.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, grid_h, grid_w)`` — forecasted IV surface.
        """
        # Temporal
        _, (h_n, _) = self.lstm(temporal_seq)
        lstm_out = h_n[-1]  # last layer hidden state → (B, hidden_dim)

        # Spatial
        cnn_out = self.cnn(spatial_grid)
        cnn_out = cnn_out.view(cnn_out.size(0), -1)  # (B, 32*4*4)

        # Fuse & reshape
        fused = torch.cat([lstm_out, cnn_out], dim=1)
        surface = self.fusion(fused)
        return surface.view(-1, self.grid_h, self.grid_w)
