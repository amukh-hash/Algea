"""Path-Signature friction model for LOB execution.

Uses Rough Path Theory signature transforms to encode high-frequency
path-dependency of the Limit Order Book (bid/ask/depth/flow) and
estimate execution friction costs.
"""
from __future__ import annotations

import torch
import torch.nn as nn

try:
    import signatory
except ImportError:
    signatory = None  # type: ignore[assignment]


class SignatureFrictionModel(nn.Module):
    """Estimate execution friction from LOB path signatures.

    Parameters
    ----------
    channels : int
        Number of channels in the LOB path
        (e.g. bid, ask, depth, trade-flow = 4).
    depth : int
        Truncation depth for the path signature.
    output_dim : int
        Number of friction-cost outputs (e.g. per-asset).
    """

    def __init__(self, channels: int, depth: int, output_dim: int):
        super().__init__()
        self.channels = channels
        self.depth = depth

        # Signature feature dimension
        if signatory is not None:
            self.sig_channels = signatory.signature_channels(channels, depth)
        else:
            # Heuristic approximation for the fallback path
            self.sig_channels = sum(channels ** d for d in range(1, depth + 1))

        self.impact_estimator = nn.Sequential(
            nn.Linear(self.sig_channels, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Softplus(),  # friction is strictly positive
        )

    def forward(self, lob_path: torch.Tensor) -> torch.Tensor:
        """Estimate friction cost from an LOB path.

        Parameters
        ----------
        lob_path : torch.Tensor
            Shape ``(batch, timesteps, channels)`` containing
            bid, ask, depth, trade-flow (or similar).

        Returns
        -------
        torch.Tensor
            Shape ``(batch, output_dim)`` — strictly positive friction
            cost estimates.
        """
        if signatory is not None:
            sig = signatory.signature(lob_path, self.depth)
        else:
            # Fallback: use flattened time-mean as a stand-in for the
            # signature feature when the ``signatory`` library is not
            # installed (CI / testing).
            mean_features = lob_path.mean(dim=1)  # (B, channels)
            # Expand to match expected sig dimension
            repeats = (self.sig_channels + self.channels - 1) // self.channels
            sig = mean_features.repeat(1, repeats)[:, : self.sig_channels]

        return self.impact_estimator(sig)
