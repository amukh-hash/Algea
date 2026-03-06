"""StrictEmbargoDataset — Temporal quarantine for PatchTST training.

Implements a strict purge-and-embargo protocol at the Train/Validation
boundary to prevent data leakage through overlapping PatchTST strides.

The embargo physically drops ``seq_len + stride`` rows at the split
boundary, severing any possible information flow from the validation
set backward into the final training patches.
"""
from __future__ import annotations

import logging
from typing import Optional

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class StrictEmbargoDataset(Dataset):
    """Time-series dataset with strict temporal embargo at split boundary.

    Parameters
    ----------
    data : torch.Tensor
        Shape ``(total_time_steps, n_features)`` — full chronological data.
    targets : torch.Tensor
        Shape ``(total_time_steps,)`` or ``(total_time_steps, target_dim)``
        — forward returns or prediction targets.
    split_idx : int
        The chronological index where train ends and validation begins.
    seq_len : int
        PatchTST context length (lookback window).
    stride : int
        PatchTST patch stride.
    is_val : bool
        If True, this dataset produces validation samples (post-embargo).
        If False, produces training samples (pre-embargo).
    forecast_horizon : int
        Number of forward steps in the target. Used to ensure no target
        leaks across the embargo boundary.
    """

    def __init__(
        self,
        data: torch.Tensor,
        targets: torch.Tensor,
        split_idx: int,
        seq_len: int,
        stride: int,
        is_val: bool,
        forecast_horizon: int = 1,
    ):
        super().__init__()

        if data.shape[0] != targets.shape[0]:
            raise ValueError(
                f"data and targets must have same length along dim 0. "
                f"Got data={data.shape[0]}, targets={targets.shape[0]}"
            )

        # The embargo offset physically severs the overlap region
        embargo_offset = seq_len + stride + forecast_horizon
        total_len = data.shape[0]

        if is_val:
            # Validation: start after the embargo zone
            val_start = split_idx + embargo_offset
            if val_start >= total_len:
                raise ValueError(
                    f"Embargo offset ({embargo_offset}) extends beyond data length "
                    f"({total_len}). split_idx={split_idx}. "
                    "Increase dataset size or reduce seq_len/stride."
                )
            self.data = data[val_start:]
            self.targets = targets[val_start:]
            self._offset = val_start
            logger.info(
                "EMBARGO VAL  split=%d, embargo=%d, val_start=%d, val_len=%d, "
                "DROPPED %d rows at boundary",
                split_idx, embargo_offset, val_start, len(self.data),
                embargo_offset,
            )
        else:
            # Training: end before the split index
            self.data = data[:split_idx]
            self.targets = targets[:split_idx]
            self._offset = 0
            logger.info(
                "EMBARGO TRAIN split=%d, train_len=%d",
                split_idx, len(self.data),
            )

        self.seq_len = seq_len
        self.stride = stride
        self.is_val = is_val
        self.forecast_horizon = forecast_horizon

        # Compute valid sample indices
        # Each sample needs seq_len lookback + forecast_horizon forward
        min_start = seq_len
        max_start = len(self.data) - forecast_horizon
        if max_start <= min_start:
            raise ValueError(
                f"Not enough data for sampling: len={len(self.data)}, "
                f"seq_len={seq_len}, horizon={forecast_horizon}"
            )

        self._indices = list(range(min_start, max_start))

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a (sequence, target) pair.

        Returns
        -------
        x : torch.Tensor
            Shape ``(seq_len, n_features)`` — lookback window.
        y : torch.Tensor
            Shape ``(forecast_horizon,)`` or ``(forecast_horizon, target_dim)``
            — forward targets.
        """
        t = self._indices[idx]
        x = self.data[t - self.seq_len : t]
        y = self.targets[t : t + self.forecast_horizon]

        # Squeeze single-step horizon for convenience
        if self.forecast_horizon == 1 and y.dim() > 0:
            y = y.squeeze(0)

        return x, y

    @property
    def embargo_rows_dropped(self) -> int:
        """Number of chronological rows dropped at the boundary."""
        return self.seq_len + self.stride + self.forecast_horizon

    @staticmethod
    def create_train_val_pair(
        data: torch.Tensor,
        targets: torch.Tensor,
        train_fraction: float = 0.8,
        seq_len: int = 512,
        stride: int = 8,
        forecast_horizon: int = 1,
    ) -> tuple["StrictEmbargoDataset", "StrictEmbargoDataset"]:
        """Factory method to create matched train/val datasets.

        Parameters
        ----------
        data : torch.Tensor
            Full chronological feature tensor.
        targets : torch.Tensor
            Full chronological target tensor.
        train_fraction : float
            Fraction of data for training (by chronological index).
        seq_len, stride, forecast_horizon : int
            PatchTST and prediction parameters.

        Returns
        -------
        (train_dataset, val_dataset)
        """
        split_idx = int(len(data) * train_fraction)
        logger.info(
            "SPLIT data(%d) at idx %d (%.1f%% train, %.1f%% val+embargo)",
            len(data), split_idx,
            train_fraction * 100, (1 - train_fraction) * 100,
        )

        train_ds = StrictEmbargoDataset(
            data, targets, split_idx, seq_len, stride,
            is_val=False, forecast_horizon=forecast_horizon,
        )
        val_ds = StrictEmbargoDataset(
            data, targets, split_idx, seq_len, stride,
            is_val=True, forecast_horizon=forecast_horizon,
        )

        return train_ds, val_ds
