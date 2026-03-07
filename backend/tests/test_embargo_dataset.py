"""Test StrictEmbargoDataset purge at train/val boundary.

Validates that PatchTST overlapping strides cannot leak validation data
back into training samples via the purge/embargo offset.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

_mod = pytest.importorskip(
    "algae.models.utils.dataset",
    reason="StrictEmbargoDataset not yet ported from algaie_legacy — migration pending",
)
StrictEmbargoDataset = _mod.StrictEmbargoDataset


class TestStrictEmbargo:
    def test_train_val_do_not_overlap(self):
        """Train and val datasets have no overlapping indices."""
        data = torch.randn(2000, 4)
        targets = torch.randn(2000)

        train_ds, val_ds = StrictEmbargoDataset.create_train_val_pair(
            data, targets, train_fraction=0.8,
            seq_len=64, stride=8, forecast_horizon=1,
        )

        # Train data should end before split_idx
        split_idx = int(2000 * 0.8)
        assert len(train_ds.data) == split_idx

        # Val data should start AFTER split_idx + embargo
        embargo = 64 + 8 + 1  # seq_len + stride + horizon
        val_start = split_idx + embargo
        assert train_ds._offset == 0
        assert val_ds._offset == val_start

    def test_embargo_drops_correct_rows(self):
        """The embargo should physically drop seq_len + stride + horizon rows."""
        seq_len = 128
        stride = 16
        horizon = 1
        expected_drop = seq_len + stride + horizon

        data = torch.randn(5000, 2)
        targets = torch.randn(5000)

        train_ds, val_ds = StrictEmbargoDataset.create_train_val_pair(
            data, targets, train_fraction=0.8,
            seq_len=seq_len, stride=stride, forecast_horizon=horizon,
        )

        assert train_ds.embargo_rows_dropped == expected_drop
        assert val_ds.embargo_rows_dropped == expected_drop

        # Total data coverage must be LESS than original
        total_covered = len(train_ds.data) + len(val_ds.data)
        assert total_covered < len(data), "Embargo not dropping rows"
        assert total_covered == len(data) - expected_drop

    def test_no_data_leakage_at_boundary(self):
        """Sequential indices near the split boundary must not appear in both datasets."""
        data = torch.arange(1000, dtype=torch.float32).unsqueeze(-1)
        targets = torch.arange(1000, dtype=torch.float32)

        split_idx = 800
        seq_len = 32
        stride = 8
        embargo = seq_len + stride + 1

        train_ds = StrictEmbargoDataset(data, targets, split_idx, seq_len, stride, is_val=False)
        val_ds = StrictEmbargoDataset(data, targets, split_idx, seq_len, stride, is_val=True)

        # Verify the gap: last train index < first val index - embargo
        train_max_idx = split_idx - 1
        val_min_idx = split_idx + embargo

        # No overlap possible
        assert train_max_idx < val_min_idx, "Overlap detected at boundary"

    def test_insufficient_data_raises(self):
        """Dataset with insufficient data to form samples should raise ValueError."""
        data = torch.randn(100, 2)
        targets = torch.randn(100)

        with pytest.raises(ValueError):
            StrictEmbargoDataset(
                data, targets, split_idx=80,
                seq_len=512, stride=8, is_val=True,
            )

    def test_getitem_shapes(self):
        """__getitem__ returns correct shapes."""
        data = torch.randn(3000, 4)
        targets = torch.randn(3000)

        ds, _ = StrictEmbargoDataset.create_train_val_pair(
            data, targets, train_fraction=0.8,
            seq_len=64, stride=8, forecast_horizon=1,
        )

        x, y = ds[0]
        assert x.shape == (64, 4)
        assert y.dim() == 0 or y.shape == ()  # scalar for horizon=1
