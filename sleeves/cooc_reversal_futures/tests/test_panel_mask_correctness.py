"""Tests for PanelDataset mask correctness with missing roots.

Covers:
- Padding/masking works when certain roots are missing on specific days
- Days with < min_instruments_per_day are correctly filtered out
- Mask correctly identifies padded vs real instruments
- Feature values are zero in padded positions
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
import pytest

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

from sleeves.cooc_reversal_futures.pipeline.panel_dataset import (
    PanelDataset,
    panel_collate_fn,
)


def _make_sparse_dataset(
    instruments: list[str],
    n_days: int = 50,
    missing_prob: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a dataset where instruments are randomly missing on some days."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-02", periods=n_days)
    features = ["f1", "f2", "f3"]

    rows = []
    for d in dates:
        for inst in instruments:
            if rng.random() < missing_prob:
                continue  # This instrument is missing on this day
            rows.append({
                "trading_day": d,
                "instrument": inst,
                **{f: rng.normal(0, 0.01) for f in features},
                "y": rng.normal(0, 0.005),
                "r_oc": rng.normal(0, 0.01),
            })
    return pd.DataFrame(rows)


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
class TestPanelMaskCorrectness:
    """Verify PanelDataset handles missing instruments correctly."""

    @pytest.fixture
    def instruments(self):
        return ["ES", "NQ", "YM", "RTY", "CL", "GC", "SI", "ZN",
                "ZB", "6E", "6J", "HG", "6B", "6A"]

    @pytest.fixture
    def sparse_dataset(self, instruments):
        return _make_sparse_dataset(instruments, n_days=50, missing_prob=0.15)

    def test_mask_true_only_for_present(self, instruments, sparse_dataset):
        """mask[i] should be True only for instruments actually present."""
        features = ["f1", "f2", "f3"]
        ds = PanelDataset(
            sparse_dataset, features,
            min_instruments_per_day=2,
            max_instruments=len(instruments),
        )

        for idx in range(min(10, len(ds))):
            item = ds[idx]
            n_inst = item["n_instruments"]
            mask = item["mask"]

            # First n_inst positions should be True
            assert mask[:n_inst].all(), f"Day {idx}: first {n_inst} mask positions should be True"
            # Remaining should be False (padded)
            if n_inst < len(instruments):
                assert not mask[n_inst:].any(), f"Day {idx}: padded positions should be False"

    def test_padded_features_are_zero(self, instruments, sparse_dataset):
        """Features in padded positions should be zero."""
        features = ["f1", "f2", "f3"]
        ds = PanelDataset(
            sparse_dataset, features,
            min_instruments_per_day=2,
            max_instruments=len(instruments),
        )

        for idx in range(min(10, len(ds))):
            item = ds[idx]
            n_inst = item["n_instruments"]
            X = item["X"]

            if n_inst < len(instruments):
                padded_X = X[n_inst:]
                assert (padded_X == 0).all(), f"Day {idx}: padded features should be zero"

    def test_min_instruments_filter(self, instruments, sparse_dataset):
        """Days with fewer than min_instruments_per_day should be excluded."""
        features = ["f1", "f2", "f3"]

        # Set high threshold — many days should be excluded
        ds_strict = PanelDataset(
            sparse_dataset, features,
            min_instruments_per_day=13,
            max_instruments=len(instruments),
        )
        ds_relaxed = PanelDataset(
            sparse_dataset, features,
            min_instruments_per_day=2,
            max_instruments=len(instruments),
        )

        assert len(ds_strict) <= len(ds_relaxed), (
            "Strict filter should have fewer days than relaxed"
        )

    def test_batch_collation(self, instruments, sparse_dataset):
        """panel_collate_fn should produce correct batched shapes."""
        features = ["f1", "f2", "f3"]
        ds = PanelDataset(
            sparse_dataset, features,
            min_instruments_per_day=2,
            max_instruments=len(instruments),
        )

        batch = [ds[i] for i in range(min(4, len(ds)))]
        collated = panel_collate_fn(batch)

        B = len(batch)
        N = len(instruments)
        F = len(features)

        assert collated["X"].shape == (B, N, F)
        assert collated["y"].shape == (B, N)
        assert collated["mask"].shape == (B, N)
        assert len(collated["trading_days"]) == B
        assert collated["n_instruments"].shape == (B,)

    def test_fully_missing_day_excluded(self, instruments):
        """A day where no instruments have data should be excluded."""
        features = ["f1", "f2", "f3"]

        # Create dataset with one day having all instruments and one with none
        rows = []
        for inst in instruments:
            rows.append({
                "trading_day": pd.Timestamp("2024-01-02"),
                "instrument": inst,
                "f1": 0.01, "f2": 0.02, "f3": 0.03,
                "y": 0.001,
            })
        # Day 2024-01-03: only 1 instrument (below min_instruments_per_day=2)
        rows.append({
            "trading_day": pd.Timestamp("2024-01-03"),
            "instrument": "ES",
            "f1": 0.01, "f2": 0.02, "f3": 0.03,
            "y": 0.001,
        })
        df = pd.DataFrame(rows)

        ds = PanelDataset(df, features, min_instruments_per_day=2, max_instruments=len(instruments))
        assert len(ds) == 1, "Only the first day should pass the min_instruments filter"

    def test_deterministic_instrument_ordering(self, instruments, sparse_dataset):
        """Instruments should be sorted alphabetically within each panel."""
        features = ["f1", "f2", "f3"]
        ds = PanelDataset(
            sparse_dataset, features,
            min_instruments_per_day=2,
            max_instruments=len(instruments),
        )

        for idx in range(min(5, len(ds))):
            day, panel_instruments, *_ = ds._panels[idx]
            assert panel_instruments == sorted(panel_instruments), (
                f"Day {day}: instruments not sorted: {panel_instruments}"
            )
