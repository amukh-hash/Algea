"""Test NaN-free invariant for IV grid interpolation.

Validates that the cubic spline interpolation produces zero NaN values
in the 10×10 IV grid, even with sparse or missing input data.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.data_canonical.ingestion_vrp import (
    IVGridIntegrityError,
    _interpolate_iv_grid,
    process_options_chain,
)


class TestNaNInvariant:
    def test_dense_input_no_nan(self):
        """Dense IV data produces zero NaN in the output grid."""
        rng = np.random.default_rng(42)
        n_points = 200
        moneyness = rng.uniform(0.8, 1.2, n_points).astype(np.float32)
        dtes = rng.uniform(1, 90, n_points).astype(np.float32)
        ivs = rng.uniform(0.1, 0.5, n_points).astype(np.float32)

        grid = _interpolate_iv_grid(moneyness, dtes, ivs)

        assert grid.shape == (10, 10)
        assert grid.dtype == np.float32
        assert np.isnan(grid).sum() == 0, "NaN found in IV grid"
        assert np.isinf(grid).sum() == 0, "Inf found in IV grid"

    def test_sparse_input_no_nan(self):
        """Sparse IV data (only 10 points) still produces zero NaN via fallback."""
        rng = np.random.default_rng(42)
        n_points = 10
        moneyness = np.linspace(0.85, 1.15, n_points, dtype=np.float32)
        dtes = np.linspace(5, 80, n_points, dtype=np.float32)
        ivs = rng.uniform(0.15, 0.4, n_points).astype(np.float32)

        grid = _interpolate_iv_grid(moneyness, dtes, ivs)

        assert grid.shape == (10, 10)
        assert np.isnan(grid).sum() == 0, "NaN found in sparse IV grid"

    def test_clustered_input_no_nan(self):
        """Data clustered in one corner of the grid still interpolates without NaN."""
        rng = np.random.default_rng(42)
        n_points = 30
        # All points clustered near ATM short-dated
        moneyness = rng.uniform(0.95, 1.05, n_points).astype(np.float32)
        dtes = rng.uniform(1, 15, n_points).astype(np.float32)
        ivs = rng.uniform(0.15, 0.35, n_points).astype(np.float32)

        grid = _interpolate_iv_grid(moneyness, dtes, ivs)

        assert grid.shape == (10, 10)
        assert np.isnan(grid).sum() == 0

    def test_iv_values_physically_reasonable(self):
        """All IV values should be clamped to [0.01, 5.0]."""
        rng = np.random.default_rng(42)
        n_points = 100
        moneyness = rng.uniform(0.8, 1.2, n_points).astype(np.float32)
        dtes = rng.uniform(1, 90, n_points).astype(np.float32)
        ivs = rng.uniform(0.05, 0.8, n_points).astype(np.float32)

        grid = _interpolate_iv_grid(moneyness, dtes, ivs)

        assert grid.min() >= 0.01, f"IV below 0.01: {grid.min()}"
        assert grid.max() <= 5.0, f"IV above 5.0: {grid.max()}"

    def test_insufficient_points_raises(self):
        """Fewer than 4 valid points should raise IVGridIntegrityError."""
        moneyness = np.array([1.0, 1.05], dtype=np.float32)
        dtes = np.array([30, 60], dtype=np.float32)
        ivs = np.array([0.2, 0.25], dtype=np.float32)

        with pytest.raises(IVGridIntegrityError, match="Insufficient"):
            _interpolate_iv_grid(moneyness, dtes, ivs)

    def test_nan_inputs_filtered(self):
        """NaN values in raw input should be filtered before interpolation."""
        rng = np.random.default_rng(42)
        n_points = 100
        moneyness = rng.uniform(0.8, 1.2, n_points).astype(np.float32)
        dtes = rng.uniform(1, 90, n_points).astype(np.float32)
        ivs = rng.uniform(0.1, 0.5, n_points).astype(np.float32)

        # Inject NaN values
        ivs[0:10] = np.nan
        moneyness[50:55] = np.nan

        grid = _interpolate_iv_grid(moneyness, dtes, ivs)

        assert grid.shape == (10, 10)
        assert np.isnan(grid).sum() == 0, "NaN propagated from input to output"

    def test_process_options_chain(self):
        """Integration test: process_options_chain produces valid grid."""
        import pandas as pd

        rng = np.random.default_rng(42)
        n = 50
        underlying = 450.0
        df = pd.DataFrame({
            "strike": rng.uniform(360, 540, n),
            "dte": rng.uniform(1, 90, n).astype(int),
            "iv": rng.uniform(0.1, 0.6, n),
        })

        grid = process_options_chain(df, underlying)
        assert grid.shape == (10, 10)
        assert np.isnan(grid).sum() == 0
