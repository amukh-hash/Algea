"""
Tests for Lag-Llama series builder (squared returns, EWMA baseline).
"""
import numpy as np
import pandas as pd
import pytest

from algaie.models.tsfm.lag_llama.series_builder import (
    build_abs_neg_ret_series,
    build_ewma_variance,
    build_log_returns,
    build_sqret_series,
    ewma_rv_quantiles,
)


def _make_close(n: int = 500, seed: int = 42) -> pd.Series:
    """Generate synthetic close prices."""
    rng = np.random.RandomState(seed)
    returns = rng.normal(0.0005, 0.012, n)
    prices = 100.0 * np.exp(np.cumsum(returns))
    idx = pd.bdate_range("2022-01-03", periods=n)
    return pd.Series(prices, index=idx, name="close")


class TestSqretSeries:
    def test_deterministic(self):
        close = _make_close()
        s1 = build_sqret_series(close)
        s2 = build_sqret_series(close)
        pd.testing.assert_series_equal(s1, s2)

    def test_nonnegative(self):
        s = build_sqret_series(_make_close())
        assert (s >= 0).all()

    def test_clipped(self):
        close = _make_close()
        lr = build_log_returns(close)
        raw_sqret = lr ** 2
        clip_val = np.nanpercentile(raw_sqret.dropna().values, 95.0)
        s = build_sqret_series(close, upper_clip_percentile=95.0)
        # Clipped values should be at or below the clip boundary
        assert s.max() <= clip_val + 1e-8

    def test_length(self):
        close = _make_close(n=100)
        s = build_sqret_series(close)
        assert len(s) == 99  # one less due to diff


class TestAbsNegRet:
    def test_nonnegative(self):
        s = build_abs_neg_ret_series(_make_close())
        assert (s >= 0).all()

    def test_positive_returns_are_zero(self):
        close = _make_close()
        lr = build_log_returns(close)
        neg = build_abs_neg_ret_series(close)
        # Where returns are positive, abs_neg_ret should be 0
        assert (neg[lr > 0] == 0).all()


class TestEWMABaseline:
    def test_output_nonnegative(self):
        ewma = build_ewma_variance(_make_close())
        assert (ewma.dropna() >= 0).all()

    def test_quantiles_monotone(self):
        close = _make_close(n=500)
        ewma = build_ewma_variance(close)
        qs = ewma_rv_quantiles(ewma)
        vals = [qs[q] for q in sorted(qs.keys())]
        for i in range(1, len(vals)):
            assert vals[i] >= vals[i - 1] - 1e-8
