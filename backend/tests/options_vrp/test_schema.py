"""Tests for option chain schema validation."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from algae.data.options.schema import ChainValidationError, validate_chain


def _make_chain(n: int = 20, **overrides) -> pd.DataFrame:
    """Build a valid synthetic option chain."""
    strikes = np.linspace(380, 420, n)
    bids = np.maximum(0.5, np.random.uniform(0.5, 5.0, n))
    asks = bids + np.random.uniform(0.05, 0.5, n)
    mids = (bids + asks) / 2
    data = {
        "date": [date(2024, 6, 1)] * n,
        "underlying": ["SPY"] * n,
        "underlying_price": [400.0] * n,
        "expiry": [date(2024, 7, 1)] * n,
        "dte": [30] * n,
        "option_type": ["put"] * (n // 2) + ["call"] * (n - n // 2),
        "strike": strikes,
        "bid": bids,
        "ask": asks,
        "mid": mids,
        "implied_vol": np.random.uniform(0.15, 0.35, n),
        "open_interest": np.random.randint(100, 5000, n),
        "volume": np.random.randint(10, 1000, n),
        "multiplier": [100] * n,
        "risk_free_rate": [0.05] * n,
        "dividend_yield": [0.013] * n,
    }
    data.update(overrides)
    return pd.DataFrame(data)


class TestSchemaValidation:
    def test_valid_chain_passes(self):
        df = _make_chain()
        result = validate_chain(df)
        assert len(result) == len(df)

    def test_missing_column_fails(self):
        df = _make_chain().drop(columns=["implied_vol"])
        with pytest.raises(ChainValidationError, match="Missing columns"):
            validate_chain(df)

    def test_null_iv_fails(self):
        df = _make_chain()
        df.loc[0, "implied_vol"] = np.nan
        with pytest.raises(ChainValidationError, match="null implied_vol"):
            validate_chain(df)

    def test_extreme_iv_fails(self):
        df = _make_chain()
        df.loc[0, "implied_vol"] = 6.0  # above 5.0 limit
        with pytest.raises(ChainValidationError, match="outside"):
            validate_chain(df)

    def test_inverted_bid_ask_fails(self):
        df = _make_chain()
        df.loc[0, "bid"] = 10.0
        df.loc[0, "ask"] = 5.0
        df.loc[0, "mid"] = 7.5
        with pytest.raises(ChainValidationError, match="bid > ask"):
            validate_chain(df)

    def test_mid_outside_bid_ask_fails(self):
        df = _make_chain()
        df.loc[0, "mid"] = df.loc[0, "ask"] + 1.0
        with pytest.raises(ChainValidationError, match="mid outside"):
            validate_chain(df)

    def test_no_dte_coverage_fails(self):
        df = _make_chain()
        df["dte"] = 10  # below min_dte=30
        with pytest.raises(ChainValidationError, match="No expiries"):
            validate_chain(df, require_dte_coverage=True)

    def test_sparse_chain_fails(self):
        df = _make_chain(n=3)  # fewer than 5 strikes
        with pytest.raises(ChainValidationError, match="Sparse"):
            validate_chain(df, require_dte_coverage=False)

    def test_empty_chain_fails(self):
        df = _make_chain().iloc[:0]
        with pytest.raises(ChainValidationError, match="empty"):
            validate_chain(df)
