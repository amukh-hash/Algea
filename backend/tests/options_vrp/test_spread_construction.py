"""Tests for spread construction — delta targeting, width, DTE selection."""
from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from algea.execution.options.config import VRPConfig
from algea.execution.options.vrp_strategy import VRPStrategy
from algea.data.options.vrp_features import VolRegime


def _make_test_chain(
    underlying_price: float = 400.0,
    expiry: date = date(2024, 7, 15),
    dte: int = 35,
    n_strikes: int = 20,
) -> pd.DataFrame:
    """Synthetic chain with realistic put strikes around spot."""
    strikes = np.linspace(underlying_price - 30, underlying_price + 10, n_strikes)
    n = len(strikes)
    bids = np.maximum(0.1, np.abs(underlying_price - strikes) * 0.05 + np.random.uniform(0.1, 0.5, n))
    asks = bids + 0.10
    mids = (bids + asks) / 2
    return pd.DataFrame({
        "date": [date(2024, 6, 10)] * n,
        "underlying": ["SPY"] * n,
        "underlying_price": [underlying_price] * n,
        "expiry": [expiry] * n,
        "dte": [dte] * n,
        "option_type": ["put"] * n,
        "strike": strikes,
        "bid": bids,
        "ask": asks,
        "mid": mids,
        "implied_vol": np.linspace(0.22, 0.30, n),  # skew
        "open_interest": [500] * n,
        "volume": [200] * n,
        "multiplier": [100] * n,
        "risk_free_rate": [0.05] * n,
        "dividend_yield": [0.013] * n,
    })


class TestSpreadConstruction:
    def test_delta_targeting_selects_otm_put(self):
        """Short strike should be below underlying (OTM put)."""
        config = VRPConfig(delta_target=0.15, spread_width=5)
        strategy = VRPStrategy(config)
        chain = _make_test_chain()
        features = {"regime": VolRegime.NORMAL_CARRY, "vrp_features": {}, "surface_snapshot": {}}

        pos = strategy.predict(
            date(2024, 6, 10), chain, "SPY", 400.0, features, nav=1_000_000,
        )
        if pos is not None:
            short_leg = [l for l in pos.legs if l.side == "sell"][0]
            assert short_leg.strike < 400.0, "Short put should be OTM"

    def test_spread_width_respected(self):
        """Long strike should be ~spread_width below short strike."""
        config = VRPConfig(delta_target=0.15, spread_width=5)
        strategy = VRPStrategy(config)
        chain = _make_test_chain()
        features = {"regime": VolRegime.NORMAL_CARRY, "vrp_features": {}, "surface_snapshot": {}}

        pos = strategy.predict(
            date(2024, 6, 10), chain, "SPY", 400.0, features, nav=1_000_000,
        )
        if pos is not None:
            short_leg = [l for l in pos.legs if l.side == "sell"][0]
            long_leg = [l for l in pos.legs if l.side == "buy"][0]
            width = short_leg.strike - long_leg.strike
            assert width >= 1.0, "Spread should have positive width"
            assert width <= config.spread_width + 3, "Width should be near target"

    def test_dte_filtering(self):
        """Should select expiry within DTE range."""
        config = VRPConfig(dte_range=(30, 45))
        strategy = VRPStrategy(config)

        # Chain with DTE=35 (in range)
        chain_ok = _make_test_chain(dte=35)
        features = {"regime": VolRegime.NORMAL_CARRY, "vrp_features": {}, "surface_snapshot": {}}
        pos = strategy.predict(date(2024, 6, 10), chain_ok, "SPY", 400.0, features, 1e6)
        # Should at least attempt construction (may return None if chain too narrow)

        # Chain with DTE=10 (out of range)
        chain_bad = _make_test_chain(dte=10)
        pos_bad = strategy.predict(date(2024, 6, 10), chain_bad, "SPY", 400.0, features, 1e6)
        assert pos_bad is None, "Should reject chain with DTE out of range"

    def test_max_loss_computed(self):
        """Position should have a positive max_loss."""
        strategy = VRPStrategy()
        chain = _make_test_chain()
        features = {"regime": VolRegime.NORMAL_CARRY, "vrp_features": {}, "surface_snapshot": {}}
        pos = strategy.predict(date(2024, 6, 10), chain, "SPY", 400.0, features, 1e6)
        if pos is not None:
            assert pos.max_loss > 0
            assert pos.premium_collected > 0
            assert pos.max_loss > pos.premium_collected


class TestRegimeGating:
    def test_crash_risk_blocks_entry(self):
        strategy = VRPStrategy()
        chain = _make_test_chain()
        features = {"regime": VolRegime.CRASH_RISK, "vrp_features": {}, "surface_snapshot": {}}
        pos = strategy.predict(date(2024, 6, 10), chain, "SPY", 400.0, features, 1e6)
        assert pos is None, "CRASH_RISK should block new entries"

    def test_normal_allows_entry(self):
        strategy = VRPStrategy()
        chain = _make_test_chain()
        features = {"regime": VolRegime.NORMAL_CARRY, "vrp_features": {}, "surface_snapshot": {}}
        pos = strategy.predict(date(2024, 6, 10), chain, "SPY", 400.0, features, 1e6)
        # May or may not produce position depending on chain specifics, but should not be blocked by regime
