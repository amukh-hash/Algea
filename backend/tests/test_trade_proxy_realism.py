"""Tests for trade proxy realism (Deliverable C)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sleeves.cooc_reversal_futures.pipeline.types import TradeProxyConfig, TradeProxyRealism


def _make_panel(n_instruments: int = 6, n_days: int = 10) -> pd.DataFrame:
    """Create synthetic panel data with root, r_oc, r_co columns."""
    np.random.seed(42)
    roots = ["ES", "NQ", "YM", "RTY", "GC", "CL"][:n_instruments]
    rows = []
    for d in range(n_days):
        day = f"2025-01-{10 + d:02d}"
        for root in roots:
            r_co = np.random.randn() * 0.02
            r_oc = np.random.randn() * 0.03
            rows.append({
                "trading_day": day,
                "instrument": root,
                "root": root,
                "r_co": r_co,
                "r_oc": r_oc,
                "y": r_oc,
                "close": 5000.0 if root == "ES" else 18000.0,
                "multiplier": 50.0 if root == "ES" else 20.0,
            })
    return pd.DataFrame(rows)


class TestPerRootCosts:
    def test_different_costs_per_root(self):
        from sleeves.cooc_reversal_futures.pipeline.types import TradeProxyRealism

        realism = TradeProxyRealism(
            cost_per_contract_by_root={"ES": 1.0, "NQ": 3.0},
        )
        assert realism.cost_for_root("ES") == 1.0
        assert realism.cost_for_root("NQ") == 3.0
        assert realism.cost_for_root("YM") == 2.5  # fallback

    def test_per_root_slippage(self):
        from sleeves.cooc_reversal_futures.pipeline.types import TradeProxyRealism

        realism = TradeProxyRealism(
            slippage_bps_open_by_root={"ES": 0.5, "NQ": 2.0},
        )
        assert realism.slippage_open_for_root("ES") == 0.5
        assert realism.slippage_open_for_root("NQ") == 2.0
        assert realism.slippage_open_for_root("CL") == 1.0  # fallback


class TestShockMultiplier:
    def test_shock_increases_slippage(self):
        from sleeves.cooc_reversal_futures.pipeline.trade_proxy import _daily_proxy_return

        df = _make_panel(n_instruments=4, n_days=1)
        # Add shock_flag
        df["shock_flag"] = 1.0
        df["pred"] = np.random.randn(len(df))

        cfg = TradeProxyConfig(
            cost_per_contract=2.5,
            slippage_bps_open=1.0,
            slippage_bps_close=1.0,
            gross_target=1.0,
        )
        realism_shock = TradeProxyRealism(shock_slippage_multiplier=3.0)
        realism_normal = TradeProxyRealism(shock_slippage_multiplier=1.0)

        ret_shock = _daily_proxy_return(
            df, "pred", "r_oc", cfg,
            realism=realism_shock,
        )
        ret_normal = _daily_proxy_return(
            df, "pred", "r_oc", cfg,
            realism=realism_normal,
        )
        # Shock should have higher costs, so net return should be lower
        assert ret_shock <= ret_normal


class TestPartialFillDeterminism:
    def test_deterministic(self):
        from sleeves.cooc_reversal_futures.pipeline.trade_proxy import _daily_proxy_return

        df = _make_panel(n_instruments=4, n_days=1)
        df["shock_flag"] = 1.0
        df["pred"] = np.random.randn(len(df))

        cfg = TradeProxyConfig(
            cost_per_contract=0.0,
            slippage_bps_open=0.0,
            slippage_bps_close=0.0,
            gross_target=1.0,
        )
        realism = TradeProxyRealism(
            partial_fill_prob_shock=1.0,  # always trigger
            partial_fill_seed=42,
        )

        r1 = _daily_proxy_return(
            df, "pred", "r_oc", cfg,
            realism=realism, day_hash=12345,
        )
        r2 = _daily_proxy_return(
            df, "pred", "r_oc", cfg,
            realism=realism, day_hash=12345,
        )
        assert r1 == r2, "Partial fill should be deterministic with same seed+hash"


class TestRealismSerialization:
    def test_to_dict(self):
        from sleeves.cooc_reversal_futures.pipeline.types import TradeProxyRealism

        realism = TradeProxyRealism(
            cost_per_contract_by_root={"ES": 1.0},
            shock_slippage_multiplier=2.5,
        )
        d = realism.to_dict()
        assert d["shock_slippage_multiplier"] == 2.5
        assert d["cost_per_contract_by_root"] == {"ES": 1.0}
