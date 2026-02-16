"""Tests for R7: Paper-trading shadow harness."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from algaie.execution.shadow_runner import (
    ShadowRunConfig,
    ShadowReport,
    shadow_evaluate,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_orders(n: int = 30, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic model orders."""
    rng = np.random.RandomState(seed)
    days = pd.bdate_range("2025-01-02", periods=n, freq="B")
    return pd.DataFrame({
        "trading_day": days,
        "root": "ES",
        "side": rng.choice(["BUY", "SELL"], size=n),
        "target_qty": 1,
        "assumed_slippage_bps": 1.0,
    })


def _make_fills(orders: pd.DataFrame, *, mismatch_frac: float = 0.0, seed: int = 99) -> pd.DataFrame:
    """Generate fills matching orders, with optional mismatch."""
    rng = np.random.RandomState(seed)
    n = len(orders)
    fills = orders[["trading_day", "root", "side"]].copy()
    fills["filled_qty"] = 1
    fills["fill_price"] = 5000.0 + rng.randn(n) * 10
    fills["realized_slippage_bps"] = 1.0 + rng.randn(n) * 0.5
    fills["realized_pnl"] = rng.randn(n) * 100  # noisy pnl

    if mismatch_frac > 0:
        n_drop = max(int(n * mismatch_frac), 1)
        fills = fills.iloc[n_drop:]

    return fills.reset_index(drop=True)


def _make_proxy_daily(orders: pd.DataFrame, seed: int = 77) -> pd.DataFrame:
    """Generate proxy daily returns aligned with orders."""
    rng = np.random.RandomState(seed)
    days = orders["trading_day"].unique()
    return pd.DataFrame({
        "trading_day": days,
        "proxy_return": rng.randn(len(days)) * 0.002,
    })


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestShadowRunConfig:
    def test_defaults(self):
        cfg = ShadowRunConfig()
        assert cfg.min_days == 20
        assert cfg.max_slippage_drift_bps == 5.0

    def test_to_dict_roundtrip(self):
        cfg = ShadowRunConfig(min_days=10)
        d = cfg.to_dict()
        assert d["min_days"] == 10
        assert isinstance(d, dict)


class TestShadowReport:
    def test_default_report(self):
        r = ShadowReport()
        assert r.n_days == 0
        assert r.promotion_gate_passed is False

    def test_to_dict(self):
        r = ShadowReport(n_days=50, realized_sharpe=1.2)
        d = r.to_dict()
        assert d["n_days"] == 50
        assert d["realized_sharpe"] == 1.2


class TestShadowEvaluateBasic:
    def test_basic_run(self):
        """End-to-end with all fills matching."""
        orders = _make_orders(n=30)
        fills = _make_fills(orders)
        proxy = _make_proxy_daily(orders)
        report = shadow_evaluate(orders, fills, proxy)
        assert report.n_days == 30
        assert report.fill_mismatch_rate == 0.0
        assert isinstance(report.realized_sharpe, float)
        assert isinstance(report.assumed_sharpe, float)

    def test_gate_details_populated(self):
        orders = _make_orders(n=30)
        fills = _make_fills(orders)
        proxy = _make_proxy_daily(orders)
        report = shadow_evaluate(orders, fills, proxy)
        assert "min_days" in report.gate_details
        assert "realized_sharpe" in report.gate_details
        assert "fill_mismatch" in report.gate_details
        assert "slippage_drift" in report.gate_details
        assert "realized_worst_1pct" in report.gate_details


class TestShadowMismatch:
    def test_high_mismatch_fails(self):
        """Many unmatched fills → fill_mismatch gate fails."""
        orders = _make_orders(n=30)
        fills = _make_fills(orders, mismatch_frac=0.5)
        proxy = _make_proxy_daily(orders)
        cfg = ShadowRunConfig(fill_mismatch_max=0.10, min_days=1)
        report = shadow_evaluate(orders, fills, proxy, cfg=cfg)
        assert report.fill_mismatch_rate > 0.10
        assert report.gate_details["fill_mismatch"]["passed"] is False

    def test_zero_mismatch_passes(self):
        orders = _make_orders(n=30)
        fills = _make_fills(orders, mismatch_frac=0.0)
        proxy = _make_proxy_daily(orders)
        report = shadow_evaluate(orders, fills, proxy)
        assert report.fill_mismatch_rate == 0.0
        assert report.gate_details["fill_mismatch"]["passed"] is True


class TestShadowPromotion:
    def test_insufficient_days_fails(self):
        """Too few days → min_days gate fails."""
        orders = _make_orders(n=5)
        fills = _make_fills(orders)
        proxy = _make_proxy_daily(orders)
        cfg = ShadowRunConfig(min_days=20)
        report = shadow_evaluate(orders, fills, proxy, cfg=cfg)
        assert report.gate_details["min_days"]["passed"] is False
        assert report.promotion_gate_passed is False

    def test_composite_gate_all_pass(self):
        """Generous thresholds → promotion passes."""
        orders = _make_orders(n=30)
        fills = _make_fills(orders)
        proxy = _make_proxy_daily(orders)
        cfg = ShadowRunConfig(
            min_days=5,
            promotion_sharpe_min=-99.0,
            promotion_worst1_min=-999.0,
            fill_mismatch_max=1.0,
            max_slippage_drift_bps=999.0,
        )
        report = shadow_evaluate(orders, fills, proxy, cfg=cfg)
        assert report.promotion_gate_passed is True


class TestShadowSlippageDrift:
    def test_drift_computed(self):
        orders = _make_orders(n=30)
        fills = _make_fills(orders)
        proxy = _make_proxy_daily(orders)
        report = shadow_evaluate(orders, fills, proxy)
        # Drift should be non-zero since realized slippage has noise
        assert isinstance(report.mean_slippage_drift_bps, float)


class TestShadowEmptyInputs:
    def test_no_matching_days(self):
        """When fills don't overlap with orders → empty report."""
        orders = _make_orders(n=5)
        # Create fills with completely different days
        fills = pd.DataFrame({
            "trading_day": pd.bdate_range("2030-01-02", periods=5, freq="B"),
            "root": "ES",
            "side": "BUY",
            "filled_qty": 1,
            "fill_price": 5000.0,
            "realized_slippage_bps": 1.0,
            "realized_pnl": 0.0,
        })
        proxy = _make_proxy_daily(orders)
        report = shadow_evaluate(orders, fills, proxy)
        # All orders unmatched
        assert report.fill_mismatch_rate == 1.0
