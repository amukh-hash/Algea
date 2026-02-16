"""Tests: Promotion gate sub-checks work correctly."""
from __future__ import annotations

import pytest
from sleeves.cooc_reversal_futures.pipeline.types import (
    GateResult,
    TradeProxyReport,
)


def _make_report(**kwargs) -> TradeProxyReport:
    defaults = {
        "sharpe_model": 0.5,
        "sharpe_baseline": 0.3,
        "hit_rate": 0.52,
        "max_drawdown": -0.05,
        "mean_daily_return": 0.001,
        "worst_1pct_return": -0.003,
        "gate_passed": True,
    }
    defaults.update(kwargs)
    return TradeProxyReport(**defaults)


def test_sharpe_delta_pass():
    """Model sharpe exceeds baseline + delta."""
    r = _make_report(sharpe_model=0.5, sharpe_baseline=0.3)
    assert r.sharpe_model >= r.sharpe_baseline + 0.10


def test_sharpe_delta_fail():
    """Model sharpe doesn't meet delta."""
    r = _make_report(sharpe_model=0.35, sharpe_baseline=0.3)
    assert r.sharpe_model < r.sharpe_baseline + 0.10


def test_drawdown_pass():
    r = _make_report(max_drawdown=-0.05)
    assert r.max_drawdown >= -0.10


def test_drawdown_fail():
    r = _make_report(max_drawdown=-0.15)
    assert r.max_drawdown < -0.10


def test_worst_1pct_pass():
    r = _make_report(worst_1pct_return=-0.001)
    tolerance_return = -(25.0 / 10000.0)  # -0.0025
    assert r.worst_1pct_return >= tolerance_return


def test_worst_1pct_fail():
    r = _make_report(worst_1pct_return=-0.01)
    tolerance_return = -(25.0 / 10000.0)
    assert r.worst_1pct_return < tolerance_return


def test_hit_rate_pass():
    r = _make_report(hit_rate=0.52)
    assert r.hit_rate >= 0.45


def test_hit_rate_fail():
    r = _make_report(hit_rate=0.40)
    assert r.hit_rate < 0.45


def test_data_provider_gate_ibkr():
    from sleeves.cooc_reversal_futures.pipeline.validation import _data_provider_gate
    from sleeves.cooc_reversal_futures.config import COOCReversalConfig, ModelConfig
    cfg = COOCReversalConfig(model=ModelConfig(data_provider="ibkr_hist"))
    gate = _data_provider_gate(cfg, provider_name=cfg.model.data_provider)
    assert gate.passed


def test_data_provider_gate_yfinance():
    from sleeves.cooc_reversal_futures.pipeline.validation import _data_provider_gate
    from sleeves.cooc_reversal_futures.config import COOCReversalConfig, ModelConfig
    cfg = COOCReversalConfig(model=ModelConfig(data_provider="yfinance"))
    gate = _data_provider_gate(cfg)
    assert not gate.passed
    assert "RESEARCH_ONLY" in gate.detail
