"""Tests for R2: Tier2 realism calibration — per-root ADV, cost decomposition,
and sensitivity sweep."""
import numpy as np
import pandas as pd
import pytest

from sleeves.cooc_reversal_futures.pipeline.types import Tier2ImpactConfig
from sleeves.cooc_reversal_futures.pipeline.trade_proxy import (
    compute_tier2_cost_decomposition,
    compute_tier2_impact_bps,
    evaluate_realism_ladder,
    evaluate_tier2_sensitivity,
)


# ---------------------------------------------------------------------------
# Shared fixture — same _build_panel as test_realism_ladder.py
# ---------------------------------------------------------------------------

def _build_panel(n_days=50, n_inst=6, seed=42):
    rng = np.random.RandomState(seed)
    base = pd.Timestamp("2024-01-02")
    rows = []
    for d in range(n_days):
        td = (base + pd.offsets.BDay(d)).date()
        r_co = rng.randn(n_inst) * 0.01
        r_oc = -0.7 * r_co + rng.randn(n_inst) * 0.003
        for i in range(n_inst):
            rows.append({
                "trading_day": td, "root": f"R{i}", "instrument": f"R{i}",
                "r_co": r_co[i], "r_oc": r_oc[i],
            })
    panel = pd.DataFrame(rows)
    panel["y"] = -panel["r_oc"]
    return panel


# ---------------------------------------------------------------------------
# Tests: per-root ADV in evaluate_realism_ladder
# ---------------------------------------------------------------------------

class TestPerRootADV:
    def test_adv_by_root_propagates(self):
        """Tier2 should use per-root ADV when provided."""
        panel = _build_panel()
        preds = panel["y"].values

        # Without per-root ADV (legacy fallback = 200)
        res_default = evaluate_realism_ladder(panel, preds)

        # With per-root ADV (high ADV = lower impact)
        adv = {f"R{i}": 5000.0 for i in range(6)}
        res_high_adv = evaluate_realism_ladder(
            panel, preds, adv_by_root=adv,
        )

        # High ADV → lower impact → higher Tier2 Sharpe
        t2_default = res_default["TIER2_SPREAD_IMPACT"]["sharpe"]
        t2_high = res_high_adv["TIER2_SPREAD_IMPACT"]["sharpe"]
        assert t2_high >= t2_default - 0.01, (
            f"High ADV should give better Tier2 Sharpe: {t2_high:.3f} vs {t2_default:.3f}"
        )

    def test_low_adv_reduces_sharpe(self):
        """Low ADV should reduce Tier2 Sharpe due to higher impact."""
        panel = _build_panel()
        preds = panel["y"].values

        high_impact_cfg = Tier2ImpactConfig(base_bps=3.0, k=1.0, p=0.5)

        adv_high = {f"R{i}": 10000.0 for i in range(6)}
        adv_low = {f"R{i}": 10.0 for i in range(6)}

        res_high = evaluate_realism_ladder(
            panel, preds, tier2_cfg=high_impact_cfg, adv_by_root=adv_high,
        )
        res_low = evaluate_realism_ladder(
            panel, preds, tier2_cfg=high_impact_cfg, adv_by_root=adv_low,
        )

        assert res_high["TIER2_SPREAD_IMPACT"]["sharpe"] >= res_low["TIER2_SPREAD_IMPACT"]["sharpe"] - 0.05

    def test_adv_used_reported(self):
        """Tier2 result should report the ADV used."""
        panel = _build_panel()
        preds = panel["y"].values
        adv = {f"R{i}": 500.0 for i in range(6)}
        res = evaluate_realism_ladder(panel, preds, adv_by_root=adv)
        assert "adv_used" in res["TIER2_SPREAD_IMPACT"]
        assert res["TIER2_SPREAD_IMPACT"]["adv_used"] == pytest.approx(500.0)


# ---------------------------------------------------------------------------
# Tests: cost decomposition
# ---------------------------------------------------------------------------

class TestCostDecomposition:
    def test_tier0_zero_cost(self):
        panel = _build_panel(n_days=20)
        preds = panel["y"].values
        decomp = compute_tier2_cost_decomposition(panel, preds)
        t0 = decomp["TIER0_ZERO_COST"]
        assert t0["commission_bps_mean"] == 0.0
        assert t0["slippage_bps_mean"] == 0.0
        assert t0["impact_bps_mean"] == 0.0
        assert t0["total_cost_bps_mean"] == 0.0

    def test_tier1_has_slippage(self):
        panel = _build_panel(n_days=20)
        preds = panel["y"].values
        decomp = compute_tier2_cost_decomposition(panel, preds)
        t1 = decomp["TIER1_SIMPLE_COST"]
        assert t1["slippage_bps_mean"] > 0

    def test_tier2_has_impact(self):
        panel = _build_panel(n_days=20)
        preds = panel["y"].values
        adv = {f"R{i}": 200.0 for i in range(6)}
        decomp = compute_tier2_cost_decomposition(
            panel, preds, adv_by_root=adv,
        )
        t2 = decomp["TIER2_SPREAD_IMPACT"]
        assert t2["impact_bps_mean"] > 0
        assert t2["total_cost_bps_mean"] > t2["slippage_bps_mean"]

    def test_contracts_adv_distribution(self):
        panel = _build_panel(n_days=20)
        preds = panel["y"].values
        adv = {f"R{i}": 1000.0 for i in range(6)}
        decomp = compute_tier2_cost_decomposition(
            panel, preds, adv_by_root=adv,
        )
        t2 = decomp["TIER2_SPREAD_IMPACT"]
        assert "contracts_adv_distribution" in t2
        assert len(t2["contracts_adv_distribution"]) == 6

    def test_downscale_trigger_rate(self):
        """With very low cap, everything should trigger."""
        panel = _build_panel(n_days=20)
        preds = panel["y"].values
        adv = {f"R{i}": 200.0 for i in range(6)}
        low_cap = Tier2ImpactConfig(impact_cap_bps=0.0001)  # impossibly low cap
        decomp = compute_tier2_cost_decomposition(
            panel, preds, tier2_cfg=low_cap, adv_by_root=adv,
        )
        t2 = decomp["TIER2_SPREAD_IMPACT"]
        # All root-days should hit cap
        assert t2["downscale_trigger_rate"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Tests: sensitivity sweep
# ---------------------------------------------------------------------------

class TestSensitivitySweep:
    def test_returns_expected_keys(self):
        panel = _build_panel(n_days=20)
        preds = panel["y"].values
        results = evaluate_tier2_sensitivity(
            panel, preds,
            slippage_steps=(1.0, 5.0),
            impact_k_steps=(0.05, 0.1),
        )
        assert "slip_1.0bps" in results
        assert "slip_5.0bps" in results
        assert "impact_k_0.05" in results
        assert "impact_k_0.10" in results

    def test_higher_slip_lower_sharpe(self):
        panel = _build_panel()
        preds = panel["y"].values
        results = evaluate_tier2_sensitivity(
            panel, preds,
            slippage_steps=(1.0, 10.0),
            impact_k_steps=(),
        )
        assert results["slip_1.0bps"]["sharpe"] >= results["slip_10.0bps"]["sharpe"] - 0.05

    def test_each_step_has_metrics(self):
        panel = _build_panel(n_days=20)
        preds = panel["y"].values
        results = evaluate_tier2_sensitivity(
            panel, preds,
            slippage_steps=(2.0,),
            impact_k_steps=(0.1,),
        )
        for label, metrics in results.items():
            assert "sharpe" in metrics, f"Missing sharpe in {label}"
            assert "hit_rate" in metrics, f"Missing hit_rate in {label}"
            assert "max_drawdown" in metrics, f"Missing max_drawdown in {label}"
