"""Tests for F3 realism ladder — tier monotonicity and impact model."""
import numpy as np
import pandas as pd
import pytest

from sleeves.cooc_reversal_futures.pipeline.types import (
    RealismTier, Tier2ImpactConfig,
)
from sleeves.cooc_reversal_futures.pipeline.trade_proxy import (
    compute_tier2_impact_bps, evaluate_realism_ladder,
)


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


class TestComputeTier2Impact:
    def test_higher_participation_higher_impact(self):
        cfg = Tier2ImpactConfig(base_bps=0.5, k=0.1, p=0.5)
        low = compute_tier2_impact_bps(1.0, 1000.0, impact_cfg=cfg)
        high = compute_tier2_impact_bps(500.0, 1000.0, impact_cfg=cfg)
        assert high > low, f"Higher participation should cost more: {high} > {low}"

    def test_cap_respected(self):
        cfg = Tier2ImpactConfig(base_bps=0.5, k=100.0, p=1.0, impact_cap_bps=5.0)
        impact = compute_tier2_impact_bps(10000.0, 100.0, impact_cfg=cfg)
        assert impact <= 5.0, f"Impact should be capped at 5.0, got {impact}"

    def test_default_impact_positive(self):
        impact = compute_tier2_impact_bps(1.0, 200.0)
        assert impact > 0


class TestRealismTier:
    def test_enum_ordering(self):
        assert RealismTier.TIER0_ZERO_COST < RealismTier.TIER1_SIMPLE_COST
        assert RealismTier.TIER1_SIMPLE_COST < RealismTier.TIER2_SPREAD_IMPACT


class TestEvaluateRealismLadder:
    def test_cost_monotonicity(self):
        """Higher realism should produce equal or lower Sharpe (more costs).

        Uses high-impact Tier2 config to ensure Tier2 costs exceed Tier1.
        """
        panel = _build_panel()
        # Oracle predictions
        preds = panel["y"].values

        # Use high-impact config so Tier2 > Tier1 cost
        high_impact = Tier2ImpactConfig(base_bps=3.0, k=1.0, p=0.5)
        results = evaluate_realism_ladder(panel, preds, tier2_cfg=high_impact)

        t0 = results["TIER0_ZERO_COST"]["sharpe"]
        t1 = results["TIER1_SIMPLE_COST"]["sharpe"]
        t2 = results["TIER2_SPREAD_IMPACT"]["sharpe"]

        # Tier0 (zero cost) should have highest or equal Sharpe
        assert t0 >= t1 - 0.1, f"Tier0 ({t0:.3f}) should >= Tier1 ({t1:.3f})"
        # Tier2 (higher impact) should have lower or equal Sharpe
        assert t1 >= t2 - 0.1, f"Tier1 ({t1:.3f}) should >= Tier2 ({t2:.3f})"

    def test_summary_keys(self):
        panel = _build_panel(n_days=20)
        preds = panel["y"].values
        results = evaluate_realism_ladder(panel, preds)

        assert "summary" in results
        summary = results["summary"]
        assert "tier0_sharpe" in summary
        assert "tier1_sharpe" in summary
        assert "tier2_sharpe" in summary
        assert "cost_erosion_t0_t1" in summary
        assert "cost_erosion_t1_t2" in summary

    def test_impact_parameter_sensitivity(self):
        """Doubling impact k should reduce Tier2 Sharpe."""
        panel = _build_panel()
        preds = panel["y"].values

        cfg_low = Tier2ImpactConfig(k=0.01)
        cfg_high = Tier2ImpactConfig(k=1.0)

        res_low = evaluate_realism_ladder(panel, preds, tier2_cfg=cfg_low)
        res_high = evaluate_realism_ladder(panel, preds, tier2_cfg=cfg_high)

        sharpe_low = res_low["TIER2_SPREAD_IMPACT"]["sharpe"]
        sharpe_high = res_high["TIER2_SPREAD_IMPACT"]["sharpe"]

        assert sharpe_low >= sharpe_high - 0.05, (
            f"Lower impact k should give higher Sharpe: {sharpe_low:.3f} vs {sharpe_high:.3f}"
        )
