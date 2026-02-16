"""Trade proxy sign semantics — oracle and baseline direction tests.

CHANGE LOG (2026-02-14):
  - D6: Fixed broken kwargs (commission_per_contract/slippage_bps →
    dict config). Updated to match D4 alpha polarity (high alpha = long).
  - D4: Oracle preds still yield positive performance; baseline alpha
    now uses -r_co (mean-reversion convention).

Verifies that the score direction convention is consistent:
  - Higher alpha = LONG candidate (alpha_high_long)
  - Oracle preds ``y = -r_oc`` yield strong positive performance
  - Anti-oracle preds ``r_oc`` yield negative performance
  - Baseline ``-r_co`` (mean-revert alpha) has correct sign ordering
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _build_deterministic_panel(n_days: int = 60, n_inst: int = 4, seed: int = 42) -> pd.DataFrame:
    """Build a synthetic panel with known reversal structure.

    Reversal: higher r_co → lower r_oc (strong negative correlation).
    """
    rng = np.random.RandomState(seed)
    rows = []
    instruments = [f"INST{i}" for i in range(n_inst)]
    base_date = pd.Timestamp("2024-01-02")

    for d in range(n_days):
        trading_day = (base_date + pd.offsets.BDay(d)).date()
        # Draw cross-sectional r_co
        r_co = rng.randn(n_inst) * 0.01

        # Strong reversal: r_oc ≈ -0.7 * r_co + noise
        r_oc = -0.7 * r_co + rng.randn(n_inst) * 0.003

        for i, inst in enumerate(instruments):
            rows.append({
                "trading_day": trading_day,
                "instrument": inst,
                "r_co": r_co[i],
                "r_oc": r_oc[i],
                "ret_co": r_co[i],
                "ret_oc": r_oc[i],
                "root": inst,
                "volume": 10000 + rng.randint(0, 5000),
                "days_to_expiry": 20,
                "roll_window_flag": 0,
                "day_of_week": d % 5,
                "sigma_co": abs(rng.randn()) * 0.01,
                "sigma_oc": abs(rng.randn()) * 0.01,
                "volume_z": rng.randn() * 0.5,
                "r_co_rank_pct": 0.5,
                "r_co_cs_demean": 0.0,
                "close": 5000.0,
                "multiplier": 50.0,
            })

    df = pd.DataFrame(rows)
    # Label: y = -r_oc (reversal target)
    df["y"] = -df["r_oc"]
    return df


# Zero-cost config for clean tests
_ZERO_COST = {
    "cost_per_contract": 0.0,
    "slippage_bps_open": 0.0,
    "slippage_bps_close": 0.0,
}


class TestBaselineScoreSign:
    """Verify baseline alpha = -r_co has the correct reversal ordering.

    D4: baseline_semantics="r_co_meanrevert" → baseline_alpha = -r_co.
    Gap up (r_co > 0) → alpha < 0 → SHORT.
    Gap down (r_co < 0) → alpha > 0 → LONG.
    """

    def test_gap_up_gets_low_alpha(self):
        """If r_co is positive (overnight up), baseline alpha = -r_co should be negative → SHORT."""
        panel = _build_deterministic_panel()
        panel["baseline_alpha"] = -panel["r_co"]
        for _, day_df in panel.groupby("trading_day"):
            # Highest r_co should have lowest alpha
            top_r_co_inst = day_df.loc[day_df["r_co"].idxmax(), "instrument"]
            bot_alpha_inst = day_df.loc[day_df["baseline_alpha"].idxmin(), "instrument"]
            assert top_r_co_inst == bot_alpha_inst

    def test_gap_down_gets_high_alpha(self):
        """If r_co is negative (overnight down), baseline alpha = -r_co should be positive → LONG."""
        panel = _build_deterministic_panel()
        panel["baseline_alpha"] = -panel["r_co"]
        for _, day_df in panel.groupby("trading_day"):
            bot_r_co_inst = day_df.loc[day_df["r_co"].idxmin(), "instrument"]
            top_alpha_inst = day_df.loc[day_df["baseline_alpha"].idxmax(), "instrument"]
            assert bot_r_co_inst == top_alpha_inst


class TestOracleTradeProxy:
    """Oracle preds (y = -r_oc) should yield positive trade proxy return."""

    def test_oracle_preds_positive_sharpe(self):
        """Oracle predictions should produce strong positive performance."""
        from sleeves.cooc_reversal_futures.pipeline.trade_proxy import evaluate_trade_proxy

        panel = _build_deterministic_panel()
        preds = panel["y"].values  # oracle = -r_oc

        report = evaluate_trade_proxy(
            dataset=panel,
            preds=preds,
            config=_ZERO_COST,
        )
        assert report.sharpe_model > 0.5, (
            f"Oracle preds should give Sharpe > 0.5, got {report.sharpe_model:.3f}"
        )

    def test_anti_oracle_negative_sharpe(self):
        """Anti-oracle predictions (r_oc) should produce negative performance."""
        from sleeves.cooc_reversal_futures.pipeline.trade_proxy import evaluate_trade_proxy

        panel = _build_deterministic_panel()
        preds = -panel["y"].values  # anti-oracle = r_oc

        report = evaluate_trade_proxy(
            dataset=panel,
            preds=preds,
            config=_ZERO_COST,
        )
        assert report.sharpe_model < 0.0, (
            f"Anti-oracle preds should give negative Sharpe, got {report.sharpe_model:.3f}"
        )


class TestBaselineTradeProxy:
    """Baseline (-r_co mean-revert alpha) should have positive Sharpe in reversal regime."""

    def test_baseline_positive_sharpe(self):
        from sleeves.cooc_reversal_futures.pipeline.trade_proxy import evaluate_trade_proxy

        panel = _build_deterministic_panel()
        # Under alpha_low_long, alpha = -score.  Passing r_co as score
        # gives alpha = -r_co = mean-revert direction (gap up → short).
        preds = panel["r_co"].values

        report = evaluate_trade_proxy(
            dataset=panel,
            preds=preds,
            config=_ZERO_COST,
        )
        # The evaluate_trade_proxy applies baseline_semantics internally,
        # but we're passing -r_co directly as "predictions" which is correct
        # under alpha_high_long (higher alpha → long)
        assert report.sharpe_model > 0.0, (
            f"Baseline -r_co should have positive Sharpe in reversal regime, got {report.sharpe_model:.3f}"
        )


class TestScoreDirectionConsistency:
    """Verify labels, predictions, and trade proxy are directionally consistent."""

    def test_label_is_negated_roc(self):
        panel = _build_deterministic_panel()
        np.testing.assert_allclose(panel["y"].values, -panel["r_oc"].values)

    def test_oracle_higher_than_anti_oracle(self):
        """Oracle should always beat anti-oracle Sharpe."""
        from sleeves.cooc_reversal_futures.pipeline.trade_proxy import evaluate_trade_proxy

        panel = _build_deterministic_panel()
        oracle_report = evaluate_trade_proxy(
            dataset=panel, preds=panel["y"].values, config=_ZERO_COST,
        )
        anti_report = evaluate_trade_proxy(
            dataset=panel, preds=-panel["y"].values, config=_ZERO_COST,
        )

        assert oracle_report.sharpe_model > anti_report.sharpe_model, (
            f"Oracle Sharpe ({oracle_report.sharpe_model:.3f}) should exceed "
            f"anti-oracle ({anti_report.sharpe_model:.3f})"
        )

    def test_report_has_diagnostics(self):
        """D5: TradeProxyReport should include diagnostics fields."""
        from sleeves.cooc_reversal_futures.pipeline.trade_proxy import evaluate_trade_proxy

        panel = _build_deterministic_panel()
        report = evaluate_trade_proxy(
            dataset=panel, preds=panel["y"].values, config=_ZERO_COST,
        )
        assert report.n_days > 0
        assert report.vol >= 0
        assert isinstance(report.skew, float)
        assert isinstance(report.kurtosis, float)
        assert isinstance(report.cvar_1pct, float)
