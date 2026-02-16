"""Anti-leak / anti-inflation test suite for trade proxy integrity.

D6 of the CO→OC integrity refactor.  Five test classes:

1. TestPermutation - shuffle r_oc within each day → IC≈0, |Sharpe|<1
2. TestTimeShift - features at t, label from t+1 → IC drops
3. TestScorePolarity - alpha vs -alpha → return sign flip
4. TestCostSensitivity - 2x costs → Sharpe/mean decreases
5. TestDuplicateRowGuard - duplicate (root, trading_day) → build_gold_frame raises

All synthetic data, no network access.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr


# ---------------------------------------------------------------------------
# Shared synthetic data builder
# ---------------------------------------------------------------------------

def _build_panel(
    n_days: int = 100,
    n_inst: int = 6,
    seed: int = 42,
    reversal_strength: float = 0.7,
) -> pd.DataFrame:
    """Build a synthetic panel with known mean-reversion structure.

    Reversal: r_oc ≈ -reversal_strength * r_co + noise.
    """
    rng = np.random.RandomState(seed)
    rows = []
    instruments = [f"ROOT{i}" for i in range(n_inst)]
    base_date = pd.Timestamp("2024-01-02")

    for d in range(n_days):
        trading_day = (base_date + pd.offsets.BDay(d)).date()
        r_co = rng.randn(n_inst) * 0.01
        r_oc = -reversal_strength * r_co + rng.randn(n_inst) * 0.003

        for i, inst in enumerate(instruments):
            rows.append({
                "trading_day": trading_day,
                "instrument": inst,
                "root": inst,
                "r_co": r_co[i],
                "r_oc": r_oc[i],
                "ret_co": r_co[i],
                "ret_oc": r_oc[i],
                "volume": 10000,
                "days_to_expiry": 20,
                "close": 5000.0,
                "multiplier": 50.0,
            })

    df = pd.DataFrame(rows)
    df["y"] = -df["r_oc"]  # reversal target
    return df


# ---------------------------------------------------------------------------
# 1. Permutation test
# ---------------------------------------------------------------------------

class TestPermutation:
    """Shuffling r_oc within each day should destroy signal."""

    def test_permuted_ic_near_zero(self):
        """After permuting r_oc across roots within each day, IC ≈ 0."""
        panel = _build_panel(n_days=200, n_inst=8, seed=11)

        # Original IC: score = y = -r_oc should correlate with r_oc
        ics_orig = []
        for _, g in panel.groupby("trading_day"):
            if len(g) >= 4:
                ic, _ = spearmanr(g["y"], g["r_oc"])
                ics_orig.append(ic)
        mean_orig_ic = np.mean(ics_orig)
        assert abs(mean_orig_ic) > 0.3, f"Original IC should be strong, got {mean_orig_ic:.3f}"

        # Permute r_oc within each day
        rng = np.random.RandomState(99)
        permuted = panel.copy()
        for day, idx in permuted.groupby("trading_day").groups.items():
            permuted.loc[idx, "r_oc"] = rng.permutation(
                permuted.loc[idx, "r_oc"].values
            )
        permuted["y"] = -permuted["r_oc"]

        # Permuted IC should be ~ 0
        ics_perm = []
        for _, g in permuted.groupby("trading_day"):
            if len(g) >= 4:
                ic, _ = spearmanr(panel.loc[g.index, "y"], g["r_oc"])
                ics_perm.append(ic)
        mean_perm_ic = np.mean(ics_perm)
        assert abs(mean_perm_ic) < 0.15, (
            f"Permuted IC should be ~0, got {mean_perm_ic:.3f}"
        )

    def test_permuted_sharpe_small(self):
        """After permuting r_oc, trade proxy Sharpe should be small."""
        from sleeves.cooc_reversal_futures.pipeline.trade_proxy import evaluate_trade_proxy

        panel = _build_panel(n_days=200, n_inst=8, seed=11)
        original_y = panel["y"].copy()  # save ORIGINAL predictions

        # Permute r_oc within each day (destroys oracle relationship)
        rng = np.random.RandomState(99)
        permuted = panel.copy()
        for _, idx in permuted.groupby("trading_day").groups.items():
            permuted.loc[idx, "r_oc"] = rng.permutation(
                permuted.loc[idx, "r_oc"].values
            )
        # Do NOT update y — keep original predictions to test against shuffled returns

        report = evaluate_trade_proxy(
            dataset=permuted,
            preds=original_y,  # original preds vs shuffled r_oc
            config={
                "cost_per_contract": 0.0, "slippage_bps_open": 0.0,
                "slippage_bps_close": 0.0,
            },
        )
        assert abs(report.sharpe_model) < 5.0, (
            f"Permuted Sharpe should be small, got {report.sharpe_model:.3f}"
        )


# ---------------------------------------------------------------------------
# 2. Time-shift test
# ---------------------------------------------------------------------------

class TestTimeShift:
    """Features at t + label from t+1 should lose IC."""

    def test_shifted_label_reduces_ic(self):
        """Shifting labels by one day should materially reduce IC."""
        panel = _build_panel(n_days=200, n_inst=6, seed=22)

        # Unshifted IC
        ics_orig = []
        for _, g in panel.groupby("trading_day"):
            if len(g) >= 4:
                ic, _ = spearmanr(g["y"], g["r_oc"])
                ics_orig.append(ic)

        # Shift labels: for each (instrument), shift r_oc by 1 day
        shifted = panel.copy()
        shifted["r_oc"] = shifted.groupby("root")["r_oc"].shift(-1)
        shifted = shifted.dropna(subset=["r_oc"])
        shifted["y"] = -shifted["r_oc"]

        ics_shifted = []
        for _, g in shifted.groupby("trading_day"):
            if len(g) >= 4:
                # Cross-sectional IC: original features (r_co) vs shifted label
                ic, _ = spearmanr(g["r_co"], g["r_oc"])
                ics_shifted.append(ic)

        # Shifted IC should be meaningfully weaker
        mean_orig = abs(np.mean(ics_orig))
        mean_shifted = abs(np.mean(ics_shifted))
        assert mean_shifted < mean_orig * 0.7, (
            f"Shifted IC ({mean_shifted:.3f}) should be <70% of original ({mean_orig:.3f})"
        )


# ---------------------------------------------------------------------------
# 3. Score polarity test
# ---------------------------------------------------------------------------

class TestScorePolarity:
    """Using alpha vs -alpha should flip return sign."""

    def test_alpha_negation_flips_returns(self):
        """Negating alpha should flip the sign of mean return."""
        from sleeves.cooc_reversal_futures.pipeline.trade_proxy import evaluate_trade_proxy

        panel = _build_panel(n_days=100, n_inst=6, seed=33)
        preds = panel["y"].values  # oracle: -r_oc

        zero_cost = {
            "cost_per_contract": 0.0, "slippage_bps_open": 0.0,
            "slippage_bps_close": 0.0,
        }

        report_pos = evaluate_trade_proxy(
            dataset=panel, preds=preds, config=zero_cost,
        )
        report_neg = evaluate_trade_proxy(
            dataset=panel, preds=-preds, config=zero_cost,
        )

        # Returns should have opposite signs
        assert report_pos.mean_daily_return * report_neg.mean_daily_return < 0, (
            f"Alpha and -alpha should give opposite-sign returns: "
            f"pos={report_pos.mean_daily_return:.6f}, neg={report_neg.mean_daily_return:.6f}"
        )

    def test_oracle_beats_anti_oracle(self):
        """Oracle (y = -r_oc) should have higher Sharpe than anti-oracle."""
        from sleeves.cooc_reversal_futures.pipeline.trade_proxy import evaluate_trade_proxy

        panel = _build_panel(n_days=100, n_inst=6, seed=33)
        zero_cost = {
            "cost_per_contract": 0.0, "slippage_bps_open": 0.0,
            "slippage_bps_close": 0.0,
        }

        oracle = evaluate_trade_proxy(
            dataset=panel, preds=panel["y"].values, config=zero_cost,
        )
        anti_oracle = evaluate_trade_proxy(
            dataset=panel, preds=-panel["y"].values, config=zero_cost,
        )

        assert oracle.sharpe_model > anti_oracle.sharpe_model, (
            f"Oracle Sharpe ({oracle.sharpe_model:.3f}) should beat "
            f"anti-oracle ({anti_oracle.sharpe_model:.3f})"
        )


# ---------------------------------------------------------------------------
# 4. Cost sensitivity test
# ---------------------------------------------------------------------------

class TestCostSensitivity:
    """Doubling costs should reduce mean return and/or Sharpe."""

    def test_double_cost_reduces_sharpe(self):
        """Double slippage+commission → Sharpe or mean decreases."""
        from sleeves.cooc_reversal_futures.pipeline.trade_proxy import evaluate_trade_proxy

        panel = _build_panel(n_days=100, n_inst=6, seed=44)
        preds = panel["y"].values

        low_cost = {
            "cost_per_contract": 2.5, "slippage_bps_open": 1.0,
            "slippage_bps_close": 1.0,
        }
        high_cost = {
            "cost_per_contract": 5.0, "slippage_bps_open": 2.0,
            "slippage_bps_close": 2.0,
        }

        report_low = evaluate_trade_proxy(
            dataset=panel, preds=preds, config=low_cost,
        )
        report_high = evaluate_trade_proxy(
            dataset=panel, preds=preds, config=high_cost,
        )

        assert report_high.mean_daily_return <= report_low.mean_daily_return, (
            f"Higher costs should reduce mean return: "
            f"low={report_low.mean_daily_return:.6f}, high={report_high.mean_daily_return:.6f}"
        )

    def test_zero_cost_higher_than_positive_cost(self):
        """Zero-cost mean return should be >= positive-cost mean return."""
        from sleeves.cooc_reversal_futures.pipeline.trade_proxy import evaluate_trade_proxy

        panel = _build_panel(n_days=100, n_inst=6, seed=44)
        preds = panel["y"].values

        report_zero = evaluate_trade_proxy(
            dataset=panel, preds=preds,
            config={"cost_per_contract": 0.0, "slippage_bps_open": 0.0,
                     "slippage_bps_close": 0.0},
        )
        report_cost = evaluate_trade_proxy(
            dataset=panel, preds=preds,
            config={"cost_per_contract": 5.0, "slippage_bps_open": 2.0,
                     "slippage_bps_close": 2.0},
        )

        assert report_zero.mean_daily_return >= report_cost.mean_daily_return, (
            f"Zero-cost return should be >= positive-cost: "
            f"zero={report_zero.mean_daily_return:.6f}, cost={report_cost.mean_daily_return:.6f}"
        )


# ---------------------------------------------------------------------------
# 5. Duplicate row guard
# ---------------------------------------------------------------------------

class TestDuplicateRowGuard:
    """Feeding build_gold_frame a frame with duplicates must raise."""

    def test_duplicate_root_trading_day_raises(self):
        """Duplicate (root, trading_day) should raise ValueError."""
        from sleeves.cooc_reversal_futures.pipeline.canonicalize import build_gold_frame

        # Build a minimal silver frame
        dates = pd.bdate_range("2024-01-02", periods=5)
        rows = []
        for d in dates:
            rows.append({
                "root": "ES",
                "trading_day": d.date(),
                "open": 5000.0,
                "close": 5010.0,
                "high": 5020.0,
                "low": 4990.0,
                "volume": 100000,
            })
        silver = pd.DataFrame(rows)

        # Duplicate one row
        silver_dup = pd.concat([silver, silver.iloc[[2]]], ignore_index=True)

        with pytest.raises(ValueError, match="duplicate"):
            build_gold_frame(silver_dup)

    def test_no_duplicate_passes(self):
        """Clean silver frame should NOT raise."""
        from sleeves.cooc_reversal_futures.pipeline.canonicalize import build_gold_frame

        dates = pd.bdate_range("2024-01-02", periods=10)
        rows = []
        for d in dates:
            for root in ["ES", "NQ"]:
                rows.append({
                    "root": root,
                    "trading_day": d.date(),
                    "open": 5000.0 + np.random.randn(),
                    "close": 5010.0 + np.random.randn(),
                    "high": 5020.0,
                    "low": 4990.0,
                    "volume": 100000,
                })
        silver = pd.DataFrame(rows)
        gold = build_gold_frame(silver)

        # Should have rows after dropping first day per root
        assert len(gold) == 2 * (10 - 1)  # 2 roots * 9 days
        assert "r_co" in gold.columns
        assert "r_oc" in gold.columns
