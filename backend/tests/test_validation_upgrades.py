"""Tests for F5/G2 validation upgrades — splits, gates, multi-seed stability."""
import numpy as np
import pandas as pd
import pytest
from datetime import date


class TestContiguousOOSSplit:
    def test_split_produces_contiguous_oos(self):
        from sleeves.cooc_reversal_futures.pipeline.splits import contiguous_oos_split

        # Build panel spanning 18 months
        rows = []
        base = pd.Timestamp("2023-01-02")
        for d in range(380):  # ~18 months
            td = (base + pd.offsets.BDay(d)).date()
            for i in range(4):
                rows.append({"trading_day": td, "root": f"R{i}", "r_oc": 0.01})
        panel = pd.DataFrame(rows)

        split = contiguous_oos_split(panel, oos_months=6, embargo_days=2)
        oos_start = date.fromisoformat(split.val_start)
        oos_end = date.fromisoformat(split.val_end)
        train_end = date.fromisoformat(split.train_end)

        # OOS should span roughly 6 months
        oos_span = (oos_end - oos_start).days
        assert oos_span >= 100, f"OOS too short: {oos_span} days"

        # Embargo: train_end < oos_start
        assert train_end < oos_start, "Train end should be before OOS start"

    def test_insufficient_data_raises(self):
        from sleeves.cooc_reversal_futures.pipeline.splits import contiguous_oos_split

        rows = [{"trading_day": date(2024, 1, d + 1), "root": "R0", "r_oc": 0.0}
                for d in range(5)]
        panel = pd.DataFrame(rows)

        with pytest.raises(ValueError, match="Insufficient"):
            contiguous_oos_split(panel, oos_months=12)


class TestICDistributionGate:
    def test_passes_with_good_ics(self):
        from sleeves.cooc_reversal_futures.pipeline.validation import _ic_distribution_gate
        ics = np.array([0.1, 0.2, 0.15, 0.05, 0.12, 0.08, 0.18, 0.11, 0.09, 0.14])
        result = _ic_distribution_gate(ics, ic_tail_floor=-0.05)
        assert result.passed

    def test_fails_with_tail_blow(self):
        from sleeves.cooc_reversal_futures.pipeline.validation import _ic_distribution_gate
        ics = np.array([0.1, 0.2, -0.3, -0.2, -0.15, 0.05, 0.08, -0.1, -0.25, 0.01])
        result = _ic_distribution_gate(ics, ic_tail_floor=-0.05)
        assert not result.passed


class TestMultiSeedStability:
    def test_passes_stable_seeds(self):
        from sleeves.cooc_reversal_futures.pipeline.validation import _multi_seed_stability_gate
        sharpes = {11: 1.0, 22: 1.1, 33: 0.95, 44: 1.05, 55: 1.02}
        result = _multi_seed_stability_gate(sharpes, max_cv=0.30)
        assert result.passed

    def test_fails_unstable_seeds(self):
        from sleeves.cooc_reversal_futures.pipeline.validation import _multi_seed_stability_gate
        sharpes = {11: 2.0, 22: 0.1, 33: -0.5, 44: 1.5}
        result = _multi_seed_stability_gate(sharpes, max_cv=0.30)
        assert not result.passed

    def test_single_seed_skipped(self):
        from sleeves.cooc_reversal_futures.pipeline.validation import _multi_seed_stability_gate
        result = _multi_seed_stability_gate({42: 1.0}, max_cv=0.30)
        assert result.passed

    # --- G2: baseline-relative delta tests ---

    def test_catastrophic_delta_floor_fails(self):
        """If worst seed delta < floor, gate should fail."""
        from sleeves.cooc_reversal_futures.pipeline.validation import _multi_seed_stability_gate
        model_sharpes = {1: 0.5, 2: 0.8, 3: 0.6, 4: 0.2}
        baseline_sharpes = {1: 0.7, 2: 0.6, 3: 0.8, 4: 0.9}
        # Deltas: -0.2, +0.2, -0.2, -0.7 → min=-0.7
        result = _multi_seed_stability_gate(
            model_sharpes, max_cv=1.0,  # relax raw CV
            baseline_sharpes_by_seed=baseline_sharpes,
            catastrophic_delta_floor=-0.25,
        )
        assert not result.passed
        assert "min_delta" in result.detail

    def test_catastrophic_delta_floor_passes(self):
        """If all seed deltas >= floor, this check passes."""
        from sleeves.cooc_reversal_futures.pipeline.validation import _multi_seed_stability_gate
        model_sharpes = {1: 0.8, 2: 0.9, 3: 0.7, 4: 0.85}
        baseline_sharpes = {1: 0.7, 2: 0.6, 3: 0.6, 4: 0.7}
        # Deltas: +0.1, +0.3, +0.1, +0.15 → min=+0.1
        result = _multi_seed_stability_gate(
            model_sharpes, max_cv=1.0,
            baseline_sharpes_by_seed=baseline_sharpes,
            catastrophic_delta_floor=-0.25,
            max_delta_cv=1.0,  # relax: only testing floor here
        )
        assert result.passed

    def test_delta_cv_fails_high_dispersion(self):
        """If delta CV > threshold, gate should fail."""
        from sleeves.cooc_reversal_futures.pipeline.validation import _multi_seed_stability_gate
        model_sharpes = {1: 1.5, 2: 0.5, 3: 1.5, 4: 0.5}
        baseline_sharpes = {1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5}
        # Deltas: 1.0, 0.0, 1.0, 0.0 → high CV
        result = _multi_seed_stability_gate(
            model_sharpes, max_cv=1.0,
            baseline_sharpes_by_seed=baseline_sharpes,
            catastrophic_delta_floor=-1.0,
            max_delta_cv=0.30,
        )
        assert not result.passed
        assert "delta_cv" in result.detail


class TestStressWindowGate:
    def test_passes_no_catastrophe(self):
        from sleeves.cooc_reversal_futures.pipeline.validation import _stress_window_gate
        results = [
            {"name": "covid", "sharpe_model": 0.5, "max_drawdown": -0.05, "max_drawdown_baseline": -0.08},
            {"name": "taper", "sharpe_model": 0.3, "max_drawdown": -0.03, "max_drawdown_baseline": -0.05},
        ]
        gate = _stress_window_gate(results)
        assert gate.passed

    def test_fails_catastrophic_sharpe(self):
        from sleeves.cooc_reversal_futures.pipeline.validation import _stress_window_gate
        results = [
            {"name": "crisis", "sharpe_model": -1.0, "max_drawdown": -0.30, "max_drawdown_baseline": -0.15},
        ]
        gate = _stress_window_gate(results, catastrophic_sharpe_floor=-0.50)
        assert not gate.passed


class TestContiguousOOSGate:
    def test_passes_small_degradation(self):
        from sleeves.cooc_reversal_futures.pipeline.validation import _contiguous_oos_gate
        gate = _contiguous_oos_gate(oos_sharpe=0.8, cv_mean_sharpe=1.0, max_degradation=0.50)
        assert gate.passed

    def test_fails_large_degradation(self):
        from sleeves.cooc_reversal_futures.pipeline.validation import _contiguous_oos_gate
        gate = _contiguous_oos_gate(oos_sharpe=0.2, cv_mean_sharpe=1.0, max_degradation=0.50)
        assert not gate.passed

    def test_handles_zero_cv(self):
        from sleeves.cooc_reversal_futures.pipeline.validation import _contiguous_oos_gate
        gate = _contiguous_oos_gate(oos_sharpe=0.1, cv_mean_sharpe=0.0)
        assert gate.passed  # oos >= 0

    # --- G2: absolute OOS delta tests ---

    def test_absolute_oos_delta_fails(self):
        """If OOS Sharpe delta vs baseline < floor, gate should fail."""
        from sleeves.cooc_reversal_futures.pipeline.validation import _contiguous_oos_gate
        gate = _contiguous_oos_gate(
            oos_sharpe=0.5, cv_mean_sharpe=0.6,
            baseline_oos_sharpe=0.55,
            sharpe_delta_min_oos=0.05,
        )
        # delta = 0.5 - 0.55 = -0.05 < 0.05
        assert not gate.passed
        assert "oos_delta" in gate.detail

    def test_absolute_oos_delta_passes(self):
        """If OOS Sharpe delta vs baseline >= floor, this check passes."""
        from sleeves.cooc_reversal_futures.pipeline.validation import _contiguous_oos_gate
        gate = _contiguous_oos_gate(
            oos_sharpe=0.7, cv_mean_sharpe=0.8,
            baseline_oos_sharpe=0.55,
            sharpe_delta_min_oos=0.05,
        )
        # delta = 0.7 - 0.55 = 0.15 >= 0.05
        assert gate.passed

    def test_worst_1pct_tolerance_fails(self):
        """If worst 1% day gap exceeds tolerance, gate should fail."""
        from sleeves.cooc_reversal_futures.pipeline.validation import _contiguous_oos_gate
        gate = _contiguous_oos_gate(
            oos_sharpe=0.8, cv_mean_sharpe=0.8,
            worst_1pct_oos=-0.10,
            worst_1pct_baseline=-0.05,
            worst_1pct_tolerance=0.02,
        )
        # gap = -0.10 - (-0.05) = -0.05 < -0.02
        assert not gate.passed
        assert "worst_1pct" in gate.detail

    def test_worst_1pct_tolerance_passes(self):
        """If worst 1% day gap within tolerance, passes."""
        from sleeves.cooc_reversal_futures.pipeline.validation import _contiguous_oos_gate
        gate = _contiguous_oos_gate(
            oos_sharpe=0.8, cv_mean_sharpe=0.8,
            worst_1pct_oos=-0.06,
            worst_1pct_baseline=-0.05,
            worst_1pct_tolerance=0.02,
        )
        # gap = -0.06 - (-0.05) = -0.01 >= -0.02
        assert gate.passed
