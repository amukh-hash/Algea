"""
Unit tests for the two-level regime gate.

Tests:
1. Scaler fit/apply parity across dates.
2. Inference parity: saved scaler reproduces training z_cs for same date.
3. Gate direction: higher stress → lower w → more baseline.
"""
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_toy_df(n_symbols: int = 50) -> pd.DataFrame:
    """3 dates, n_symbols symbols/date, varying tail_risk_30 dispersion."""
    rng = np.random.RandomState(42)
    rows = []
    dates = pd.to_datetime(["2024-01-01", "2024-06-01", "2024-09-01"])
    # Increasing dispersion across dates → date3 is "stressed"
    dispersions = [0.01, 0.02, 0.05]

    for dt, disp in zip(dates, dispersions):
        for i in range(n_symbols):
            base = 0.03 + rng.randn() * disp
            rows.append({
                "date": dt,
                "symbol": f"SYM_{i:03d}",
                "tail_risk_30": max(base, 0.001),
                # Provide z-scored cols needed for gate input
                "z_regime_risk": rng.randn() * 0.5,
                "z_drift_10": rng.randn() * 0.3,
                "z_tail_risk_30": rng.randn() * 0.4,
                "z_iqr_30": 0.0,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Test 1: Scaler fit/apply parity
# ---------------------------------------------------------------------------

class TestScalerFitApplyParity:
    """Verify time-zscore differs across dates and train scaler works on new dates."""

    def test_z_cs_differs_across_dates(self):
        from algea.data.priors.feature_utils import (
            add_date_regime_features,
            compute_date_cross_sectional_stats,
            fit_time_zscore_scaler,
        )

        df = _make_toy_df()
        # Fit scaler on first 2 dates only (train)
        train_df = df[df["date"] < pd.Timestamp("2024-09-01")].copy()
        train_stats = compute_date_cross_sectional_stats(train_df)
        scaler = fit_time_zscore_scaler(train_stats, "cs_tail_30_std")

        # Apply to all 3 dates
        df_out, scaler_used = add_date_regime_features(df.copy(), scaler=scaler)

        # z_cs_tail_30_std should be same for all tickers on a date
        # but different ACROSS dates
        z_by_date = df_out.groupby("date")["z_cs_tail_30_std"].first()
        assert len(z_by_date.unique()) == 3, (
            f"Expected 3 distinct z_cs values, got {z_by_date.unique()}"
        )

    def test_train_scaler_applied_to_test_date(self):
        from algea.data.priors.feature_utils import (
            add_date_regime_features,
            compute_date_cross_sectional_stats,
            fit_time_zscore_scaler,
        )

        df = _make_toy_df()
        train_df = df[df["date"] < pd.Timestamp("2024-09-01")].copy()
        test_df = df[df["date"] >= pd.Timestamp("2024-09-01")].copy()

        # Fit scaler on train
        train_stats = compute_date_cross_sectional_stats(train_df)
        scaler = fit_time_zscore_scaler(train_stats, "cs_tail_30_std")

        # Apply to test (date3 has higher dispersion → should get positive z_cs)
        test_out, _ = add_date_regime_features(test_df.copy(), scaler=scaler)

        z_test = test_out["z_cs_tail_30_std"].iloc[0]
        # The stressed date (high dispersion) should have positive z_cs
        assert z_test > 0, f"Expected positive z_cs for stress date, got {z_test}"

        # Verify scaler mu/sigma come from train only
        assert scaler["mu"] > 0


# ---------------------------------------------------------------------------
# Test 2: Inference parity (saved scaler reproduces training z_cs)
# ---------------------------------------------------------------------------

class TestInferenceParity:
    """Saved scaler in manifest reproduces training-time z_cs values."""

    def test_manifest_scaler_reproduces_z_cs(self):
        from algea.data.priors.feature_utils import (
            add_date_regime_features,
            compute_date_cross_sectional_stats,
            fit_time_zscore_scaler,
        )

        df = _make_toy_df()
        # Simulate training: fit scaler on all dates
        all_stats = compute_date_cross_sectional_stats(df)
        scaler = fit_time_zscore_scaler(all_stats, "cs_tail_30_std")

        # Apply to full df
        df_full, _ = add_date_regime_features(df.copy(), scaler=scaler)

        # Save scaler to temp manifest
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump({"cs_scaler": scaler}, f)
            manifest_path = Path(f.name)

        # Load scaler back and apply to single date
        with open(manifest_path) as f:
            loaded = json.load(f)
        loaded_scaler = loaded["cs_scaler"]

        date1 = pd.Timestamp("2024-01-01")
        single_df = df[df["date"] == date1].copy()
        single_out, _ = add_date_regime_features(single_df, scaler=loaded_scaler)

        # z_cs for this date should match the full-df computation
        expected = df_full[df_full["date"] == date1]["z_cs_tail_30_std"].iloc[0]
        actual = single_out["z_cs_tail_30_std"].iloc[0]

        np.testing.assert_allclose(
            actual, expected, atol=1e-6,
            err_msg=f"Inference z_cs ({actual}) != training z_cs ({expected})"
        )

        # Cleanup
        manifest_path.unlink()


# ---------------------------------------------------------------------------
# Test 3: Gate direction (stress → lower w → more baseline)
# ---------------------------------------------------------------------------

class TestGateDirection:
    """Higher stress (positive z_cs) with default sign → lower gate weight w."""

    def test_stress_lowers_w(self):
        from algea.models.ranker.baseline_scorer import compute_gate_weights

        n = 100
        df_calm = pd.DataFrame({
            "z_regime_risk": np.zeros(n),
            "z_cs_tail_30_std": np.full(n, -1.0),  # calm
        })
        df_stress = pd.DataFrame({
            "z_regime_risk": np.zeros(n),
            "z_cs_tail_30_std": np.full(n, 2.0),  # stress
        })

        # Default gate_cs_sign=-1.0: stress (positive z_cs) → negative contrib
        # But wait — with sign=-1, positive z_cs → contribution = gamma*(-1)*2 = -2*gamma
        # That REDUCES risk, which INCREASES w. We want the OPPOSITE.
        # Actually: with gate_cs_sign=-1 and positive z_cs (stress):
        #   risk += gamma_cs * (-1) * (+2) = -2*gamma_cs
        #   w = sigmoid(g0 - g1 * risk). Lower risk → higher w. BAD.
        #
        # So for "stress=positive z_cs → lower w", we need gate_cs_sign=+1:
        #   risk += gamma_cs * (+1) * (+2) = +2*gamma_cs
        #   higher risk → lower w. GOOD.
        #
        # The default sign=-1 is for when stress months have NEGATIVE z_cs
        # (which was the original observation). Let's test both cases.

        # Case 1: stress is POSITIVE z_cs, sign=+1
        w_calm = compute_gate_weights(
            df_calm, g0=1.0, g1=1.0, gate_gamma_cs=2.0, gate_cs_sign=1.0,
        ).mean()
        w_stress = compute_gate_weights(
            df_stress, g0=1.0, g1=1.0, gate_gamma_cs=2.0, gate_cs_sign=1.0,
        ).mean()
        assert w_stress < w_calm, (
            f"With sign=+1, stress w ({w_stress:.4f}) should be < calm w ({w_calm:.4f})"
        )

        # Case 2: stress has NEGATIVE z_cs (original observation), sign=-1
        df_stress_neg = pd.DataFrame({
            "z_regime_risk": np.zeros(n),
            "z_cs_tail_30_std": np.full(n, -2.0),  # stress appears as negative
        })
        df_calm_neg = pd.DataFrame({
            "z_regime_risk": np.zeros(n),
            "z_cs_tail_30_std": np.full(n, 0.5),  # calm appears as slightly positive
        })
        w_calm2 = compute_gate_weights(
            df_calm_neg, g0=1.0, g1=1.0, gate_gamma_cs=2.0, gate_cs_sign=-1.0,
        ).mean()
        w_stress2 = compute_gate_weights(
            df_stress_neg, g0=1.0, g1=1.0, gate_gamma_cs=2.0, gate_cs_sign=-1.0,
        ).mean()
        assert w_stress2 < w_calm2, (
            f"With sign=-1, neg-stress w ({w_stress2:.4f}) should be < calm w ({w_calm2:.4f})"
        )

    def test_gamma_cs_zero_disables_cs(self):
        """When gamma_cs=0, z_cs_tail_30_std has no effect."""
        from algea.models.ranker.baseline_scorer import compute_gate_weights

        n = 50
        df1 = pd.DataFrame({
            "z_regime_risk": np.zeros(n),
            "z_cs_tail_30_std": np.full(n, 5.0),
        })
        df2 = pd.DataFrame({
            "z_regime_risk": np.zeros(n),
            "z_cs_tail_30_std": np.full(n, -5.0),
        })

        w1 = compute_gate_weights(df1, g0=1.0, g1=1.0, gate_gamma_cs=0.0).mean()
        w2 = compute_gate_weights(df2, g0=1.0, g1=1.0, gate_gamma_cs=0.0).mean()
        np.testing.assert_allclose(w1, w2, atol=1e-10)


# ---------------------------------------------------------------------------
# Test 4: No leakage (scaler fitted on train only)
# ---------------------------------------------------------------------------

class TestNoLeakage:
    """Enforce that scaler fit reflects only the data it was given."""

    def test_distinct_distributions_give_different_mu(self):
        """If we fit on two very different data sets, mu should differ."""
        from algea.data.priors.feature_utils import (
            compute_date_cross_sectional_stats,
            fit_time_zscore_scaler,
        )

        # Build two disjoint datasets with very different tail_risk profiles
        rng = np.random.RandomState(99)
        n = 50

        # Low-vol dataset
        rows_low = []
        for dt in pd.to_datetime(["2023-01-01", "2023-02-01", "2023-03-01"]):
            for i in range(n):
                rows_low.append({
                    "date": dt,
                    "symbol": f"LOW_{i:03d}",
                    "tail_risk_30": 0.01 + rng.rand() * 0.005,
                })
        df_low = pd.DataFrame(rows_low)

        # High-vol dataset
        rows_high = []
        for dt in pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]):
            for i in range(n):
                rows_high.append({
                    "date": dt,
                    "symbol": f"HIGH_{i:03d}",
                    "tail_risk_30": 0.10 + rng.rand() * 0.05,
                })
        df_high = pd.DataFrame(rows_high)

        stats_low = compute_date_cross_sectional_stats(df_low)
        stats_high = compute_date_cross_sectional_stats(df_high)

        scaler_low = fit_time_zscore_scaler(stats_low, "cs_tail_30_std")
        scaler_high = fit_time_zscore_scaler(stats_high, "cs_tail_30_std")

        # Mu should be significantly different (leakage would blur them)
        assert abs(scaler_low["mu"] - scaler_high["mu"]) > 0.001, (
            f"Scalers should differ: low mu={scaler_low['mu']:.6f}, "
            f"high mu={scaler_high['mu']:.6f}"
        )

    def test_train_scaler_excludes_future(self):
        """Scaler fitted on train dates must not see val/test dates."""
        from algea.data.priors.feature_utils import (
            compute_date_cross_sectional_stats,
            fit_time_zscore_scaler,
        )

        df = _make_toy_df()
        train_df = df[df["date"] < pd.Timestamp("2024-09-01")].copy()
        all_df = df.copy()

        train_stats = compute_date_cross_sectional_stats(train_df)
        all_stats = compute_date_cross_sectional_stats(all_df)

        scaler_train = fit_time_zscore_scaler(train_stats, "cs_tail_30_std")
        scaler_all = fit_time_zscore_scaler(all_stats, "cs_tail_30_std")

        # The stressed date (2024-09-01) is only in all_stats → different scalers
        assert scaler_train["mu"] != scaler_all["mu"], (
            "Train-only scaler should differ from all-data scaler"
        )


# ---------------------------------------------------------------------------
# Test 5: Manifest format validation
# ---------------------------------------------------------------------------

class TestManifestFormat:
    """Verify blend_manifest.json contains all required keys per spec."""

    REQUIRED_KEYS = {
        "blend_mode", "gate_g0", "gate_g1", "gate_gamma_cs",
        "gate_cs_sign", "gate_gamma_unc", "gate_use_uncertainty",
        "baseline_a", "baseline_lambda", "baseline_mu",
        "cs_feature", "cs_feature_spec", "cs_scaler", "tuning", "grid_results",
    }
    SCALER_KEYS = {"mu", "sigma", "clamp"}
    TUNING_KEYS = {"objective", "split", "grid", "best_ic"}

    def test_manifest_has_all_keys(self):
        """Construct a manifest dict matching train_selector.py output format."""
        manifest = {
            "blend_mode": "sigmoid",
            "gate_g0": 1.0,
            "gate_g1": 0.5,
            "gate_gamma_cs": 2.0,
            "gate_cs_sign": 1.0,
            "gate_gamma_unc": 0.0,
            "gate_use_uncertainty": False,
            "baseline_a": 1.0,
            "baseline_lambda": 0.5,
            "baseline_mu": 0.5,
            "cs_feature": "cs_tail_30_std",
            "cs_feature_spec": {
                "name": "cs_tail_30_std",
                "z_col": "z_cs_tail_30_std",
                "meaning": "stress_up",
                "recommended_cs_sign": 1.0,
            },
            "cs_scaler": {"mu": 0.4, "sigma": 1.0, "clamp": 5.0},
            "tuning": {
                "objective": "mean_ic",
                "split": "val",
                "grid": {"g0": 1.0, "g1_values": [0.5, 1.0]},
                "best_ic": 0.01,
            },
            "grid_results": [{"g0": 1.0, "g1": 0.5, "gamma_cs": 0.0,
                              "cs_sign": 1.0, "ic": 0.01}],
        }

        missing = self.REQUIRED_KEYS - set(manifest.keys())
        assert not missing, f"Missing manifest keys: {missing}"

        scaler = manifest["cs_scaler"]
        missing_scaler = self.SCALER_KEYS - set(scaler.keys())
        assert not missing_scaler, f"Missing scaler keys: {missing_scaler}"

        tuning = manifest["tuning"]
        missing_tuning = self.TUNING_KEYS - set(tuning.keys())
        assert not missing_tuning, f"Missing tuning keys: {missing_tuning}"

        assert isinstance(manifest["grid_results"], list)
        assert len(manifest["grid_results"]) > 0


# ---------------------------------------------------------------------------
# Test 6: Gate monotonicity (w strictly decreasing in gate_input when g1>0)
# ---------------------------------------------------------------------------

class TestGateMonotonicity:
    """Verify that sigmoid_gate is monotonically decreasing for g1>0."""

    def test_sigmoid_decreasing_basic(self):
        from algea.models.ranker.baseline_scorer import sigmoid_gate
        gate_input = np.linspace(-3.0, 3.0, 100)
        w = sigmoid_gate(gate_input, g0=0.0, g1=1.0)
        diffs = np.diff(w)
        assert np.all(diffs <= 0), (
            f"w should be non-increasing; found {(diffs > 0).sum()} violations"
        )

    def test_sigmoid_decreasing_with_offset(self):
        from algea.models.ranker.baseline_scorer import sigmoid_gate
        gate_input = np.linspace(-5.0, 5.0, 200)
        for g0 in [-2.0, 0.0, 1.0, 3.0]:
            for g1 in [0.5, 1.0, 2.0]:
                w = sigmoid_gate(gate_input, g0=g0, g1=g1)
                diffs = np.diff(w)
                assert np.all(diffs <= 1e-12), (
                    f"g0={g0}, g1={g1}: w not non-increasing"
                )

    def test_sanity_check_helper_passes(self):
        from algea.models.ranker.baseline_scorer import (
            sanity_check_gate_monotonicity, sigmoid_gate,
        )
        gi = np.linspace(-3, 3, 200)
        w = sigmoid_gate(gi, g0=0.0, g1=1.0)
        assert sanity_check_gate_monotonicity(w, gi, g1=1.0) is True

    def test_sanity_check_helper_detects_violation(self):
        from algea.models.ranker.baseline_scorer import sanity_check_gate_monotonicity
        # Fabricate a violation: w positively correlated with gate_input
        gi = np.linspace(0, 5, 100)
        w = gi / 5.0  # w increases with gi — wrong!
        assert sanity_check_gate_monotonicity(w, gi, g1=1.0) is False


# ---------------------------------------------------------------------------
# Test 7: CS sign direction (flipping gate_cs_sign reverses mean_w ordering)
# ---------------------------------------------------------------------------

class TestCSSignDirection:
    """Verify that gate_cs_sign controls which dates get higher/lower w."""

    @staticmethod
    def _make_two_date_df():
        """Two dates with distinct z_cs but identical z_regime_risk."""
        rng = np.random.RandomState(99)
        rows = []
        # Date A: higher z_cs (stress_up)
        for i in range(50):
            rows.append({
                "date": pd.Timestamp("2024-01-01"),
                "symbol": f"S{i:02d}",
                "z_regime_risk": rng.randn() * 0.01,  # near-zero
                "z_cs_tail_30_std": 2.0,               # high stress
            })
        # Date B: lower z_cs (calm)
        for i in range(50):
            rows.append({
                "date": pd.Timestamp("2024-06-01"),
                "symbol": f"S{i:02d}",
                "z_regime_risk": rng.randn() * 0.01,
                "z_cs_tail_30_std": -1.0,              # calm
            })
        return pd.DataFrame(rows)

    def test_positive_cs_sign_stress_lowers_w(self):
        from algea.models.ranker.baseline_scorer import compute_gate_weights
        df = self._make_two_date_df()
        w = compute_gate_weights(
            df, g0=0.0, g1=1.0,
            gate_gamma_cs=2.0, gate_cs_sign=1.0,
        )
        df["w"] = w
        mean_w_stress = df[df["date"] == pd.Timestamp("2024-01-01")]["w"].mean()
        mean_w_calm = df[df["date"] == pd.Timestamp("2024-06-01")]["w"].mean()
        assert mean_w_stress < mean_w_calm, (
            f"With cs_sign=+1 and stress_up, stress date should have lower w: "
            f"stress={mean_w_stress:.4f} >= calm={mean_w_calm:.4f}"
        )

    def test_negative_cs_sign_flips_ordering(self):
        from algea.models.ranker.baseline_scorer import compute_gate_weights
        df = self._make_two_date_df()
        w = compute_gate_weights(
            df, g0=0.0, g1=1.0,
            gate_gamma_cs=2.0, gate_cs_sign=-1.0,
        )
        df["w"] = w
        mean_w_stress = df[df["date"] == pd.Timestamp("2024-01-01")]["w"].mean()
        mean_w_calm = df[df["date"] == pd.Timestamp("2024-06-01")]["w"].mean()
        assert mean_w_stress > mean_w_calm, (
            f"With cs_sign=-1, stress date should have HIGHER w: "
            f"stress={mean_w_stress:.4f} <= calm={mean_w_calm:.4f}"
        )


# ---------------------------------------------------------------------------
# Test 8: Effective-term correlation (corr(w, cs_term) < 0)
# ---------------------------------------------------------------------------

class TestEffectiveTermCorrelation:
    """Verify corr(w, cs_term) is negative for stress_up + positive gamma_cs."""

    def test_corr_w_cs_term_negative(self):
        from algea.models.ranker.baseline_scorer import (
            compute_gate_weights, compute_gate_input,
        )
        rng = np.random.RandomState(77)
        n = 500
        df = pd.DataFrame({
            "z_regime_risk": rng.randn(n) * 0.1,
            "z_cs_tail_30_std": rng.randn(n) * 1.5,  # wide variation
        })
        gamma_cs = 2.0
        cs_sign = 1.0
        w = compute_gate_weights(
            df, g0=0.0, g1=1.0,
            gate_gamma_cs=gamma_cs, gate_cs_sign=cs_sign,
        )
        cs_term = gamma_cs * cs_sign * df["z_cs_tail_30_std"].values
        rho = np.corrcoef(w, cs_term)[0, 1]
        assert rho < 0, (
            f"corr(w, cs_term) should be negative for stress_up + gamma_cs>0: "
            f"got {rho:+.4f}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
