"""Tests for R4: confidence-based exposure gating."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.app.portfolio.exposure_policy import (
    ExposurePolicyConfig,
    compute_day_confidence,
    compute_gross_scale,
    compute_gross_schedule,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_cfg() -> ExposurePolicyConfig:
    return ExposurePolicyConfig(
        crash_threshold=0.05,
        caution_threshold=0.15,
        caution_scale=0.5,
    )


@pytest.fixture
def std_cfg() -> ExposurePolicyConfig:
    return ExposurePolicyConfig(
        crash_threshold=0.05,
        caution_threshold=0.15,
        caution_scale=0.5,
        dispersion_method="std",
    )


# ---------------------------------------------------------------------------
# compute_day_confidence tests
# ---------------------------------------------------------------------------

class TestComputeDayConfidence:
    def test_iqr_deterministic(self, default_cfg):
        alpha = pd.Series([0.01, 0.02, 0.03, 0.04])
        sigma = pd.Series([0.01, 0.01, 0.01, 0.01])
        conf = compute_day_confidence(alpha, sigma, default_cfg)
        # IQR = q75 - q25 = 0.0325 - 0.0175 = 0.015
        # risk = median(sigma) = 0.01
        # conf = 0.015 / (1e-6 + 0.01) = ~1.4999
        assert conf == pytest.approx(0.015 / (1e-6 + 0.01), rel=1e-4)

    def test_std_deterministic(self, std_cfg):
        alpha = pd.Series([0.01, 0.02, 0.03, 0.04])
        sigma = pd.Series([0.01, 0.01, 0.01, 0.01])
        conf = compute_day_confidence(alpha, sigma, std_cfg)
        expected_std = float(alpha.std())
        expected = expected_std / (1e-6 + 0.01)
        assert conf == pytest.approx(expected, rel=1e-4)

    def test_constant_alpha_zero_dispersion(self, default_cfg):
        """Constant alpha → IQR=0 → conf=0 → crash."""
        alpha = pd.Series([0.02, 0.02, 0.02, 0.02])
        sigma = pd.Series([0.01, 0.01, 0.01, 0.01])
        conf = compute_day_confidence(alpha, sigma, default_cfg)
        assert conf == pytest.approx(0.0)

    def test_tiny_risk_high_conf(self, default_cfg):
        """Very small sigma → very high confidence."""
        alpha = pd.Series([0.01, 0.03, 0.05, 0.07])
        sigma = pd.Series([1e-10, 1e-10, 1e-10, 1e-10])
        conf = compute_day_confidence(alpha, sigma, default_cfg)
        assert conf > 100.0  # disp / eps → very large

    def test_single_alpha_returns_zero(self, default_cfg):
        """Only one instrument → can't compute dispersion."""
        alpha = pd.Series([0.02])
        sigma = pd.Series([0.01])
        conf = compute_day_confidence(alpha, sigma, default_cfg)
        assert conf == 0.0

    def test_missing_sigma_returns_zero_risk(self, default_cfg):
        """Empty sigma → risk=0, conf=disp/eps (very high)."""
        alpha = pd.Series([0.01, 0.02, 0.03, 0.04])
        sigma = pd.Series(dtype=float)
        conf = compute_day_confidence(alpha, sigma, default_cfg)
        # risk=0, conf = IQR / eps → very large
        assert conf > 1000.0

    def test_nan_alphas_excluded(self, default_cfg):
        alpha = pd.Series([0.01, np.nan, 0.03, np.nan])
        sigma = pd.Series([0.01, 0.01, 0.01, 0.01])
        conf = compute_day_confidence(alpha, sigma, default_cfg)
        # Only 2 valid alphas — IQR([0.01, 0.03]) = 0.015
        assert conf > 0


# ---------------------------------------------------------------------------
# compute_gross_scale tests
# ---------------------------------------------------------------------------

class TestComputeGrossScale:
    def test_crash_zone(self, default_cfg):
        assert compute_gross_scale(0.01, default_cfg) == 0.0
        assert compute_gross_scale(0.0, default_cfg) == 0.0

    def test_caution_zone(self, default_cfg):
        assert compute_gross_scale(0.06, default_cfg) == 0.5
        assert compute_gross_scale(0.14, default_cfg) == 0.5

    def test_normal_zone(self, default_cfg):
        assert compute_gross_scale(0.15, default_cfg) == 1.0
        assert compute_gross_scale(1.0, default_cfg) == 1.0

    def test_exact_crash_boundary(self, default_cfg):
        """At exactly crash_threshold → caution (not crash)."""
        assert compute_gross_scale(0.05, default_cfg) == 0.5

    def test_exact_caution_boundary(self, default_cfg):
        """At exactly caution_threshold → normal."""
        assert compute_gross_scale(0.15, default_cfg) == 1.0


# ---------------------------------------------------------------------------
# compute_gross_schedule tests
# ---------------------------------------------------------------------------

class TestComputeGrossSchedule:
    def test_no_sigma_col_returns_ones(self, default_cfg):
        """If sigma_col is missing, default to full gross."""
        df = pd.DataFrame({
            "trading_day": [1, 1, 2, 2],
            "model_alpha": [0.01, 0.02, 0.03, 0.04],
        })
        schedule = compute_gross_schedule(df, cfg=default_cfg)
        assert len(schedule) == 2
        assert (schedule == 1.0).all()

    def test_schedule_shape_matches_days(self, default_cfg):
        rng = np.random.RandomState(42)
        n_days, n_inst = 10, 6
        rows = []
        for d in range(n_days):
            alpha = rng.randn(n_inst) * 0.01
            sigma = np.abs(rng.randn(n_inst)) * 0.01 + 0.005
            for i in range(n_inst):
                rows.append({
                    "trading_day": d,
                    "model_alpha": alpha[i],
                    "sigma_pred": sigma[i],
                })
        df = pd.DataFrame(rows)
        schedule = compute_gross_schedule(df, cfg=default_cfg)
        assert len(schedule) == n_days

    def test_constant_alpha_produces_crash(self, default_cfg):
        """Constant alpha across instruments → zero dispersion → crash."""
        df = pd.DataFrame({
            "trading_day": [1, 1, 1, 1],
            "model_alpha": [0.02, 0.02, 0.02, 0.02],
            "sigma_pred": [0.01, 0.01, 0.01, 0.01],
        })
        schedule = compute_gross_schedule(df, cfg=default_cfg)
        assert schedule.iloc[0] == 0.0

    def test_high_dispersion_produces_normal(self, default_cfg):
        """High alpha dispersion + moderate risk → full gross."""
        df = pd.DataFrame({
            "trading_day": [1, 1, 1, 1],
            "model_alpha": [-0.10, -0.05, 0.05, 0.10],
            "sigma_pred": [0.01, 0.01, 0.01, 0.01],
        })
        schedule = compute_gross_schedule(df, cfg=default_cfg)
        assert schedule.iloc[0] == 1.0
