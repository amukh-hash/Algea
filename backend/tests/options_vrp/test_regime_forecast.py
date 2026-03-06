"""
Tests for forecast-aware regime classifier — MUST trigger CRASH_RISK on stress fixture.
"""
import csv
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from algae.data.options.vrp_features import (
    RegimeThresholds,
    VolRegime,
    classify_regime,
    classify_regime_series,
    compute_regime_features,
)
from algae.execution.options.config import VRPConfig


_FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "stress_window.csv"


def _load_stress_fixture() -> pd.DataFrame:
    df = pd.read_csv(_FIXTURE_PATH, parse_dates=["date"])
    df = df.set_index("date").sort_index()
    return df


class TestCrashMustTrigger:
    """The regime classifier MUST produce CRASH_RISK during the COVID stress window."""

    @pytest.fixture
    def stress_data(self):
        return _load_stress_fixture()

    def test_crash_risk_fires_in_stress(self, stress_data):
        close = stress_data["close"]
        vix = stress_data["vix"]

        regime_feats = compute_regime_features(close, vix)
        thresholds = RegimeThresholds()

        regimes = classify_regime_series(regime_feats, thresholds)
        crash_days = regimes[regimes == VolRegime.CRASH_RISK]

        assert len(crash_days) > 0, (
            "CRASH_RISK must fire at least once during 2020 stress window. "
            f"Got regimes: {regimes.value_counts().to_dict()}"
        )

    def test_crash_risk_fires_on_peak_stress_day(self, stress_data):
        """VIX=82.69 + massive drawdown → CRASH_RISK on 2020-03-16."""
        close = stress_data["close"]
        vix = stress_data["vix"]
        regime_feats = compute_regime_features(close, vix)

        # Find the peak VIX day
        peak_idx = vix.idxmax()
        if peak_idx in regime_feats.index:
            row = regime_feats.loc[peak_idx]
            regime = classify_regime(row, RegimeThresholds())
            assert regime == VolRegime.CRASH_RISK, (
                f"Peak stress day {peak_idx}: VIX={vix.loc[peak_idx]}, "
                f"regime={regime.value}. Should be crash_risk."
            )

    def test_crash_risk_with_forecast_amplification(self, stress_data):
        """Forecast inputs should amplify CRASH_RISK on moderate stress days."""
        close = stress_data["close"]
        vix = stress_data["vix"]
        regime_feats = compute_regime_features(close, vix)

        cfg = VRPConfig()

        # Take a moderate stress day (VIX ~25-30)
        moderate = regime_feats[(vix > 25) & (vix < 35)].iloc[0] if len(regime_feats[(vix > 25) & (vix < 35)]) > 0 else None
        if moderate is not None:
            # Without forecast
            regime_no_fc = classify_regime(moderate, RegimeThresholds())
            # With high forecast RV
            forecast = {
                "rv10_pred_p95": 0.50,
                "rv10_pred_p99": 0.70,
                "health_score": 0.95,
            }
            regime_with_fc = classify_regime(
                moderate, RegimeThresholds(), forecast_inputs=forecast, config=cfg,
            )
            # With forecast, should be at least as severe
            severity = {VolRegime.NORMAL_CARRY: 0, VolRegime.CAUTION: 1, VolRegime.CRASH_RISK: 2}
            assert severity[regime_with_fc] >= severity[regime_no_fc]


class TestForecastRegime:
    def test_forecast_crash_fires_when_healthy(self):
        """High forecast RV with good health → CRASH_RISK."""
        row = pd.Series({
            "vix_level": 20.0,  # calm VIX
            "vix_change_5d": 0.05,
            "vix_term_structure": 0.01,
            "rv_ratio_10_60": 1.0,
            "drawdown_63d": -0.02,
            "credit_change_5d": 0.0,
        })
        cfg = VRPConfig(rv10_pred_p95_crash=0.35)
        forecast = {
            "rv10_pred_p95": 0.50,
            "health_score": 0.95,
        }
        regime = classify_regime(row, RegimeThresholds(), forecast_inputs=forecast, config=cfg)
        assert regime == VolRegime.CRASH_RISK

    def test_forecast_ignored_when_unhealthy(self):
        """Low health → forecast ignored, health fail-safe upgrades to CAUTION."""
        row = pd.Series({
            "vix_level": 15.0,
            "vix_change_5d": 0.02,
            "vix_term_structure": 0.01,
            "rv_ratio_10_60": 0.9,
            "drawdown_63d": -0.01,
            "credit_change_5d": 0.0,
        })
        cfg = VRPConfig(min_forecast_health=0.80)
        forecast = {
            "rv10_pred_p95": 0.60,
            "health_score": 0.50,  # below min
        }
        regime = classify_regime(row, RegimeThresholds(), forecast_inputs=forecast, config=cfg)
        # v3: low health triggers CAUTION fail-safe (not CRASH — forecast ignored)
        assert regime == VolRegime.CAUTION

    def test_caution_from_forecast_p90(self):
        """Moderate forecast RV → CAUTION."""
        row = pd.Series({
            "vix_level": 18.0,
            "vix_change_5d": 0.12,  # > 0.10 caution threshold → 1 caution signal
            "vix_term_structure": 0.00,
            "rv_ratio_10_60": 1.1,
            "drawdown_63d": -0.02,
            "credit_change_5d": 0.0,
        })
        cfg = VRPConfig(rv10_pred_p90_caution=0.28)
        forecast = {
            "rv10_pred_p90": 0.32,
            "rv10_pred_p95": 0.30,  # below crash
            "health_score": 0.90,
        }
        regime = classify_regime(row, RegimeThresholds(), forecast_inputs=forecast, config=cfg)
        assert regime in (VolRegime.CAUTION, VolRegime.CRASH_RISK)
