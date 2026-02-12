"""Tests for regime hysteresis (Phase 2)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from algaie.data.options.vrp_features import (
    RegimeState,
    RegimeThresholds,
    VolRegime,
    classify_regime,
    classify_regime_with_hysteresis,
)
from algaie.execution.options.config import VRPConfig


def _crash_features():
    """Features that score >= 2 crash signals (VIX 40)."""
    return pd.Series({
        "vix_level": 40.0,
        "vix_change_5d": 0.30,
        "vix_term_structure": -0.05,
        "rv_ratio_10_60": 2.0,
        "drawdown_63d": -0.12,
        "credit_change_5d": -0.05,
    })


def _normal_features():
    """Features that score 0 crash/caution signals."""
    return pd.Series({
        "vix_level": 14.0,
        "vix_change_5d": 0.01,
        "vix_term_structure": 0.02,
        "rv_ratio_10_60": 0.9,
        "drawdown_63d": -0.01,
        "credit_change_5d": 0.01,
    })


class TestFlipFlopSuppression:
    def test_crash_held_for_min_days(self):
        """Once in CRASH_RISK, must stay for at least min_days even if normal signals return."""
        cfg = VRPConfig(regime_min_days_in_state=3)
        state = RegimeState()

        # Day 1: Enter CRASH_RISK
        r1 = classify_regime_with_hysteresis(_crash_features(), state, config=cfg)
        assert r1 == VolRegime.CRASH_RISK

        # Day 2: Normal signals arrive but should stay in CRASH_RISK
        r2 = classify_regime_with_hysteresis(_normal_features(), state, config=cfg)
        assert r2 == VolRegime.CRASH_RISK

        # Day 3: Normal again — still held
        r3 = classify_regime_with_hysteresis(_normal_features(), state, config=cfg)
        assert r3 == VolRegime.CRASH_RISK

    def test_exit_after_min_days(self):
        """After min_days of exit-level signals, crash de-escalates to CAUTION."""
        cfg = VRPConfig(regime_min_days_in_state=2)
        state = RegimeState()

        # Enter CRASH_RISK
        classify_regime_with_hysteresis(_crash_features(), state, config=cfg)
        assert state.current_regime == VolRegime.CRASH_RISK

        # 2 consecutive normal days → should de-escalate to CAUTION (not normal)
        classify_regime_with_hysteresis(_normal_features(), state, config=cfg)
        r = classify_regime_with_hysteresis(_normal_features(), state, config=cfg)
        assert r == VolRegime.CAUTION  # ordered de-escalation: crash → caution

        # Then 2 more normal days → exits caution to normal
        classify_regime_with_hysteresis(_normal_features(), state, config=cfg)
        r2 = classify_regime_with_hysteresis(_normal_features(), state, config=cfg)
        assert r2 == VolRegime.NORMAL_CARRY

    def test_crash_signal_resets_exit_counter(self):
        """If crash signal fires again during exit countdown, counter resets."""
        cfg = VRPConfig(regime_min_days_in_state=3)
        state = RegimeState()

        classify_regime_with_hysteresis(_crash_features(), state, config=cfg)
        # 1 normal day
        classify_regime_with_hysteresis(_normal_features(), state, config=cfg)
        assert state.consecutive_exit_days == 1
        # crash returns → resets
        classify_regime_with_hysteresis(_crash_features(), state, config=cfg)
        assert state.consecutive_exit_days == 0
        assert state.current_regime == VolRegime.CRASH_RISK


class TestCautionHysteresis:
    def test_caution_held(self):
        """CAUTION should be held for min_days before dropping to NORMAL."""
        cfg = VRPConfig(regime_min_days_in_state=2)
        state = RegimeState()

        # Features that trigger CAUTION: vix=23, change=0.12
        caution_features = pd.Series({
            "vix_level": 23.0,
            "vix_change_5d": 0.12,
            "vix_term_structure": -0.02,
            "rv_ratio_10_60": 1.3,
            "drawdown_63d": -0.05,
            "credit_change_5d": -0.015,
        })

        r1 = classify_regime_with_hysteresis(caution_features, state, config=cfg)
        assert r1 == VolRegime.CAUTION

        # 1 normal day — should stay CAUTION
        r2 = classify_regime_with_hysteresis(_normal_features(), state, config=cfg)
        assert r2 == VolRegime.CAUTION

    def test_escalation_to_crash_always_allowed(self):
        """From CAUTION, always allow escalation to CRASH_RISK."""
        cfg = VRPConfig(regime_min_days_in_state=5)
        state = RegimeState()
        state.current_regime = VolRegime.CAUTION
        state.days_in_regime = 1

        r = classify_regime_with_hysteresis(_crash_features(), state, config=cfg)
        assert r == VolRegime.CRASH_RISK


class TestRegimeStatePersistence:
    def test_roundtrip(self):
        state = RegimeState(
            current_regime=VolRegime.CRASH_RISK,
            days_in_regime=5,
            consecutive_exit_days=2,
        )
        d = state.to_dict()
        restored = RegimeState.from_dict(d)
        assert restored.current_regime == VolRegime.CRASH_RISK
        assert restored.days_in_regime == 5
        assert restored.consecutive_exit_days == 2
