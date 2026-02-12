"""
VRP signal features and regime classifier.

v2: forecast-aware regime classifier with score-based scoring,
    lowered crash thresholds, Lag-Llama forecast inputs.

Computes per-underlying per-date:
  A) VRP features  (iv_rank, iv_minus_rv, term_slope, skew, vol-of-vol, liquidity)
  B) Regime signals (vix level/change, term structure, rv regime, trend, credit proxy)
  C) Regime classifier → NORMAL_CARRY / CAUTION / CRASH_RISK
"""
from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════
# Regime enum
# ═══════════════════════════════════════════════════════════════════════════

class VolRegime(enum.Enum):
    NORMAL_CARRY = "normal_carry"
    CAUTION = "caution"
    CRASH_RISK = "crash_risk"


# ═══════════════════════════════════════════════════════════════════════════
# Config for threshold tuning (v2 — lowered defaults)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class RegimeThresholds:
    """Configurable thresholds for the rules-based regime classifier."""
    # VIX
    vix_caution: float = 22.0               # lowered from 25
    vix_crash: float = 28.0                 # lowered from 35
    vix_change_5d_crash: float = 0.18       # 18% (from 30%)
    vix_change_5d_caution: float = 0.10

    # Realised vol regime
    rv_ratio_crash: float = 1.5             # from 1.8
    rv_ratio_caution: float = 1.2

    # Drawdown
    drawdown_63d_crash: float = -0.08       # from -0.10
    drawdown_63d_caution: float = -0.04

    # Term structure (negative = backwardation → risky)
    term_structure_crash: float = -0.03
    term_structure_caution: float = -0.01

    # Credit proxy (HYG/LQD ratio change)
    credit_change_5d_crash: float = -0.02
    credit_change_5d_caution: float = -0.01


# ═══════════════════════════════════════════════════════════════════════════
# VRP Feature computation
# ═══════════════════════════════════════════════════════════════════════════

def compute_iv_rank_252(iv_series: pd.Series) -> pd.Series:
    """Percentile rank of IV over trailing 252 trading days."""
    return iv_series.rolling(252, min_periods=60).apply(
        lambda x: float(np.searchsorted(np.sort(x[:-1]), x[-1])) / max(len(x) - 1, 1),
        raw=True,
    )


def compute_realized_vol(close: pd.Series, window: int = 30) -> pd.Series:
    """Annualised realised volatility from log returns."""
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(window, min_periods=max(window // 2, 5)).std() * np.sqrt(252)


def compute_vrp_features(
    close: pd.Series,
    atm_iv: pd.Series,
    term_slope: pd.Series,
    skew_25d: pd.Series,
    avg_spread_pct: pd.Series,
    avg_oi: pd.Series,
    avg_volume: pd.Series,
) -> pd.DataFrame:
    """Compute VRP signal features.  All inputs must be same-indexed (date)."""
    rv30 = compute_realized_vol(close, 30)
    rv10 = compute_realized_vol(close, 10)
    rv60 = compute_realized_vol(close, 60)

    features = pd.DataFrame(index=close.index)
    features["iv_rank_252"] = compute_iv_rank_252(atm_iv)
    features["iv_minus_rv30"] = atm_iv - rv30
    features["term_slope"] = term_slope
    features["skew_25d"] = skew_25d
    features["vol_of_vol"] = atm_iv.rolling(20, min_periods=10).std()
    features["rv10"] = rv10
    features["rv30"] = rv30
    features["rv60"] = rv60
    features["rv_ratio_10_60"] = rv10 / rv60.replace(0, np.nan)
    features["avg_spread_pct"] = avg_spread_pct
    features["avg_oi"] = avg_oi
    features["avg_volume"] = avg_volume
    return features


# ═══════════════════════════════════════════════════════════════════════════
# Regime feature computation
# ═══════════════════════════════════════════════════════════════════════════

def compute_regime_features(
    close: pd.Series,
    vix: pd.Series,
    vix_term_structure: Optional[pd.Series] = None,
    credit_ratio: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Compute regime-relevant features from market data."""
    rf = pd.DataFrame(index=close.index)
    rf["vix_level"] = vix
    rf["vix_change_5d"] = vix.pct_change(5)

    if vix_term_structure is not None:
        rf["vix_term_structure"] = vix_term_structure
    else:
        rf["vix_term_structure"] = 0.0

    rv10 = compute_realized_vol(close, 10)
    rv60 = compute_realized_vol(close, 60)
    rf["rv_ratio_10_60"] = rv10 / rv60.replace(0, np.nan)

    # Trend filter
    ma50 = close.rolling(50, min_periods=30).mean()
    ma200 = close.rolling(200, min_periods=100).mean()
    rf["trend_50_200"] = (ma50 / ma200) - 1.0
    high_63 = close.rolling(63, min_periods=30).max()
    rf["drawdown_63d"] = (close / high_63) - 1.0

    # Credit proxy
    if credit_ratio is not None:
        rf["credit_ratio"] = credit_ratio
        rf["credit_change_5d"] = credit_ratio.pct_change(5)
    else:
        rf["credit_ratio"] = np.nan
        rf["credit_change_5d"] = 0.0

    return rf


# ═══════════════════════════════════════════════════════════════════════════
# Regime classifier (v2 — forecast-aware, score-based, lowered thresholds)
# ═══════════════════════════════════════════════════════════════════════════

def classify_regime(
    regime_features: pd.Series,
    thresholds: RegimeThresholds = RegimeThresholds(),
    forecast_inputs: Optional[Dict[str, Any]] = None,
    config: Optional[Any] = None,
) -> VolRegime:
    """Classify a single date's regime features into a VolRegime.

    v2: integrates Lag-Llama forecast inputs for forward-looking regime.
    Classic signals + forecast signals both contribute to regime score.

    Parameters
    ----------
    regime_features : Series with keys matching ``compute_regime_features`` output.
    thresholds : configurable thresholds.
    forecast_inputs : optional dict with keys like "rv10_pred_p90", "rv10_pred_p95",
                      "rv10_pred_p99", "health_score".
    config : optional VRPConfig for forecast thresholds.
    """
    vix = regime_features.get("vix_level", 15.0)
    vix_chg = regime_features.get("vix_change_5d", 0.0)
    term = regime_features.get("vix_term_structure", 0.0)
    rv_ratio = regime_features.get("rv_ratio_10_60", 1.0)
    dd = regime_features.get("drawdown_63d", 0.0)
    credit_chg = regime_features.get("credit_change_5d", 0.0)

    # Handle NaN
    vix = vix if not np.isnan(vix) else 15.0
    vix_chg = vix_chg if not np.isnan(vix_chg) else 0.0
    rv_ratio = rv_ratio if not np.isnan(rv_ratio) else 1.0
    dd = dd if not np.isnan(dd) else 0.0
    credit_chg = credit_chg if not np.isnan(credit_chg) else 0.0

    # Extract forecast thresholds from config (fall back to defaults)
    min_health = 0.80
    rv10_p90_caution = 0.28
    rv10_p95_crash = 0.35
    rv10_p99_crash = 0.45
    forecast_weight = 1.0
    credit_crash = 0.03

    if config is not None:
        min_health = getattr(config, "min_forecast_health", min_health)
        rv10_p90_caution = getattr(config, "rv10_pred_p90_caution", rv10_p90_caution)
        rv10_p95_crash = getattr(config, "rv10_pred_p95_crash", rv10_p95_crash)
        rv10_p99_crash = getattr(config, "rv10_pred_p99_crash", rv10_p99_crash)
        forecast_weight = getattr(config, "forecast_override_weight", forecast_weight)
        credit_crash = getattr(config, "regime_credit_spread_crash", credit_crash)

    # ---- Classic CRASH_RISK signals ----
    crash_signals = 0
    if vix >= thresholds.vix_crash:
        crash_signals += 2
    if vix_chg >= thresholds.vix_change_5d_crash:
        crash_signals += 1
        if term < thresholds.term_structure_crash:
            crash_signals += 1
    if rv_ratio >= thresholds.rv_ratio_crash and dd <= thresholds.drawdown_63d_crash:
        crash_signals += 2
    if credit_chg <= thresholds.credit_change_5d_crash:
        crash_signals += 1

    # ---- Forecast CRASH_RISK signals ----
    forecast_crash = False
    forecast_caution = False
    forecast_health_low = False      # health fail-safe flag
    forecast_health_value = 1.0

    if forecast_inputs is not None:
        fh = forecast_inputs.get("health_score", 0.0)
        forecast_health_value = fh
        rv_p90 = forecast_inputs.get("rv10_pred_p90", None)
        rv_p95 = forecast_inputs.get("rv10_pred_p95", None)
        rv_p99 = forecast_inputs.get("rv10_pred_p99", None)

        if fh < min_health:
            # Health fail-safe: don't trust forecasts, but flag low health
            forecast_health_low = True
            forecast_weight = 0.0  # zero out forecast contribution
        else:
            # Crash: p95 or p99 exceeds threshold
            if rv_p95 is not None and rv_p95 > rv10_p95_crash:
                forecast_crash = True
            if rv_p99 is not None and rv_p99 > rv10_p99_crash:
                forecast_crash = True
            # Caution: p90 exceeds threshold
            if rv_p90 is not None and rv_p90 > rv10_p90_caution:
                forecast_caution = True

    if forecast_crash:
        crash_signals += int(2 * forecast_weight)

    if crash_signals >= 2:
        return VolRegime.CRASH_RISK

    # ---- Classic CAUTION signals ----
    caution_signals = 0
    if vix >= thresholds.vix_caution:
        caution_signals += 1
    if vix_chg >= thresholds.vix_change_5d_caution:
        caution_signals += 1
    if term < thresholds.term_structure_caution:
        caution_signals += 1
    if rv_ratio >= thresholds.rv_ratio_caution:
        caution_signals += 1
    if dd <= thresholds.drawdown_63d_caution:
        caution_signals += 1
    if credit_chg <= thresholds.credit_change_5d_caution:
        caution_signals += 1

    # ---- Forecast CAUTION ----
    if forecast_caution:
        caution_signals += 1

    if caution_signals >= 2:
        return VolRegime.CAUTION

    # ---- Forecast health fail-safe ----
    # If health < min_health → block NORMAL_CARRY, enforce CAUTION minimum
    if forecast_health_low:
        return VolRegime.CAUTION

    return VolRegime.NORMAL_CARRY


def classify_regime_series(
    regime_features: pd.DataFrame,
    thresholds: RegimeThresholds = RegimeThresholds(),
    forecast_inputs_by_date: Optional[Dict[str, Dict[str, Any]]] = None,
    config: Optional[Any] = None,
) -> pd.Series:
    """Classify regime for every row in a regime features DataFrame."""
    def _classify_row(row: pd.Series) -> VolRegime:
        fi = None
        if forecast_inputs_by_date is not None:
            dt_key = str(row.name)
            fi = forecast_inputs_by_date.get(dt_key)
        return classify_regime(row, thresholds, forecast_inputs=fi, config=config)

    return regime_features.apply(_classify_row, axis=1)


# ═══════════════════════════════════════════════════════════════════════════
# Regime hysteresis (Phase 2 — prevent flip-flop)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class RegimeState:
    """Mutable state for regime hysteresis tracking."""
    current_regime: VolRegime = VolRegime.NORMAL_CARRY
    days_in_regime: int = 0
    consecutive_exit_days: int = 0  # days below exit threshold

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_regime": self.current_regime.value,
            "days_in_regime": self.days_in_regime,
            "consecutive_exit_days": self.consecutive_exit_days,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RegimeState":
        return cls(
            current_regime=VolRegime(d.get("current_regime", "normal_carry")),
            days_in_regime=d.get("days_in_regime", 0),
            consecutive_exit_days=d.get("consecutive_exit_days", 0),
        )


def _compute_regime_score(
    regime_features: pd.Series,
    thresholds: RegimeThresholds,
    forecast_inputs: Optional[Dict[str, Any]] = None,
    config: Optional[Any] = None,
) -> int:
    """Compute raw regime score (crash signal count).

    Extracted from classify_regime so hysteresis can operate on the score.
    """
    vix = regime_features.get("vix_level", 15.0)
    vix_chg = regime_features.get("vix_change_5d", 0.0)
    term = regime_features.get("vix_term_structure", 0.0)
    rv_ratio = regime_features.get("rv_ratio_10_60", 1.0)
    dd = regime_features.get("drawdown_63d", 0.0)
    credit_chg = regime_features.get("credit_change_5d", 0.0)

    # Handle NaN
    vix = vix if not np.isnan(vix) else 15.0
    vix_chg = vix_chg if not np.isnan(vix_chg) else 0.0
    rv_ratio = rv_ratio if not np.isnan(rv_ratio) else 1.0
    dd = dd if not np.isnan(dd) else 0.0
    credit_chg = credit_chg if not np.isnan(credit_chg) else 0.0

    forecast_weight = 1.0
    min_health = 0.80
    rv10_p95_crash = 0.35
    rv10_p99_crash = 0.45
    if config is not None:
        min_health = getattr(config, "min_forecast_health", min_health)
        rv10_p95_crash = getattr(config, "rv10_pred_p95_crash", rv10_p95_crash)
        rv10_p99_crash = getattr(config, "rv10_pred_p99_crash", rv10_p99_crash)
        forecast_weight = getattr(config, "forecast_override_weight", forecast_weight)

    score = 0
    if vix >= thresholds.vix_crash:
        score += 2
    if vix_chg >= thresholds.vix_change_5d_crash:
        score += 1
        if term < thresholds.term_structure_crash:
            score += 1
    if rv_ratio >= thresholds.rv_ratio_crash and dd <= thresholds.drawdown_63d_crash:
        score += 2
    credit_crash = getattr(config, "regime_credit_spread_crash", 0.03) if config else 0.03
    if credit_chg <= -credit_crash:
        score += 1

    # Forecast signals
    if forecast_inputs is not None:
        fh = forecast_inputs.get("health_score", 0.0)
        rv_p95 = forecast_inputs.get("rv10_pred_p95", None)
        rv_p99 = forecast_inputs.get("rv10_pred_p99", None)
        if fh >= min_health:
            if (rv_p95 is not None and rv_p95 > rv10_p95_crash) or \
               (rv_p99 is not None and rv_p99 > rv10_p99_crash):
                score += int(2 * forecast_weight)

    return score


def classify_regime_with_hysteresis(
    regime_features: pd.Series,
    state: RegimeState,
    thresholds: RegimeThresholds = RegimeThresholds(),
    forecast_inputs: Optional[Dict[str, Any]] = None,
    config: Optional[Any] = None,
) -> VolRegime:
    """Classify regime with hysteresis to prevent daily flip-flop.

    Rules:
    - Enter CRASH_RISK when crash_score >= crash_entry_threshold
    - Exit CRASH_RISK only when crash_score <= crash_exit_threshold
      for regime_min_days_in_state consecutive days
    - Same entry/exit split for CAUTION

    Mutates ``state`` in-place and returns the classified regime.
    """
    # Get config thresholds
    crash_entry = 2
    crash_exit = 0
    caution_entry = 2
    caution_exit = 0
    min_days = 2

    if config is not None:
        crash_entry = getattr(config, "regime_crash_entry_score", crash_entry)
        crash_exit = getattr(config, "regime_crash_exit_score", crash_exit)
        caution_entry = getattr(config, "regime_caution_entry_score", caution_entry)
        caution_exit = getattr(config, "regime_caution_exit_score", caution_exit)
        min_days = getattr(config, "regime_min_days_in_state", min_days)

    crash_score = _compute_regime_score(regime_features, thresholds, forecast_inputs, config)

    # Also compute caution score (same as in classify_regime)
    raw_regime = classify_regime(regime_features, thresholds, forecast_inputs, config)

    prev = state.current_regime

    # ---- Hysteresis logic ----
    if prev == VolRegime.CRASH_RISK:
        if crash_score <= crash_exit:
            state.consecutive_exit_days += 1
        else:
            state.consecutive_exit_days = 0

        if state.consecutive_exit_days >= min_days:
            # De-escalate CRASH → CAUTION (never skip to NORMAL)
            de_escalated = VolRegime.CAUTION if raw_regime != VolRegime.CRASH_RISK else VolRegime.CAUTION
            state.current_regime = de_escalated
            state.days_in_regime = 1
            state.consecutive_exit_days = 0
        else:
            # Stay in CRASH_RISK
            state.days_in_regime += 1
            return VolRegime.CRASH_RISK

    elif prev == VolRegime.CAUTION:
        if raw_regime == VolRegime.CRASH_RISK:
            # Always allow escalation
            state.current_regime = VolRegime.CRASH_RISK
            state.days_in_regime = 1
            state.consecutive_exit_days = 0
        elif raw_regime == VolRegime.NORMAL_CARRY:
            state.consecutive_exit_days += 1
            if state.consecutive_exit_days >= min_days:
                state.current_regime = VolRegime.NORMAL_CARRY
                state.days_in_regime = 1
                state.consecutive_exit_days = 0
            else:
                state.days_in_regime += 1
                return VolRegime.CAUTION
        else:
            state.consecutive_exit_days = 0
            state.days_in_regime += 1

    else:
        # NORMAL_CARRY — check for entry into higher states
        state.current_regime = raw_regime
        if raw_regime == prev:
            state.days_in_regime += 1
        else:
            state.days_in_regime = 1
            state.consecutive_exit_days = 0

    return state.current_regime
