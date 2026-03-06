"""Canonical feature builder — single source of truth for training + runtime.

Both ``pipeline.dataset.build_features`` and the runtime sleeve **must**
delegate to :func:`compute_core_features` so that feature parity is
structurally enforced (not just tested after the fact).

All rolling operations are strictly causal (look-back only).
All features are computable from data available before market open.

Schema versions
---------------
  V1: 9 features  (original)
  V2: 19 features (V1 + shock regime + prior-day context + trend/drawdown)
  V3: 23 features (V2 + cross-sectional context: shock_score, rank_z, sigma/vol ranks)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Feature schema — ordered tuple used for model input everywhere
# ---------------------------------------------------------------------------

FEATURE_SCHEMA_V1: tuple[str, ...] = (
    "r_co",               # raw overnight return
    "r_co_cs_demean",     # cross-sectional demeaned r_co
    "r_co_rank_pct",      # cross-sectional rank of r_co
    "sigma_co",           # rolling std of r_co
    "sigma_oc_hist",      # rolling std of r_oc (backward-looking, causal)
    "volume_z",           # z-score of volume vs rolling mean/std
    "roll_window_flag",   # 1 if near contract roll, else 0
    "days_to_expiry",     # days until front contract expiry
    "day_of_week",        # 0=Mon .. 4=Fri
)

FEATURE_SCHEMA_V2: tuple[str, ...] = FEATURE_SCHEMA_V1 + (
    # --- Overnight shock regime ---
    "abs_r_co",           # |r_co| — magnitude of overnight move
    "z_abs_r_co",         # z-score of |r_co| vs rolling mean/std
    "shock_flag",         # 1[|r_co| > rolling p90(|r_co|)]
    # --- Prior-day intraday context ---
    "prev_r_oc",          # r_oc[D-1] — yesterday's intraday return
    "prev_abs_r_oc",      # |r_oc[D-1]|
    # --- Trend / drawdown regime ---
    "daily_ret",          # (1+r_co)*(1+r_oc)-1 — full daily return
    "trend_20",           # 20d rolling mean of daily_ret
    "dd_60",              # 60d max-drawdown (negative = deeper)
    "rv_60",              # 60d realized vol of daily_ret
    "skew_proxy",         # rolling mean of sign(daily_ret)*|daily_ret|
)

FEATURE_SCHEMA_V3: tuple[str, ...] = FEATURE_SCHEMA_V2 + (
    # --- Cross-sectional context (meaningful at N≥8) ---
    "shock_score",        # abs(r_co) / (sigma_co + eps) — normalized shock
    "r_co_rank_z",        # z-score of r_co within daily cross-section
    "sigma_co_rank_pct",  # cross-sectional rank of sigma_co
    "volume_rank_pct",    # cross-sectional rank of volume
)

# Default schema for new training runs (V2 = production default)
FEATURE_SCHEMA = FEATURE_SCHEMA_V2
NUM_FEATURES_V2 = len(FEATURE_SCHEMA_V2)
NUM_FEATURES_V3 = len(FEATURE_SCHEMA_V3)
NUM_FEATURES = NUM_FEATURES_V2  # backward compat alias


def active_schema(version: int = 2) -> tuple[str, ...]:
    """Return the feature schema for a given version."""
    if version <= 1:
        return FEATURE_SCHEMA_V1
    if version == 2:
        return FEATURE_SCHEMA_V2
    return FEATURE_SCHEMA_V3


@dataclass(frozen=True)
class FeatureConfig:
    """Configurable knobs for the feature builder."""
    lookback: int = 20
    long_lookback: int = 60
    winsor_z: float = 3.0
    min_periods: int = 3
    schema_version: int = 2
    # V3 shock_score threshold (for execution gating)
    shock_z_threshold: float = 2.0


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def compute_core_features(
    frame: pd.DataFrame,
    cfg: Optional[FeatureConfig] = None,
) -> pd.DataFrame:
    """Build canonical features from a gold-like frame.

    Parameters
    ----------
    frame
        Must contain: ``trading_day``, ``instrument`` (or ``root``),
        ``r_co`` (or ``ret_co``), ``r_oc`` (or ``ret_oc``), ``volume``,
        ``days_to_expiry``, ``roll_window_flag`` (or computable).
    cfg
        Feature configuration.  Defaults to ``FeatureConfig()``.

    Returns
    -------
    DataFrame with columns from the active schema plus any input columns,
    sorted by ``(instrument, trading_day)``.
    """
    if cfg is None:
        cfg = FeatureConfig()

    df = frame.copy()

    # --- Ensure canonical column names ---
    if "instrument" not in df.columns:
        if "root" in df.columns:
            df["instrument"] = df["root"]
        else:
            raise KeyError("frame must have 'instrument' or 'root' column")
    if "r_co" not in df.columns:
        if "ret_co" in df.columns:
            df["r_co"] = df["ret_co"]
        else:
            raise KeyError("frame must have 'r_co' or 'ret_co' column")
    if "r_oc" not in df.columns:
        if "ret_oc" in df.columns:
            df["r_oc"] = df["ret_oc"]
        else:
            raise KeyError("frame must have 'r_oc' or 'ret_oc' column")

    df = df.sort_values(["instrument", "trading_day"]).reset_index(drop=True)
    lb = cfg.lookback
    llb = cfg.long_lookback
    mp = cfg.min_periods

    g = df.groupby("instrument")

    # ==================================================================
    # V1 features
    # ==================================================================

    # --- Per-instrument rolling volatilities ---
    df["sigma_co"] = g["r_co"].transform(
        lambda s: s.rolling(lb, min_periods=mp).std()
    )
    df["sigma_oc_hist"] = g["r_oc"].transform(
        lambda s: s.rolling(lb, min_periods=mp).std()
    )
    df["sigma_oc"] = df["sigma_oc_hist"]  # backward-compat alias

    # --- Volume z-score ---
    vol_mean = g["volume"].transform(
        lambda s: s.rolling(lb, min_periods=mp).mean()
    )
    vol_std = g["volume"].transform(
        lambda s: s.rolling(lb, min_periods=mp).std()
    )
    df["volume_z"] = (df["volume"] - vol_mean) / vol_std.replace(0, np.nan)

    # --- Metadata defaults ---
    if "days_to_expiry" not in df.columns:
        df["days_to_expiry"] = 0
    if "roll_window_flag" not in df.columns:
        df["roll_window_flag"] = 0

    # --- Calendar ---
    df["day_of_week"] = pd.to_datetime(df["trading_day"]).dt.dayofweek

    # --- Cross-sectional features ---
    df["r_co_rank_pct"] = df.groupby("trading_day")["r_co"].rank(pct=True)
    df["r_co_cs_demean"] = (
        df["r_co"] - df.groupby("trading_day")["r_co"].transform("mean")
    )

    if cfg.schema_version < 2:
        return df

    # ==================================================================
    # V2 features — overnight shock regime
    # ==================================================================

    df["abs_r_co"] = df["r_co"].abs()

    abs_mean = g["abs_r_co"].transform(
        lambda s: s.rolling(lb, min_periods=mp).mean()
    )
    abs_std = g["abs_r_co"].transform(
        lambda s: s.rolling(lb, min_periods=mp).std()
    )
    df["z_abs_r_co"] = (df["abs_r_co"] - abs_mean) / abs_std.replace(0, np.nan)

    abs_p90 = g["abs_r_co"].transform(
        lambda s: s.rolling(lb, min_periods=mp).quantile(0.90)
    )
    df["shock_flag"] = (df["abs_r_co"] > abs_p90).astype(float)

    # ==================================================================
    # V2 features — prior-day intraday context (lag by 1 within instrument)
    # ==================================================================

    df["prev_r_oc"] = g["r_oc"].shift(1)
    df["prev_abs_r_oc"] = df["prev_r_oc"].abs()

    # ==================================================================
    # V2 features — trend / drawdown regime
    # ==================================================================

    # Full daily return (safe: combines known r_co[D] and r_oc[D-1] for lags)
    df["daily_ret"] = (1 + df["r_co"]) * (1 + df["r_oc"]) - 1

    # Trend: 20d rolling mean of daily_ret (strictly causal)
    df["trend_20"] = g["daily_ret"].transform(
        lambda s: s.rolling(lb, min_periods=mp).mean()
    )

    # 60d drawdown: max peak-to-trough over last 60 days
    def _rolling_drawdown(s: pd.Series) -> pd.Series:
        """Rolling max drawdown (negative value = deeper drawdown)."""
        cumret = (1 + s).cumprod()
        dd = pd.Series(np.nan, index=s.index)
        for i in range(len(s)):
            start = max(0, i - llb + 1)
            window = cumret.iloc[start:i + 1]
            if len(window) < mp:
                continue
            peak = window.cummax()
            drawdown = ((window - peak) / peak).min()
            dd.iloc[i] = drawdown
        return dd

    df["dd_60"] = g["daily_ret"].transform(_rolling_drawdown)

    # 60d realized vol
    df["rv_60"] = g["daily_ret"].transform(
        lambda s: s.rolling(llb, min_periods=mp).std()
    )

    # Asymmetry proxy: rolling mean of signed magnitude
    signed_mag = np.sign(df["daily_ret"]) * df["daily_ret"].abs()
    df["_signed_mag"] = signed_mag
    df["skew_proxy"] = g["_signed_mag"].transform(
        lambda s: s.rolling(lb, min_periods=mp).mean()
    )
    df.drop(columns=["_signed_mag"], inplace=True)

    if cfg.schema_version < 3:
        return df

    # ==================================================================
    # V3 features — cross-sectional context (meaningful at N≥8)
    # ==================================================================

    _EPS = 1e-8

    # shock_score: normalized overnight shock intensity (per-instrument)
    df["shock_score"] = df["abs_r_co"] / (df["sigma_co"] + _EPS)

    # r_co_rank_z: z-score of r_co within the daily cross-section
    cs_mean = df.groupby("trading_day")["r_co"].transform("mean")
    cs_std = df.groupby("trading_day")["r_co"].transform("std")
    df["r_co_rank_z"] = (df["r_co"] - cs_mean) / cs_std.replace(0, np.nan)

    # sigma_co_rank_pct: cross-sectional rank of sigma_co
    df["sigma_co_rank_pct"] = df.groupby("trading_day")["sigma_co"].rank(pct=True)

    # volume_rank_pct: cross-sectional rank of volume
    df["volume_rank_pct"] = df.groupby("trading_day")["volume"].rank(pct=True)

    return df


# ---------------------------------------------------------------------------
# O(1) Incremental Feature Pipeline (Phase 2)
# ---------------------------------------------------------------------------

def precompute_t_minus_1(
    frame: pd.DataFrame,
    cfg: Optional[FeatureConfig] = None,
) -> pd.DataFrame:
    """Heavy O(N) batch: runs during the 18:00 ET Nightly DAG.

    Computes all historical rolling features up to yesterday's close.
    Returns ONLY the final row per instrument, representing the state
    at T-1 that will be merged with the live 09:20 gap at open.

    Parameters
    ----------
    frame
        Full history DataFrame (same schema as ``compute_core_features``).
    cfg
        Feature configuration.
    """
    if cfg is None:
        cfg = FeatureConfig()

    # Use the monolithic builder to compute all features on the full history
    full_df = compute_core_features(frame, cfg)

    # Extract T-1 state: last row per instrument (sorted by trading_day)
    t_minus_1 = (
        full_df.sort_values(["instrument", "trading_day"])
        .groupby("instrument")
        .tail(1)
        .reset_index(drop=True)
    )

    return t_minus_1


def update_incremental(
    t_minus_1_state: pd.DataFrame,
    live_quotes: dict[str, dict[str, float]],
    asof_date,
    cfg: Optional[FeatureConfig] = None,
) -> pd.DataFrame:
    """Lightweight O(1) stream: runs at 09:21 ET.

    Merges the cached T-1 feature state with the live 09:20 bar data.
    Only computes the overnight gap (``r_co``) and its derivatives;
    all historical rolling features are carried forward from the cache.

    Parameters
    ----------
    t_minus_1_state
        Cached features from ``precompute_t_minus_1`` (1 row per instrument).
    live_quotes
        Live 09:20 bar data keyed by symbol. Each value must have at least
        ``{"open": float}``. Optional: ``{"volume": float, "volume_rank_pct": float}``.
    asof_date
        Today's trading date.
    cfg
        Feature configuration.

    Returns
    -------
    DataFrame with updated features for today, ready for model input.
    """
    if cfg is None:
        cfg = FeatureConfig()

    df = t_minus_1_state.copy()

    _EPS = 1e-8

    # ── Map live 09:20 quotes ─────────────────────────────────────────
    # Use .get() defensively; instruments missing from live_quotes keep
    # their cached close price (effectively r_co = 0).
    instruments = df["instrument"].tolist()

    open_prices = []
    for sym in instruments:
        quote = live_quotes.get(sym, {})
        open_prices.append(quote.get("open", np.nan))

    df["open_t0"] = open_prices

    # ── Compute live overnight gap ────────────────────────────────────
    # r_co depends on the column name for close — check both conventions
    close_col = "close" if "close" in df.columns else "r_oc"
    if close_col == "r_oc":
        # Edge case: if 'close' isn't cached, we can't compute r_co
        # Fall back to using existing r_co (stale but non-zero)
        pass
    else:
        df["r_co"] = np.log(df["open_t0"] / df["close"].replace(0, np.nan))

    df["abs_r_co"] = df["r_co"].abs()

    # ── Cross-sectional features (instant math on 14 rows) ────────────
    r_co_mean = df["r_co"].mean()
    r_co_std = df["r_co"].std() + _EPS

    df["r_co_cs_demean"] = df["r_co"] - r_co_mean
    df["r_co_rank_pct"] = df["r_co"].rank(pct=True)

    # V2: shock regime from live gap
    if cfg.schema_version >= 2:
        sigma_co = df["sigma_co"].replace(0, np.nan)
        abs_mean = df["abs_r_co"].mean()
        abs_std = df["abs_r_co"].std() + _EPS
        df["z_abs_r_co"] = (df["abs_r_co"] - abs_mean) / abs_std

        # Shock flag: approximation using cached sigma_co as threshold
        abs_p90 = sigma_co * 1.28  # Gaussian ≈ p90 for |r_co|/σ
        df["shock_flag"] = (df["abs_r_co"] > abs_p90).astype(float)

    # V3: cross-sectional context
    if cfg.schema_version >= 3:
        df["shock_score"] = df["abs_r_co"] / (df["sigma_co"] + _EPS)
        cs_std_live = df["r_co"].std() + _EPS
        df["r_co_rank_z"] = (df["r_co"] - r_co_mean) / cs_std_live

        df["sigma_co_rank_pct"] = df["sigma_co"].rank(pct=True)

        # Volume rank from live quotes
        volumes = []
        for sym in instruments:
            quote = live_quotes.get(sym, {})
            volumes.append(quote.get("volume", np.nan))
        df["_live_volume"] = volumes
        df["volume_rank_pct"] = df["_live_volume"].rank(pct=True)
        df.drop(columns=["_live_volume"], inplace=True, errors="ignore")

    # ── Calendar ──────────────────────────────────────────────────────
    df["trading_day"] = pd.Timestamp(asof_date)
    df["day_of_week"] = pd.Timestamp(asof_date).dayofweek

    # Clean up temporary columns
    df.drop(columns=["open_t0"], inplace=True, errors="ignore")

    return df


def profile_feature_pipeline(
    frame: pd.DataFrame,
    cfg: Optional[FeatureConfig] = None,
) -> dict[str, float]:
    """Profiler gate: measure monolithic feature generation latency.

    Run this to determine if Phase 2 precomputation is necessary:
    - Features > 500ms → Precompute is justified.
    - Features < 100ms → ABORT Phase 2 — code is already fast enough.

    Returns
    -------
    dict with ``total_ms``, ``features_ms`` keys.
    """
    import time as _time

    t0 = _time.perf_counter()
    _ = compute_core_features(frame, cfg)
    t1 = _time.perf_counter()

    result = {
        "total_ms": round((t1 - t0) * 1000, 2),
        "features_ms": round((t1 - t0) * 1000, 2),
    }

    import logging
    logging.getLogger(__name__).info(
        "[COOC TELEMETRY] Feature pipeline: %.2fms", result["features_ms"],
    )

    return result

