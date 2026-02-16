"""Feature contract system — formal spec for every model input feature.

Each feature has:
  - availability: when the value is knowable (preopen / intraday / eod)
  - optional:     whether training can proceed without it
  - is_label_or_future: True if this feature contains same-day future info
  - timestamp_rule: what timestamp the feature value is "as-of"

The FeatureTimestampGuard in dataset.py uses this to decide which
features are safe to include as model inputs for a pre-open signal.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Literal

# ---------------------------------------------------------------------------
# Core dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FeatureSpec:
    """Contract for a single feature column."""

    name: str
    availability: Literal["preopen", "intraday", "eod"]
    optional: bool
    description: str
    timestamp_rule: Literal[
        "asof_open", "asof_prev_close", "asof_eod", "unknown"
    ]
    is_label_or_future: bool = False


# ---------------------------------------------------------------------------
# V2 feature registry (single source of truth)
# ---------------------------------------------------------------------------

FEATURE_SPECS: Dict[str, FeatureSpec] = {
    # --- V1 core features (all preopen) ---
    "r_co": FeatureSpec(
        name="r_co",
        availability="preopen",
        optional=False,
        description="Raw overnight return (close-to-open).",
        timestamp_rule="asof_open",
    ),
    "r_co_cs_demean": FeatureSpec(
        name="r_co_cs_demean",
        availability="preopen",
        optional=False,
        description="Cross-sectional demeaned r_co.",
        timestamp_rule="asof_open",
    ),
    "r_co_rank_pct": FeatureSpec(
        name="r_co_rank_pct",
        availability="preopen",
        optional=False,
        description="Cross-sectional percentile rank of r_co.",
        timestamp_rule="asof_open",
    ),
    "sigma_co": FeatureSpec(
        name="sigma_co",
        availability="preopen",
        optional=False,
        description="Rolling std of r_co (backward-looking).",
        timestamp_rule="asof_prev_close",
    ),
    "sigma_oc_hist": FeatureSpec(
        name="sigma_oc_hist",
        availability="preopen",
        optional=False,
        description=(
            "Rolling std of r_oc (backward-looking over prior days). "
            "Strictly causal: uses only closed r_oc values from D-1, D-2, … "
            "Canonical name for sigma_oc."
        ),
        timestamp_rule="asof_prev_close",
    ),
    # Backward-compat alias for sigma_oc_hist (one deprecation cycle)
    "sigma_oc": FeatureSpec(
        name="sigma_oc",
        availability="preopen",
        optional=True,
        description=(
            "DEPRECATED alias for sigma_oc_hist. "
            "Rolling std of r_oc (backward-looking). Will be removed."
        ),
        timestamp_rule="asof_prev_close",
    ),
    "volume_z": FeatureSpec(
        name="volume_z",
        availability="preopen",
        optional=False,
        description="Z-score of volume vs rolling mean/std.",
        timestamp_rule="asof_prev_close",
    ),
    "roll_window_flag": FeatureSpec(
        name="roll_window_flag",
        availability="preopen",
        optional=False,
        description="1 if near contract roll window, else 0.",
        timestamp_rule="asof_open",
    ),
    "days_to_expiry": FeatureSpec(
        name="days_to_expiry",
        availability="preopen",
        optional=False,
        description="Calendar days until front contract expiry.",
        timestamp_rule="asof_open",
    ),
    "day_of_week": FeatureSpec(
        name="day_of_week",
        availability="preopen",
        optional=False,
        description="Day of week (0=Mon .. 4=Fri).",
        timestamp_rule="asof_open",
    ),

    # --- V2 additions ---
    "abs_r_co": FeatureSpec(
        name="abs_r_co",
        availability="preopen",
        optional=True,
        description="|r_co| — magnitude of overnight move.",
        timestamp_rule="asof_open",
    ),
    "z_abs_r_co": FeatureSpec(
        name="z_abs_r_co",
        availability="preopen",
        optional=True,
        description="z-score of |r_co| vs rolling mean/std.",
        timestamp_rule="asof_open",
    ),
    "shock_flag": FeatureSpec(
        name="shock_flag",
        availability="preopen",
        optional=True,
        description="1 if |r_co| > rolling p90(|r_co|).",
        timestamp_rule="asof_open",
    ),
    "prev_r_oc": FeatureSpec(
        name="prev_r_oc",
        availability="preopen",
        optional=True,
        description="r_oc[D-1] — yesterday's intraday return.",
        timestamp_rule="asof_prev_close",
    ),
    "prev_abs_r_oc": FeatureSpec(
        name="prev_abs_r_oc",
        availability="preopen",
        optional=True,
        description="|r_oc[D-1]|.",
        timestamp_rule="asof_prev_close",
    ),
    "daily_ret": FeatureSpec(
        name="daily_ret",
        availability="preopen",
        optional=True,
        description="(1+r_co)*(1+r_oc)-1 — full daily return (prior day).",
        timestamp_rule="asof_prev_close",
    ),
    "trend_20": FeatureSpec(
        name="trend_20",
        availability="preopen",
        optional=True,
        description="20d rolling mean of daily_ret.",
        timestamp_rule="asof_prev_close",
    ),
    "dd_60": FeatureSpec(
        name="dd_60",
        availability="preopen",
        optional=True,
        description="60d max-drawdown (negative = deeper).",
        timestamp_rule="asof_prev_close",
    ),
    "rv_60": FeatureSpec(
        name="rv_60",
        availability="preopen",
        optional=True,
        description="60d realized vol of daily_ret.",
        timestamp_rule="asof_prev_close",
    ),
    "skew_proxy": FeatureSpec(
        name="skew_proxy",
        availability="preopen",
        optional=True,
        description="Rolling mean of sign(daily_ret)*|daily_ret|.",
        timestamp_rule="asof_prev_close",
    ),

    # --- Label / future-information columns (never model inputs) ---
    "r_oc": FeatureSpec(
        name="r_oc",
        availability="eod",
        optional=True,
        description="Same-day open-to-close return (label source).",
        timestamp_rule="asof_eod",
        is_label_or_future=True,
    ),
    "ret_oc": FeatureSpec(
        name="ret_oc",
        availability="eod",
        optional=True,
        description="Legacy alias for r_oc.",
        timestamp_rule="asof_eod",
        is_label_or_future=True,
    ),
    "rolling_std_ret_oc": FeatureSpec(
        name="rolling_std_ret_oc",
        availability="preopen",
        optional=True,
        description="Legacy alias for sigma_oc_hist.",
        timestamp_rule="asof_prev_close",
    ),
}

# ---------------------------------------------------------------------------
# Derived lists
# ---------------------------------------------------------------------------

PREOPEN_FEATURES: List[str] = [
    name for name, spec in FEATURE_SPECS.items()
    if spec.availability == "preopen" and not spec.is_label_or_future
]

REQUIRED_PREOPEN_FEATURES: List[str] = [
    name for name, spec in FEATURE_SPECS.items()
    if spec.availability == "preopen" and not spec.optional
    and not spec.is_label_or_future
]

OPTIONAL_PREOPEN_FEATURES: List[str] = [
    name for name, spec in FEATURE_SPECS.items()
    if spec.availability == "preopen" and spec.optional
    and not spec.is_label_or_future
]

# EOD / intraday features — these are NEVER used as model inputs
EOD_FEATURES: List[str] = [
    name for name, spec in FEATURE_SPECS.items()
    if spec.availability == "eod"
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def preopen_features() -> List[str]:
    """Return list of feature names available before market open."""
    return list(PREOPEN_FEATURES)


def risky_features_for_preopen() -> FrozenSet[str]:
    """Return set of features that must NOT be used as pre-open model inputs.

    Includes:
      - Any feature with is_label_or_future=True (same-day future info)
      - Any feature with availability != "preopen" (intraday/eod data)
    """
    return frozenset(
        name for name, spec in FEATURE_SPECS.items()
        if spec.is_label_or_future or spec.availability != "preopen"
    )
