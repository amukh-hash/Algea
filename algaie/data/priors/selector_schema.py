"""
Selector feature schema — canonical column definitions for the priors-frame.

Every column used in selector training / inference is defined here.
Downstream code should import these lists to ensure consistency.

Units
-----
* Raw teacher priors: **end-of-horizon cumulative returns**
  (``predicted_price_t+H / price_t − 1``).
* Derived features: differences of the above (same unit).
* z-features: dimensionless cross-sectional z-scores, clamped to [-5, +5].
* Targets: ``y_ret_H`` = log(close[t+H] / close[t]);
  ``y_vol_H`` = annualised realised volatility over [t, t+H].
"""
from __future__ import annotations

import hashlib
import json
from typing import List

# ═══════════════════════════════════════════════════════════════════════════
# Raw teacher priors (8 per teacher × 2 teachers = 16)
# ═══════════════════════════════════════════════════════════════════════════

_RAW_FIELDS = [
    "drift", "vol_forecast", "tail_risk", "prob_up",
    "q10", "q50", "q90", "dispersion",
]

TEACHER_PRIORS_COLS_10: List[str] = [f"{f}_10" for f in _RAW_FIELDS]
TEACHER_PRIORS_COLS_30: List[str] = [f"{f}_30" for f in _RAW_FIELDS]
TEACHER_PRIORS_COLS: List[str] = TEACHER_PRIORS_COLS_10 + TEACHER_PRIORS_COLS_30

# ═══════════════════════════════════════════════════════════════════════════
# Derived features (per horizon)
# ═══════════════════════════════════════════════════════════════════════════

DERIVED_COLS_10: List[str] = ["iqr_10", "upside_10", "downside_10", "skew_10"]
DERIVED_COLS_30: List[str] = ["iqr_30", "upside_30", "downside_30", "skew_30"]
DERIVED_COLS: List[str] = DERIVED_COLS_10 + DERIVED_COLS_30

# ═══════════════════════════════════════════════════════════════════════════
# Features selected for z-scoring  (input to selector model)
# ═══════════════════════════════════════════════════════════════════════════

# Scalar features that are z-scored cross-sectionally per date
ZSCORE_SOURCE_COLS: List[str] = [
    # 10-day teacher  (prob_up_10, iqr_10, upside_10 removed — dead features)
    "drift_10", "tail_risk_10",
    # 30-day teacher
    "drift_30", "prob_up_30", "iqr_30", "tail_risk_30",
]

# Features removed from model inputs: std=0 across universe (dead features)
DEAD_ZSCORE_COLS: List[str] = [
    "prob_up_10", "iqr_10", "upside_10",  # 10d dead
    "prob_up_30", "iqr_30",                # 30d dead (verified by auditor)
]

Z_FEATURE_COLS: List[str] = [f"z_{c}" for c in ZSCORE_SOURCE_COLS]


# Agreement / multi-horizon features (computed from z-scored values)
AGREEMENT_COLS: List[str] = [
    "z_drift_gap",    # z_drift_10 - z_drift_30
    "z_tail_gap",     # z_tail_risk_10 - z_tail_risk_30
    "z_regime_risk",  # z_iqr_30 (alias for regime-level risk signal)
]

# All model input features
MODEL_FEATURE_COLS: List[str] = Z_FEATURE_COLS + AGREEMENT_COLS

# ═══════════════════════════════════════════════════════════════════════════
# Targets
# ═══════════════════════════════════════════════════════════════════════════

TARGET_COLS: List[str] = [
    "y_ret",   # log(close[t+H] / close[t])
    "y_vol",   # realised vol over [t, t+H], annualised
]

# ═══════════════════════════════════════════════════════════════════════════
# Metadata columns
# ═══════════════════════════════════════════════════════════════════════════

META_COLS: List[str] = [
    "date",
    "symbol",
    "run_id_teacher_10d",
    "run_id_teacher_30d",
    "context_len",
    "horizon_ret",
    "feature_version_hash",
]

# ═══════════════════════════════════════════════════════════════════════════
# Full frame columns (in order)
# ═══════════════════════════════════════════════════════════════════════════

ALL_FRAME_COLS: List[str] = (
    META_COLS + TEACHER_PRIORS_COLS + DERIVED_COLS
    + Z_FEATURE_COLS + AGREEMENT_COLS + TARGET_COLS
)


# ═══════════════════════════════════════════════════════════════════════════
# Feature version hash
# ═══════════════════════════════════════════════════════════════════════════

def feature_version_hash() -> str:
    """Stable hash of the feature schema for reproducibility tracking.

    Changes whenever column names, ordering, or derivation logic change.
    """
    payload = {
        "raw": TEACHER_PRIORS_COLS,
        "derived": DERIVED_COLS,
        "z_source": ZSCORE_SOURCE_COLS,
        "z_features": Z_FEATURE_COLS,
        "agreement": AGREEMENT_COLS,
        "targets": TARGET_COLS,
        "version": 1,
    }
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:12]
