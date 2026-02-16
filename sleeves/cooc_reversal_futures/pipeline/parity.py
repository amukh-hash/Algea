"""Feature parity harness: training pipeline vs runtime sleeve features.

Compares features produced by ``dataset.build_features()`` (training path)
against ``sleeve.compute_signal_frame()`` (runtime path) to detect divergence.
"""
from __future__ import annotations

import json
import logging
from datetime import date, time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .types import FeatureParityReport

logger = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# Feature column name mapping: training ↔ runtime
# ---------------------------------------------------------------------------

# Training (dataset.py)       → Runtime (features.py)
_COLUMN_MAP_TRAIN_TO_RUNTIME = {
    "ret_co_rank_pct": "r_co_rank_pct",
    "ret_co_cs_demean": "r_co_cs_demean",
    "rolling_std_ret_co": "r_co_vol_l",
    "rolling_std_ret_oc": "r_oc_vol_l",
}

# The reverse map for aligning runtime → training
_COLUMN_MAP_RUNTIME_TO_TRAIN = {v: k for k, v in _COLUMN_MAP_TRAIN_TO_RUNTIME.items()}

# Features that exist in training but have no runtime equivalent
_TRAIN_ONLY = {"signal", "rolling_mean_volume", "roll_window_flag", "days_to_expiry"}

# Features that exist in runtime but have no training equivalent
_RUNTIME_ONLY = {"sigma_co", "z_co", "r_co_winsor", "z_co_winsor",
                 "r_oc_mean_l", "r_co_mean_l", "feature_timestamp_end"}


# ---------------------------------------------------------------------------
# Parity computation
# ---------------------------------------------------------------------------

def compute_feature_parity(
    *,
    gold_frame: pd.DataFrame,
    sleeve: Any,  # COOCReversalFuturesSleeve
    asof_days: list[date],
    config: Optional[dict] = None,
    float_abs_tol: float = 1e-10,
    float_rel_tol: float = 1e-8,
    max_mismatch_frac: float = 0.001,
    output_dir: Optional[str | Path] = None,
) -> FeatureParityReport:
    """Compare training features vs runtime features for sample days.

    Parameters
    ----------
    gold_frame
        The gold-level panel used by the training pipeline.
    sleeve
        An instance of ``COOCReversalFuturesSleeve``.
    asof_days
        Which trading days to sample for comparison.
    config
        Optional config dict.
    float_abs_tol
        Absolute tolerance for numeric comparison.
    float_rel_tol
        Relative tolerance for numeric comparison.
    max_mismatch_frac
        Max fraction of mismatches to still pass the gate.
    output_dir
        If provided, persist artifacts here.

    Returns
    -------
    FeatureParityReport
    """
    from .dataset import build_features

    # --- Training path: build features from gold frame ---
    train_features = build_features(gold_frame)

    # Ensure trading_day is a date
    if hasattr(train_features["trading_day"].iloc[0], "date"):
        train_features["_td"] = train_features["trading_day"].apply(
            lambda x: x.date() if hasattr(x, "date") else x
        )
    else:
        train_features["_td"] = train_features["trading_day"]

    per_feature_mismatches: Dict[str, int] = {}
    per_feature_total: Dict[str, int] = {}
    worst_offenders: List[Dict[str, Any]] = []

    for day in asof_days:
        # --- Training path slice ---
        train_day = train_features[train_features["_td"] == day].copy()
        if train_day.empty:
            logger.debug("No training features for %s — skipping", day)
            continue

        # --- Runtime path ---
        from datetime import datetime as dt
        decision_ts = pd.Timestamp(dt.combine(day, time(9, 30), tzinfo=ET))
        try:
            runtime_day = sleeve.compute_signal_frame(gold_frame, decision_ts)
        except Exception as exc:
            logger.warning("Runtime path failed for %s: %s", day, exc)
            continue

        # Filter runtime to matching day
        if "date" in runtime_day.columns:
            rd_key = "date"
        elif "trading_day" in runtime_day.columns:
            rd_key = "trading_day"
        else:
            logger.debug("Cannot find date column in runtime output for %s", day)
            continue

        if hasattr(runtime_day[rd_key].iloc[0], "date"):
            runtime_day = runtime_day[runtime_day[rd_key].apply(
                lambda x: x.date() if hasattr(x, "date") else x
            ) == day]
        else:
            runtime_day = runtime_day[runtime_day[rd_key] == day]

        if runtime_day.empty:
            continue

        # --- Align by root/instrument ---
        train_key = "root"
        runtime_key = "instrument" if "instrument" in runtime_day.columns else "root"

        for _, train_row in train_day.iterrows():
            root = train_row[train_key]
            rt_match = runtime_day[runtime_day[runtime_key] == root]
            if rt_match.empty:
                continue

            rt_row = rt_match.iloc[0]

            # Compare mapped features
            for train_col, runtime_col in _COLUMN_MAP_TRAIN_TO_RUNTIME.items():
                if train_col not in train_day.columns:
                    continue
                if runtime_col not in runtime_day.columns:
                    continue

                tv = train_row[train_col]
                rv = rt_row[runtime_col]

                per_feature_total[train_col] = per_feature_total.get(train_col, 0) + 1

                if pd.isna(tv) and pd.isna(rv):
                    continue  # Both NaN → match

                if pd.isna(tv) != pd.isna(rv):
                    per_feature_mismatches[train_col] = per_feature_mismatches.get(train_col, 0) + 1
                    worst_offenders.append({
                        "feature_train": train_col,
                        "feature_runtime": runtime_col,
                        "day": str(day),
                        "root": root,
                        "train_value": float(tv) if not pd.isna(tv) else None,
                        "runtime_value": float(rv) if not pd.isna(rv) else None,
                        "reason": "NaN mismatch",
                    })
                    continue

                # Numeric comparison
                diff = abs(float(tv) - float(rv))
                if diff > float_abs_tol and diff > float_rel_tol * max(abs(float(tv)), abs(float(rv)), 1e-15):
                    per_feature_mismatches[train_col] = per_feature_mismatches.get(train_col, 0) + 1
                    worst_offenders.append({
                        "feature_train": train_col,
                        "feature_runtime": runtime_col,
                        "day": str(day),
                        "root": root,
                        "train_value": float(tv),
                        "runtime_value": float(rv),
                        "abs_diff": diff,
                        "reason": "numerical mismatch",
                    })

    # --- Compute per-feature mismatch rates ---
    mismatch_rates: Dict[str, float] = {}
    for feat in sorted(set(list(per_feature_total.keys()) + list(per_feature_mismatches.keys()))):
        total = per_feature_total.get(feat, 0)
        mismatches = per_feature_mismatches.get(feat, 0)
        mismatch_rates[feat] = mismatches / max(total, 1)

    # Also add unmapped features as 100% mismatch
    for feat in _TRAIN_ONLY:
        mismatch_rates[f"{feat} (train_only)"] = 1.0
    for feat in _RUNTIME_ONLY:
        mismatch_rates[f"{feat} (runtime_only)"] = 1.0

    # --- Gate ---
    mapped_rate = sum(per_feature_mismatches.values()) / max(sum(per_feature_total.values()), 1)
    gate_passed = mapped_rate <= max_mismatch_frac

    # Sort worst offenders by abs_diff descending
    worst_offenders.sort(key=lambda x: x.get("abs_diff", 0), reverse=True)
    worst_top = worst_offenders[:20]

    # --- Persist ---
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        report_data = {
            "per_feature_mismatch_rate": mismatch_rates,
            "mapped_mismatch_rate": mapped_rate,
            "gate_passed": gate_passed,
            "worst_offenders": worst_top,
            "train_only_features": sorted(_TRAIN_ONLY),
            "runtime_only_features": sorted(_RUNTIME_ONLY),
            "column_map": _COLUMN_MAP_TRAIN_TO_RUNTIME,
        }
        (out / "feature_parity_report.json").write_text(
            json.dumps(report_data, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )

        if worst_offenders:
            pd.DataFrame(worst_offenders).to_parquet(
                out / "feature_parity_examples.parquet", index=False
            )

    return FeatureParityReport(
        per_feature_mismatch_rate=mismatch_rates,
        worst_offenders=tuple(worst_top),
        gate_passed=gate_passed,
    )
