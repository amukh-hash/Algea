"""Feature/label engineering and dataset assembly with leakage guards.

CHANGE LOG (2026-02-14):
  - D3: Added FeatureTimestampGuard with strict/non-strict modes.
  - D3: assemble_dataset now accepts session_open_ts_col and leakage_strict
    parameters to run the guard and emit leakage_report.json.
"""
from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, time, timezone
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ..features_core import FeatureConfig, active_schema, compute_core_features
from ..roll import roll_week_flag
from .types import DatasetManifest

ET = ZoneInfo("America/New_York")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FEATURE_COLUMNS = (
    "r_co",
    "r_co_cs_demean",
    "r_co_rank_pct",
    "sigma_co",
    "sigma_oc_hist",
    "volume_z",
    "roll_window_flag",
    "days_to_expiry",
    "day_of_week",
)

_PROVENANCE_COLUMNS = (
    "asof_ts",
    "feature_cutoff_ts",
    "decision_ts",
    "label_ts",
    "data_version_hash",
    "code_version_hash",
    "config_hash",
)


def _stable_hash(obj: Any) -> str:
    """Deterministic SHA-256 of a JSON-serializable object."""
    raw = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

def build_features(
    gold_frame: pd.DataFrame,
    lookback: int = 20,
) -> pd.DataFrame:
    """Compute per-instrument and cross-sectional features from gold frame.

    All rolling operations are strictly causal (look-back only).
    Uses canonical column names: ``instrument``, ``r_co``, ``r_oc``.

    Features
    --------
    Per-instrument:
      - r_co                (raw overnight return — kept as feature)
      - sigma_co            (rolling std of r_co)
      - sigma_oc_hist       (rolling std of r_oc, backward-looking)
      - volume_z            (z-score of volume vs rolling mean/std)
      - roll_window_flag
      - days_to_expiry
      - day_of_week

    Cross-sectional (computed per trading_day):
      - r_co_rank_pct
      - r_co_cs_demean
    """
    df = gold_frame.copy()

    # Ensure canonical columns exist
    if "instrument" not in df.columns and "root" in df.columns:
        df["instrument"] = df["root"]
    if "r_co" not in df.columns and "ret_co" in df.columns:
        df["r_co"] = df["ret_co"]
    if "r_oc" not in df.columns and "ret_oc" in df.columns:
        df["r_oc"] = df["ret_oc"]

    df = df.sort_values(["instrument", "trading_day"]).reset_index(drop=True)

    # --- Per-instrument rolling features ---
    df["sigma_co"] = (
        df.groupby("instrument")["r_co"]
        .transform(lambda s: s.rolling(lookback, min_periods=3).std())
    )
    df["sigma_oc_hist"] = (
        df.groupby("instrument")["r_oc"]
        .transform(lambda s: s.rolling(lookback, min_periods=3).std())
    )
    df["sigma_oc"] = df["sigma_oc_hist"]  # backward-compat alias

    # Volume z-score
    vol_mean = (
        df.groupby("instrument")["volume"]
        .transform(lambda s: s.rolling(lookback, min_periods=3).mean())
    )
    vol_std = (
        df.groupby("instrument")["volume"]
        .transform(lambda s: s.rolling(lookback, min_periods=3).std())
    )
    df["volume_z"] = (df["volume"] - vol_mean) / vol_std.replace(0, np.nan)

    # Roll window flag per row
    df["roll_window_flag"] = df["trading_day"].apply(
        lambda d: int(roll_week_flag(d))
    )

    # days_to_expiry already present from silver/gold merge
    if "days_to_expiry" not in df.columns:
        df["days_to_expiry"] = 0

    # Calendar feature
    df["day_of_week"] = pd.to_datetime(df["trading_day"]).dt.dayofweek

    # --- Cross-sectional features (per trading_day) ---
    df["r_co_rank_pct"] = df.groupby("trading_day")["r_co"].rank(pct=True)
    df["r_co_cs_demean"] = (
        df["r_co"] - df.groupby("trading_day")["r_co"].transform("mean")
    )

    # Legacy aliases for backward compatibility
    df["signal"] = -df["r_co"]
    df["rolling_std_ret_co"] = df["sigma_co"]
    df["rolling_std_ret_oc"] = df["sigma_oc_hist"]
    df["rolling_mean_volume"] = vol_mean
    df["ret_co_rank_pct"] = df["r_co_rank_pct"]
    df["ret_co_cs_demean"] = df["r_co_cs_demean"]

    return df


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

def build_labels(gold_frame: pd.DataFrame) -> pd.DataFrame:
    """Build label column: ``y = -r_oc`` (reversal score target).

    Score semantics: higher score = more bearish intraday expectation.
    This aligns with the trade proxy which longs lowest-scored and
    shorts highest-scored instruments.

    The label uses close[D] which is strictly future of features
    (features use only data available before open[D]).
    """
    cols = ["trading_day"]
    # Use canonical name if available, fall back to legacy
    inst_col = "instrument" if "instrument" in gold_frame.columns else "root"
    cols.append(inst_col)
    r_oc_col = "r_oc" if "r_oc" in gold_frame.columns else "ret_oc"
    cols.append(r_oc_col)

    df = gold_frame[cols].copy()
    df["y"] = -df[r_oc_col]      # reversal target: y = -r_oc
    if inst_col == "root":
        df["instrument"] = df["root"]
    df = df.drop(columns=[r_oc_col])
    return df


# ---------------------------------------------------------------------------
# Provenance timestamps
# ---------------------------------------------------------------------------

def _trading_day_to_decision_ts(trading_day: date) -> datetime:
    """Decision time = NY open of trading_day (09:30 ET)."""
    return datetime.combine(trading_day, time(9, 30), tzinfo=ET)


def _trading_day_to_label_ts(trading_day: date) -> datetime:
    """Label time = NY close of trading_day (16:00 ET)."""
    return datetime.combine(trading_day, time(16, 0), tzinfo=ET)


def _trading_day_to_feature_cutoff_ts(trading_day: date) -> datetime:
    """Feature cutoff = just before NY open (09:29:59 ET).

    Features are computed from data available before market open.
    """
    return datetime.combine(trading_day, time(9, 29, 59), tzinfo=ET)


# ---------------------------------------------------------------------------
# Leakage assertions
# ---------------------------------------------------------------------------

def assert_no_leakage(df: pd.DataFrame) -> None:
    """Hard-fail if any row violates leakage invariants.

    For every row:
      - feature_cutoff_ts <= decision_ts
      - label_ts > decision_ts
    """
    violations_feat = df["feature_cutoff_ts"] > df["decision_ts"]
    if violations_feat.any():
        n = int(violations_feat.sum())
        raise AssertionError(
            f"LEAKAGE: {n} rows have feature_cutoff_ts > decision_ts"
        )

    violations_label = df["label_ts"] <= df["decision_ts"]
    if violations_label.any():
        n = int(violations_label.sum())
        raise AssertionError(
            f"LEAKAGE: {n} rows have label_ts <= decision_ts"
        )


# ---------------------------------------------------------------------------
# D3: Feature Timestamp Guard  (H1: risky set derived from FeatureSpec)
# ---------------------------------------------------------------------------

try:
    from backend.app.features.feature_spec import (
        FEATURE_SPECS as _FSPECS,
        risky_features_for_preopen as _risky_features_for_preopen,
    )
except ImportError:  # pragma: no cover
    _FSPECS: Dict[str, Any] = {}  # type: ignore[no-redef]
    def _risky_features_for_preopen() -> frozenset:  # type: ignore[misc]
        return frozenset({"r_oc", "ret_oc"})  # fallback


@dataclass
class LeakageReport:
    """Structured result from FeatureTimestampGuard."""
    kept_features: List[str] = field(default_factory=list)
    dropped_features: List[str] = field(default_factory=list)
    missing_features: List[str] = field(default_factory=list)
    # H2: tracking how features were guarded
    guarded_by_global: List[str] = field(default_factory=list)
    guarded_by_per_feature: List[str] = field(default_factory=list)
    unguarded_features: List[str] = field(default_factory=list)
    required_dropped: bool = False
    n_rows_checked: int = 0
    n_violations: int = 0
    examples: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def run_feature_timestamp_guard(
    df: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    session_open_ts_col: Optional[str] = "session_open_ts",
    safety_margin_secs: int = 0,
    strict: bool = False,
    output_dir: Optional[str | Path] = None,
) -> Tuple[pd.DataFrame, LeakageReport]:
    """Check that feature timestamps do not leak future information.

    Uses FeatureSpec as the single source of truth for risky features.
    Supports both global and per-feature cutoff timestamps.

    Parameters
    ----------
    df : dataset DataFrame
    feature_columns : columns being used as model features
    session_open_ts_col : column with session open timestamp (if available)
    safety_margin_secs : safety margin in seconds before session open
    strict : if True, raise ValueError on violations; else drop + log
    output_dir : if provided, write leakage_report.json here

    Returns
    -------
    (cleaned DataFrame, LeakageReport)
    """
    import logging
    logger = logging.getLogger(__name__)

    report = LeakageReport(n_rows_checked=len(df))
    risky_set = _risky_features_for_preopen()

    # Track missing features (requested but not in DataFrame)
    report.missing_features = [f for f in feature_columns if f not in df.columns]

    # --- H1: Check for risky features being used as inputs ---
    risky_used = [f for f in feature_columns if f in risky_set and f in df.columns]
    if risky_used:
        msg = (
            f"FeatureTimestampGuard: risky features used as model inputs: "
            f"{risky_used}. These contain future information (same-day)."
        )
        report.n_violations += len(risky_used)
        # Check if any are required (non-optional)
        for feat in risky_used:
            spec = _FSPECS.get(feat)
            if spec and not spec.optional:
                report.required_dropped = True
        if strict:
            raise ValueError(msg)
        else:
            logger.warning(msg + " Dropping them.")
            report.dropped_features.extend(risky_used)

    # --- Global timestamp guard ---
    non_risky = [f for f in feature_columns
                 if f not in risky_set and f not in report.missing_features]
    has_session_ts = session_open_ts_col and session_open_ts_col in df.columns

    if not has_session_ts:
        report.unguarded_features = list(non_risky)
        logger.info(
            "FeatureTimestampGuard: session_open_ts not available. "
            "%d features are unguarded (no timestamp check possible).",
            len(non_risky),
        )
    else:
        from datetime import timedelta
        margin = timedelta(seconds=safety_margin_secs)
        session_open = pd.to_datetime(df[session_open_ts_col])

        # Global cutoff check
        if "feature_cutoff_ts" in df.columns:
            cutoff = pd.to_datetime(df["feature_cutoff_ts"])
            violations = cutoff > (session_open - margin)
            n_v = int(violations.sum())
            if n_v > 0:
                report.n_violations += n_v
                bad_rows = df[violations].head(10)
                for _, row in bad_rows.iterrows():
                    report.examples.append({
                        "root": str(row.get("root", row.get("instrument", "?"))),
                        "trading_day": str(row.get("trading_day", "?")),
                        "feature_cutoff_ts": str(row.get("feature_cutoff_ts", "?")),
                        "session_open_ts": str(row.get(session_open_ts_col, "?")),
                    })
                if strict:
                    raise ValueError(
                        f"FeatureTimestampGuard: {n_v} rows have "
                        f"feature_cutoff_ts > session_open_ts - {safety_margin_secs}s"
                    )
                else:
                    logger.warning(
                        "FeatureTimestampGuard: %d rows violate global timestamp guard.",
                        n_v,
                    )
            report.guarded_by_global = list(non_risky)
        else:
            report.unguarded_features = list(non_risky)

        # --- H2: Per-feature cutoff check ---
        per_feat_cols = {
            col.replace("feature_cutoff_ts__", ""): col
            for col in df.columns
            if col.startswith("feature_cutoff_ts__")
        }
        for feat, ts_col in per_feat_cols.items():
            if feat not in feature_columns or feat in report.dropped_features:
                continue
            feat_cutoff = pd.to_datetime(df[ts_col])
            violations = feat_cutoff > (session_open - margin)
            n_v = int(violations.sum())
            if n_v > 0:
                report.n_violations += n_v
                spec = _FSPECS.get(feat)
                feat_optional = (spec.optional if spec else True)

                if strict and not feat_optional:
                    raise ValueError(
                        f"FeatureTimestampGuard: required feature '{feat}' has "
                        f"{n_v} per-feature timestamp violations."
                    )
                elif not feat_optional:
                    report.required_dropped = True

                logger.warning(
                    "FeatureTimestampGuard: per-feature cutoff violation for '%s' "
                    "(%d rows). %s.",
                    feat, n_v,
                    "Dropping" if feat_optional else "REQUIRED — will fail",
                )
                if feat not in report.dropped_features:
                    report.dropped_features.append(feat)

            # Move from unguarded to per-feature if present
            if feat in report.unguarded_features:
                report.unguarded_features.remove(feat)
            if feat in report.guarded_by_global:
                report.guarded_by_global.remove(feat)
            if feat not in report.guarded_by_per_feature:
                report.guarded_by_per_feature.append(feat)

    # Drop risky/violating features from the DataFrame
    out = df.copy()
    cols_to_drop = [c for c in report.dropped_features if c in out.columns]
    if cols_to_drop:
        out = out.drop(columns=cols_to_drop)

    # Compute kept features
    dropped_set = set(report.dropped_features) | set(report.missing_features)
    report.kept_features = [f for f in feature_columns if f not in dropped_set]

    # Hard-fail if any required preopen feature was dropped
    if report.required_dropped:
        req_dropped = [
            f for f in report.dropped_features
            if _FSPECS.get(f) and not _FSPECS[f].optional
        ]
        raise ValueError(
            f"FeatureTimestampGuard: required preopen features dropped: "
            f"{req_dropped}. Cannot proceed — fix feature computation."
        )

    # Persist report
    if output_dir is not None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        report_path = out_path / "leakage_report.json"
        report_path.write_text(
            json.dumps(report.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )

    return out, report


# ---------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------

def assemble_dataset(
    gold_frame: pd.DataFrame,
    config: Any,
    lookback: int = 20,
    code_version: str = "1.0.0",
    output_dir: Optional[str | Path] = None,
    *,
    session_open_ts_col: Optional[str] = None,
    leakage_strict: bool = False,
) -> Tuple[pd.DataFrame, DatasetManifest]:
    """Assemble the full training dataset with features, labels, and provenance.

    Parameters
    ----------
    gold_frame : output of :func:`canonicalize.build_gold_frame`
    config : sleeve config (for hashing)
    lookback : rolling window size
    code_version : code version tag for provenance
    output_dir : if provided, persist dataset.parquet here

    Returns
    -------
    (dataset DataFrame, DatasetManifest)
    """
    # Use schema-aware feature computation from features_core.py
    schema_version = getattr(config, 'schema_version', 2)
    feat_cfg = FeatureConfig(schema_version=schema_version, lookback=lookback)
    features_df = compute_core_features(gold_frame, feat_cfg)
    labels_df = build_labels(gold_frame)

    # Merge features + labels
    merge_cols = ["trading_day"]
    if "instrument" in features_df.columns and "instrument" in labels_df.columns:
        merge_cols.append("instrument")
    elif "root" in features_df.columns and "root" in labels_df.columns:
        merge_cols.append("root")
    dataset = features_df.merge(
        labels_df, on=merge_cols, how="inner",
    )

    # Ensure canonical columns
    if "instrument" not in dataset.columns and "root" in dataset.columns:
        dataset["instrument"] = dataset["root"]
    if "r_co" not in dataset.columns and "ret_co" in dataset.columns:
        dataset["r_co"] = dataset["ret_co"]
    if "r_oc" not in dataset.columns and "ret_oc" in dataset.columns:
        dataset["r_oc"] = dataset["ret_oc"]

    # --- Provenance columns ---
    now_utc = datetime.now(timezone.utc).isoformat()

    dataset["asof_ts"] = now_utc
    dataset["feature_cutoff_ts"] = dataset["trading_day"].apply(
        _trading_day_to_feature_cutoff_ts
    )
    dataset["decision_ts"] = dataset["trading_day"].apply(
        _trading_day_to_decision_ts
    )
    dataset["label_ts"] = dataset["trading_day"].apply(
        _trading_day_to_label_ts
    )

    # Hashes (use canonical columns, fall back to legacy)
    hash_cols = ["trading_day", "instrument", "r_co", "r_oc"]
    actual_hash_cols = [c for c in hash_cols if c in dataset.columns]
    data_hash = _stable_hash(
        dataset[actual_hash_cols].to_dict(orient="list")
    )
    code_hash = _stable_hash({"code_version": code_version})
    config_hash = _stable_hash(
        {k: str(v) for k, v in (config.__dict__ if hasattr(config, "__dict__") else {"v": str(config)}).items()}
    )

    dataset["data_version_hash"] = data_hash
    dataset["code_version_hash"] = code_hash
    dataset["config_hash"] = config_hash

    # --- Leakage assertion ---
    assert_no_leakage(dataset)

    # --- D3: Feature timestamp guard ---
    leakage_dir = (Path(output_dir) / "data_audit") if output_dir else None
    dataset, _leakage_report = run_feature_timestamp_guard(
        dataset,
        feature_columns=list(_FEATURE_COLUMNS),
        session_open_ts_col=session_open_ts_col,
        strict=leakage_strict,
        output_dir=leakage_dir,
    )

    # --- Sort deterministically ---
    sort_col = "instrument" if "instrument" in dataset.columns else "root"
    dataset = dataset.sort_values([sort_col, "trading_day"]).reset_index(drop=True)

    # --- Set index (canonical: trading_day, instrument) ---
    idx_col = "instrument" if "instrument" in dataset.columns else "root"
    dataset.index = pd.MultiIndex.from_arrays(
        [dataset["trading_day"], dataset[idx_col]],
        names=["trading_day", "instrument"],
    )

    # --- Persist ---
    dataset_path = ""
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        dataset_path = str(out / "dataset.parquet")
        dataset.to_parquet(dataset_path)

    # Use active schema for feature column list, filtered to kept features
    active_features = tuple(active_schema(schema_version))
    kept_features = tuple(
        f for f in active_features if f in _leakage_report.kept_features
    ) if _leakage_report.kept_features else active_features
    dropped_optional = tuple(_leakage_report.dropped_features)

    manifest = DatasetManifest(
        dataset_path=dataset_path,
        row_count=len(dataset),
        feature_columns=kept_features,
        label_column="y",
        provenance_columns=_PROVENANCE_COLUMNS,
        data_version_hash=data_hash,
        code_version_hash=code_hash,
        config_hash=config_hash,
        dropped_features=dropped_optional,
    )

    return dataset, manifest
