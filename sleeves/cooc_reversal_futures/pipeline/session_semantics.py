"""Session semantics validator: compare daily bars from two providers.

Quantifies how different yfinance continuous-contract bars are from IBKR
RTH bars at the open/close/ret_co/ret_oc level, in basis points.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .types import SessionSemanticsReport

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_FIELDS = ("open", "close", "ret_co", "ret_oc")
_DEFAULT_THRESHOLDS = {
    "open_close_median_bps_max": 5.0,
    "open_close_p95_bps_max": 25.0,
    "frac_days_over_25bps_max": 0.05,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bps_diff(a: pd.Series, b: pd.Series,
              reference: Optional[pd.Series] = None) -> pd.Series:
    """Absolute difference in basis points.

    For price fields, ``reference`` should be the midpoint.
    For return fields, the diff is already in return space → multiply by 1e4.
    """
    if reference is not None:
        return (a - b).abs() / reference.abs().replace(0, np.nan) * 1e4
    return (a - b).abs() * 1e4


def _compute_field_stats(
    diff_bps: pd.Series,
    threshold_bps: float = 25.0,
) -> Dict[str, float]:
    """Compute summary statistics for a series of bps differences."""
    valid = diff_bps.dropna()
    if len(valid) == 0:
        return {
            "median_bps": float("nan"),
            "p95_bps": float("nan"),
            "max_bps": float("nan"),
            "frac_above_10bps": float("nan"),
            "frac_above_25bps": float("nan"),
            "n_samples": 0,
        }
    return {
        "median_bps": float(valid.median()),
        "p95_bps": float(valid.quantile(0.95)),
        "max_bps": float(valid.max()),
        "frac_above_10bps": float((valid > 10.0).mean()),
        "frac_above_25bps": float((valid > threshold_bps).mean()),
        "n_samples": int(len(valid)),
    }


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------

def compare_session_semantics(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    roots: list[str],
    sample_days: int = 60,
    seed: int = 42,
    thresholds: Optional[Dict[str, float]] = None,
    output_dir: Optional[str | Path] = None,
    *,
    correlation_threshold: float = 0.90,
) -> SessionSemanticsReport:
    """Compare daily bars from provider A and B across overlapping days.

    Parameters
    ----------
    df_a
        Provider A bars (e.g. yfinance).  Must have columns:
        ``root, trading_day, open, close, ret_co, ret_oc``.
    df_b
        Provider B bars (e.g. IBKR RTH).  Same schema.
    roots
        List of roots to compare.
    sample_days
        Max overlapping days to sample per root.
    seed
        RNG seed for reproducible sampling.
    thresholds
        Gate thresholds (keys from ``_DEFAULT_THRESHOLDS``).
    output_dir
        If provided, persist artifacts here.
    correlation_threshold
        Minimum acceptable baseline proxy return correlation per root.

    Returns
    -------
    SessionSemanticsReport
    """
    thresholds = {**_DEFAULT_THRESHOLDS, **(thresholds or {})}
    rng = np.random.default_rng(seed)

    all_stats: Dict[str, Dict[str, float]] = {}
    sample_rows: List[pd.DataFrame] = []

    for root in sorted(roots):
        a = df_a[df_a["root"] == root].copy() if "root" in df_a.columns else df_a.copy()
        b = df_b[df_b["root"] == root].copy() if "root" in df_b.columns else df_b.copy()

        # Identify join key
        join_col = "trading_day" if "trading_day" in a.columns else "timestamp"
        if join_col == "timestamp":
            a[join_col] = pd.to_datetime(a[join_col]).dt.date
            b[join_col] = pd.to_datetime(b[join_col]).dt.date

        merged = a.merge(b, on=join_col, suffixes=("_a", "_b"), how="inner")
        if merged.empty:
            logger.warning("No overlapping days for root %s — skipping", root)
            continue

        # Sample
        if len(merged) > sample_days:
            idx = rng.choice(len(merged), sample_days, replace=False)
            merged = merged.iloc[sorted(idx)].reset_index(drop=True)

        # Compute bps diffs
        for field in _DEFAULT_FIELDS:
            col_a = f"{field}_a"
            col_b = f"{field}_b"
            if col_a not in merged.columns or col_b not in merged.columns:
                continue

            if field in ("open", "close"):
                ref = (merged[col_a] + merged[col_b]) / 2
                diff = _bps_diff(merged[col_a], merged[col_b], reference=ref)
            else:
                diff = _bps_diff(merged[col_a], merged[col_b])

            key = f"{root}_{field}"
            all_stats[key] = _compute_field_stats(diff)

        merged["root"] = root
        sample_rows.append(merged)

    # --- Gate evaluation ---
    gate_passed = True

    # Check open/close median and p95 across all roots
    for field in ("open", "close"):
        for root in sorted(roots):
            key = f"{root}_{field}"
            if key not in all_stats:
                continue
            stats = all_stats[key]
            if stats["median_bps"] > thresholds["open_close_median_bps_max"]:
                gate_passed = False
                logger.warning(
                    "Session semantics FAIL: %s median_bps=%.1f > %.1f",
                    key, stats["median_bps"], thresholds["open_close_median_bps_max"],
                )
            if stats["p95_bps"] > thresholds["open_close_p95_bps_max"]:
                gate_passed = False
                logger.warning(
                    "Session semantics FAIL: %s p95_bps=%.1f > %.1f",
                    key, stats["p95_bps"], thresholds["open_close_p95_bps_max"],
                )
            if stats["frac_above_25bps"] > thresholds["frac_days_over_25bps_max"]:
                gate_passed = False

    # --- Persist ---
    sample_rows_path = ""
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        (out / "session_semantics_report.json").write_text(
            json.dumps({"per_field_stats": all_stats, "gate_passed": gate_passed},
                       indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )

        if sample_rows:
            combined = pd.concat(sample_rows, ignore_index=True)
            sample_rows_path = str(out / "session_semantics_samples.parquet")
            combined.to_parquet(sample_rows_path, index=False)

    return SessionSemanticsReport(
        per_field_stats=all_stats,
        gate_passed=gate_passed,
        sample_rows_path=sample_rows_path,
    )


# ---------------------------------------------------------------------------
# R1: Provider invariance helpers
# ---------------------------------------------------------------------------

def _compute_baseline_proxy_correlation(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    roots: list[str],
) -> Dict[str, float]:
    """Compute daily baseline proxy return correlation per root.

    Baseline proxy return = -r_co (mean-revert alpha).
    Correlation is Pearson on overlapping days.
    """
    corr_by_root: Dict[str, float] = {}
    for root in sorted(roots):
        a = df_a[df_a["root"] == root] if "root" in df_a.columns else df_a
        b = df_b[df_b["root"] == root] if "root" in df_b.columns else df_b

        join_col = "trading_day" if "trading_day" in a.columns else "timestamp"

        r_co_col_a = "ret_co" if "ret_co" in a.columns else "r_co"
        r_co_col_b = "ret_co" if "ret_co" in b.columns else "r_co"

        if r_co_col_a not in a.columns or r_co_col_b not in b.columns:
            continue

        merged = a[[join_col, r_co_col_a]].merge(
            b[[join_col, r_co_col_b]],
            on=join_col,
            suffixes=("_a", "_b"),
            how="inner",
        )
        if len(merged) < 5:
            corr_by_root[root] = float("nan")
            continue

        col_a = f"{r_co_col_a}_a" if f"{r_co_col_a}_a" in merged.columns else r_co_col_a
        col_b = f"{r_co_col_b}_b" if f"{r_co_col_b}_b" in merged.columns else r_co_col_b

        c = merged[col_a].corr(merged[col_b])
        corr_by_root[root] = float(c) if np.isfinite(c) else float("nan")

    return corr_by_root


def _compute_r_co_quantiles(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    roots: list[str],
    quantiles: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90),
) -> Dict[str, Dict[str, float]]:
    """Compute r_co quantile comparison across providers per root."""
    result: Dict[str, Dict[str, float]] = {}
    for root in sorted(roots):
        a = df_a[df_a["root"] == root] if "root" in df_a.columns else df_a
        b = df_b[df_b["root"] == root] if "root" in df_b.columns else df_b

        r_co_a = a.get("ret_co", a.get("r_co"))
        r_co_b = b.get("ret_co", b.get("r_co"))

        if r_co_a is None or r_co_b is None:
            continue

        entry: Dict[str, float] = {}
        for q in quantiles:
            entry[f"q{int(q * 100):02d}_a"] = float(r_co_a.quantile(q))
            entry[f"q{int(q * 100):02d}_b"] = float(r_co_b.quantile(q))
        result[root] = entry
    return result


def _compute_missing_open_close(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    roots: list[str],
) -> Dict[str, Dict[str, int]]:
    """Count missing open/close per root per provider."""
    result: Dict[str, Dict[str, int]] = {}
    for root in sorted(roots):
        a = df_a[df_a["root"] == root] if "root" in df_a.columns else df_a
        b = df_b[df_b["root"] == root] if "root" in df_b.columns else df_b

        miss_a = 0
        miss_b = 0
        for col in ("open", "close"):
            if col in a.columns:
                miss_a += int(a[col].isna().sum())
            if col in b.columns:
                miss_b += int(b[col].isna().sum())

        result[root] = {"provider_a_missing": miss_a, "provider_b_missing": miss_b}
    return result


def build_provider_invariance_report(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    roots: list[str],
    *,
    sample_days: int = 60,
    seed: int = 42,
    thresholds: Optional[Dict[str, float]] = None,
    correlation_threshold: float = 0.90,
    output_dir: Optional[str | Path] = None,
) -> "ProviderInvarianceReport":
    """Build a full provider invariance report.

    Combines session semantics comparison, baseline proxy correlation,
    r_co quantile comparison, and missing data counts.
    """
    from .types import ProviderInvarianceReport

    ss_report = compare_session_semantics(
        df_a, df_b, roots,
        sample_days=sample_days, seed=seed,
        thresholds=thresholds, output_dir=output_dir,
        correlation_threshold=correlation_threshold,
    )

    corr = _compute_baseline_proxy_correlation(df_a, df_b, roots)
    quantiles = _compute_r_co_quantiles(df_a, df_b, roots)
    missing = _compute_missing_open_close(df_a, df_b, roots)

    flags: list[str] = []

    # Session semantics gate
    if not ss_report.gate_passed:
        flags.append("session_semantics_gate_failed")

    # Correlation gate
    for root, c in corr.items():
        if np.isnan(c):
            flags.append(f"{root}: insufficient data for correlation")
        elif c < correlation_threshold:
            flags.append(f"{root}: baseline proxy corr={c:.3f} < {correlation_threshold}")

    overall = len(flags) == 0

    # Persist extended report
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        import json as _json
        (out / "provider_invariance_report.json").write_text(
            _json.dumps({
                "baseline_proxy_correlation": corr,
                "r_co_quantile_comparison": quantiles,
                "missing_open_close_counts": missing,
                "overall_consistent": overall,
                "flags": flags,
            }, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )

    return ProviderInvarianceReport(
        session_semantics=ss_report,
        baseline_proxy_correlation=corr,
        r_co_quantile_comparison=quantiles,
        missing_open_close_counts=missing,
        overall_consistent=overall,
        flags=tuple(flags),
    )

