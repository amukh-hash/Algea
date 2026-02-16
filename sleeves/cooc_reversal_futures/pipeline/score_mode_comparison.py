"""Score-mode comparison for promotion evaluation.

Evaluates three scoring modes on the same validation windows:

1. **Baseline**: ``score = r_co`` (raw overnight return, no model).
2. **Model raw**: ``score = score_pred`` (model's raw score output).
3. **Model risk-adjusted**: ``score = derived_score`` (stabilized score / risk).

Each mode is run through the trade proxy to produce daily PnL.
Output: ``score_mode_comparison.json`` + per-mode daily PnL parquets.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModeResult:
    """Results for a single scoring mode."""
    mode: str
    sharpe: float
    mean_daily_ret: float
    hit_rate: float
    max_drawdown: float
    worst_1pct: float
    n_days: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ScoreModeComparisonReport:
    """Consolidated comparison across all scoring modes."""
    modes: tuple[ModeResult, ...] = ()
    best_mode: str = ""
    improvement_over_baseline: float = 0.0
    gate_passed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "modes": [m.to_dict() for m in self.modes],
            "best_mode": self.best_mode,
            "improvement_over_baseline": self.improvement_over_baseline,
            "gate_passed": self.gate_passed,
        }


# ---------------------------------------------------------------------------
# Core comparison
# ---------------------------------------------------------------------------

def compare_score_modes(
    dataset: pd.DataFrame,
    *,
    score_cols: Dict[str, str] | None = None,
    ret_col: str = "r_oc",
    top_k: int = 3,
    gross_target: float = 0.8,
    min_sharpe_delta: float = 0.1,
) -> ScoreModeComparisonReport:
    """Run trade proxy for each scoring mode and compare.

    Parameters
    ----------
    dataset
        Must contain ``trading_day``, ``instrument``, ``r_co``, ``r_oc``,
        and optionally ``score_pred``, ``derived_score``.
    score_cols
        Map from mode name → column name.  Defaults to standard modes.
    ret_col
        Intraday return column for PnL calculation.
    top_k
        Number of instruments long/short per day.
    gross_target
        Target gross exposure.
    min_sharpe_delta
        Minimum Sharpe improvement over baseline for gate to pass.
    """
    if score_cols is None:
        score_cols = {}
        if "r_co" in dataset.columns:
            score_cols["baseline"] = "r_co"
        if "score_pred" in dataset.columns:
            score_cols["model_raw"] = "score_pred"
        if "derived_score" in dataset.columns:
            score_cols["model_risk_adjusted"] = "derived_score"

    results: List[ModeResult] = []
    daily_pnl_frames: Dict[str, pd.DataFrame] = {}

    for mode_name, col in score_cols.items():
        if col not in dataset.columns:
            logger.warning("Score column '%s' missing for mode '%s' — skipping.", col, mode_name)
            continue

        daily_rets = _compute_daily_returns(dataset, col, ret_col, top_k, gross_target)
        daily_pnl_frames[mode_name] = daily_rets

        arr = daily_rets["daily_ret"].values
        sharpe = _annualized_sharpe(arr)
        mean_ret = float(np.mean(arr))
        hit_rate = float(np.mean(arr > 0)) if len(arr) > 0 else 0.0

        # Max drawdown
        cum = np.cumsum(arr)
        peak = np.maximum.accumulate(cum)
        dd = cum - peak
        max_dd = float(np.min(dd)) if len(dd) > 0 else 0.0

        # Worst 1% day
        worst_1pct = float(np.percentile(arr, 1)) if len(arr) > 20 else float(np.min(arr)) if len(arr) > 0 else 0.0

        results.append(ModeResult(
            mode=mode_name,
            sharpe=sharpe,
            mean_daily_ret=mean_ret,
            hit_rate=hit_rate,
            max_drawdown=max_dd,
            worst_1pct=worst_1pct,
            n_days=len(arr),
        ))

    if not results:
        return ScoreModeComparisonReport()

    # Find best mode by Sharpe
    results_tuple = tuple(results)
    best = max(results, key=lambda r: r.sharpe)
    baseline = next((r for r in results if r.mode == "baseline"), results[0])
    improvement = best.sharpe - baseline.sharpe

    gate_passed = improvement >= min_sharpe_delta

    return ScoreModeComparisonReport(
        modes=results_tuple,
        best_mode=best.mode,
        improvement_over_baseline=improvement,
        gate_passed=gate_passed,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_daily_returns(
    dataset: pd.DataFrame,
    score_col: str,
    ret_col: str,
    top_k: int,
    gross_target: float,
) -> pd.DataFrame:
    """Simple long/short proxy: long bottom-k by score, short top-k."""
    records = []
    for day, grp in dataset.groupby("trading_day"):
        if len(grp) < 2 * top_k:
            continue
        ranked = grp.sort_values(score_col)
        longs = ranked.head(top_k)
        shorts = ranked.tail(top_k)

        n_positions = 2 * top_k
        weight = gross_target / max(n_positions, 1)

        long_ret = longs[ret_col].mean() * weight * top_k
        short_ret = -shorts[ret_col].mean() * weight * top_k
        records.append({"trading_day": day, "daily_ret": long_ret + short_ret})

    return pd.DataFrame(records)


def _annualized_sharpe(daily_returns: np.ndarray, trading_days: int = 252) -> float:
    """Annualized Sharpe ratio from daily returns."""
    if len(daily_returns) < 5:
        return 0.0
    mean = np.mean(daily_returns)
    std = np.std(daily_returns, ddof=1)
    if std < 1e-12:
        return 0.0
    return float(mean / std * np.sqrt(trading_days))


def run_score_mode_comparison(
    dataset: pd.DataFrame,
    model: Any,
    preprocessor: Any,
    cv_splits: Any,
    config: Any,
) -> dict:
    """CLI-facing wrapper: run score mode comparison and return a dict.

    Parameters
    ----------
    dataset
        Full dataset with ``r_co``, ``r_oc``, ``trading_day``, ``instrument``.
    model
        Trained model with ``.predict()`` method.
    preprocessor
        Feature preprocessor with ``.transform()`` method.
    cv_splits
        Cross-validation splits (unused currently but kept for API compat).
    config
        Pipeline config (unused currently but kept for API compat).

    Returns
    -------
    dict
        Keys: ``best_mode``, ``improvement_over_baseline``, ``gate_passed``,
        ``modes`` list.
    """
    # Generate predictions and add to dataset — ensure clean column access
    ds = dataset.copy()
    # Reset any index levels to regular columns (handles MultiIndex too)
    if ds.index.name is not None or (hasattr(ds.index, "names") and any(n is not None for n in ds.index.names)):
        # Only reset index levels that are NOT already columns to avoid duplicates
        idx_names = [ds.index.name] if ds.index.name else list(ds.index.names)
        cols_to_keep = [n for n in idx_names if n is not None and n not in ds.columns]
        cols_to_drop = [n for n in idx_names if n is not None and n in ds.columns]
        if cols_to_drop:
            ds = ds.reset_index(drop=True)
            # Re-copy from original to preserve column data
            ds = dataset.reset_index(drop=True).copy()
        elif cols_to_keep:
            ds = ds.reset_index()
    try:
        X = preprocessor.transform(ds)
        preds = model.predict(X).flatten()
        ds["score_pred"] = preds[:len(ds)]
    except Exception as e:
        logger.warning("Could not generate predictions for score comparison: %s", e)
        return {"best_mode": "baseline", "gate_passed": False, "modes": []}

    report = compare_score_modes(ds)
    return report.to_dict()


def save_report(
    report: ScoreModeComparisonReport,
    path: Path,
    daily_pnl_dir: Optional[Path] = None,
) -> None:
    """Write comparison report to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    logger.info("Score mode comparison → %s (best=%s, passed=%s)", path, report.best_mode, report.gate_passed)
