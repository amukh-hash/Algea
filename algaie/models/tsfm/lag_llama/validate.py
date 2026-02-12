"""
Forecast validation — rolling coverage, stability metrics, health score.

Used to assess model quality and gate forecast usage.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from algaie.models.tsfm.lag_llama.config import LagLlamaConfig
from algaie.models.tsfm.lag_llama.inference import ForecastResult


@dataclass
class ValidationReport:
    """Quality report for a set of forecasts against realised outcomes."""
    coverage: Dict[float, float]       # {quantile: empirical coverage}
    health_score: float                # [0, 1]
    monotonicity_violations: int
    stability_mae: float               # mean absolute change between consecutive forecasts
    n_forecasts: int
    details: Dict[str, float] = field(default_factory=dict)


def compute_quantile_coverage(
    forecasts: List[ForecastResult],
    realised: pd.Series,
    quantile: float,
) -> float:
    """Compute empirical coverage for a given quantile.

    Coverage = fraction of times realised <= forecast quantile.
    For a well-calibrated p-quantile forecast, coverage ≈ p.
    """
    hits = 0
    total = 0
    for fc in forecasts:
        dt = fc.as_of_date
        if dt in realised.index:
            rv_actual = float(realised.loc[dt])
            rv_pred = fc.quantiles.get(quantile, np.nan)
            if not np.isnan(rv_pred):
                if rv_actual <= rv_pred:
                    hits += 1
                total += 1

    return hits / max(total, 1)


def compute_health_score(
    coverage: Dict[float, float],
    config: LagLlamaConfig,
) -> float:
    """Combine coverage into a single health score in [0, 1].

    Measures how close empirical coverage aligns to nominal quantiles.
    Perfect calibration → 1.0.
    """
    if not coverage:
        return 0.0

    errors = []
    for q, cov in coverage.items():
        # Ideal coverage: q
        errors.append(abs(cov - q))

    mean_error = float(np.mean(errors))
    # Map: error=0 → health=1, error=0.5 → health=0
    health = max(0.0, 1.0 - 2.0 * mean_error)
    return health


def validate_forecasts(
    forecasts: List[ForecastResult],
    realised_rv: pd.Series,
    config: LagLlamaConfig,
) -> ValidationReport:
    """Run full validation suite on a list of forecasts.

    Parameters
    ----------
    forecasts : ordered list of ForecastResult
    realised_rv : series of actual annualised RV (index = date str or date)
    config : LagLlamaConfig

    Returns
    -------
    ValidationReport with coverage, health score, stability.
    """
    # Coverage per quantile
    coverage: Dict[float, float] = {}
    for q in config.quantiles:
        coverage[q] = compute_quantile_coverage(forecasts, realised_rv, q)

    health = compute_health_score(coverage, config)

    # Monotonicity violations
    mono_violations = 0
    for fc in forecasts:
        sorted_qs = sorted(fc.quantiles.keys())
        for i in range(1, len(sorted_qs)):
            if fc.quantiles[sorted_qs[i]] < fc.quantiles[sorted_qs[i - 1]]:
                mono_violations += 1

    # Stability: mean absolute change in median forecast
    stability_diffs = []
    for i in range(1, len(forecasts)):
        prev_med = forecasts[i - 1].quantiles.get(0.50, 0.0)
        curr_med = forecasts[i].quantiles.get(0.50, 0.0)
        stability_diffs.append(abs(curr_med - prev_med))
    stability_mae = float(np.mean(stability_diffs)) if stability_diffs else 0.0

    return ValidationReport(
        coverage=coverage,
        health_score=health,
        monotonicity_violations=mono_violations,
        stability_mae=stability_mae,
        n_forecasts=len(forecasts),
    )
