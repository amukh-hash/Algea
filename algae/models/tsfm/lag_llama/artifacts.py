"""
Forecast artifact persistence — save/load forecast results and validation reports.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from algae.models.tsfm.lag_llama.inference import ForecastResult


def save_forecast(
    forecast: ForecastResult,
    root: Path,
) -> Path:
    """Persist forecast to {root}/lag_llama/forecasts/date=.../underlying=.../forecast.json."""
    out_dir = (
        root / "lag_llama" / "forecasts"
        / f"date={forecast.as_of_date}"
        / f"underlying={forecast.underlying}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "forecast.json"
    path.write_text(json.dumps(forecast.to_dict(), indent=2, default=str), encoding="utf-8")
    return path


def load_forecast(
    root: Path,
    as_of_date: str,
    underlying: str,
) -> Optional[ForecastResult]:
    """Load a previously saved forecast."""
    path = (
        root / "lag_llama" / "forecasts"
        / f"date={as_of_date}"
        / f"underlying={underlying}"
        / "forecast.json"
    )
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))

    # Reconstruct quantile keys as floats
    quantiles = {float(k): v for k, v in data.get("quantiles", {}).items()}
    return ForecastResult(
        as_of_date=data["as_of_date"],
        underlying=data["underlying"],
        series_type=data["series_type"],
        quantiles=quantiles,
        model_id=data["model_id"],
        inference_seed=data["inference_seed"],
        health_score=data.get("health_score", 1.0),
        is_fallback=data.get("is_fallback", False),
    )


def save_validation_report(
    report_dict: Dict[str, Any],
    root: Path,
    run_id: str,
) -> Path:
    """Persist validation report."""
    out_dir = root / "lag_llama" / "validation" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "validation.json"
    path.write_text(json.dumps(report_dict, indent=2, default=str), encoding="utf-8")
    return path
