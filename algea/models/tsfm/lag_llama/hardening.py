"""
Forecast hardening — clamp, monotonicity enforcement, baseline blending,
and health-score gating for Lag-Llama output.

Rules:
- Quantiles must be monotonically increasing (sorted if violated).
- RV values clamped to [rv_clamp_min, rv_clamp_max].
- If health_score < min, increase blend weight toward baseline.
- Always produce forecasts (model or blended fallback).
"""
from __future__ import annotations

import logging
from dataclasses import replace
from typing import Dict, Optional

import numpy as np

from algea.models.tsfm.lag_llama.config import LagLlamaConfig
from algea.models.tsfm.lag_llama.inference import ForecastResult

logger = logging.getLogger(__name__)


class ForecastSanitizer:
    """Clamp bounds, enforce monotonicity, audit log violations."""

    def __init__(self, config: LagLlamaConfig) -> None:
        self.config = config

    def sanitize(self, forecast: ForecastResult) -> ForecastResult:
        """Sanitize a single ForecastResult in-place."""
        cfg = self.config
        quantiles = dict(forecast.quantiles)
        violations: list[str] = []

        # 1. Clamp
        for q, v in quantiles.items():
            clamped = float(np.clip(v, cfg.rv_clamp_min, cfg.rv_clamp_max))
            if clamped != v:
                violations.append(f"q{q}: clamped {v:.4f} -> {clamped:.4f}")
            quantiles[q] = clamped

        # 2. Enforce monotonicity — sort by quantile level
        sorted_qs = sorted(quantiles.keys())
        sorted_vals = [quantiles[q] for q in sorted_qs]
        for i in range(1, len(sorted_vals)):
            if sorted_vals[i] < sorted_vals[i - 1]:
                violations.append(
                    f"monotonicity: q{sorted_qs[i]}={sorted_vals[i]:.4f} < "
                    f"q{sorted_qs[i-1]}={sorted_vals[i-1]:.4f}; correcting"
                )
                sorted_vals[i] = sorted_vals[i - 1]

        quantiles = {q: v for q, v in zip(sorted_qs, sorted_vals)}

        if violations:
            logger.warning(
                "ForecastSanitizer [%s/%s]: %s",
                forecast.underlying, forecast.as_of_date,
                "; ".join(violations),
            )

        return ForecastResult(
            as_of_date=forecast.as_of_date,
            underlying=forecast.underlying,
            series_type=forecast.series_type,
            quantiles=quantiles,
            model_id=forecast.model_id,
            inference_seed=forecast.inference_seed,
            health_score=forecast.health_score,
            is_fallback=forecast.is_fallback,
            raw_sqret_quantiles=forecast.raw_sqret_quantiles,
        )


class BaselineBlender:
    """Blend model forecast with EWMA baseline using health-score gating."""

    def __init__(self, config: LagLlamaConfig) -> None:
        self.config = config

    def blend(
        self,
        model_forecast: ForecastResult,
        baseline_quantiles: Dict[float, float],
        health_score: float,
    ) -> ForecastResult:
        """Blend model and baseline according to health score.

        When health_score >= min:
            blend_w = baseline_blend_weight (default weight toward baseline)
        When health_score < min:
            blend_w increases linearly toward 1.0 as health drops to 0.
        """
        cfg = self.config

        if health_score >= cfg.calibration_coverage_min:
            w_baseline = cfg.baseline_blend_weight
        else:
            # Linear ramp: at health=0, w=1.0; at health=min, w=baseline_blend_weight
            frac = health_score / max(cfg.calibration_coverage_min, 1e-6)
            w_baseline = 1.0 - frac * (1.0 - cfg.baseline_blend_weight)

        blended = {}
        for q in model_forecast.quantiles:
            mv = model_forecast.quantiles[q]
            bv = baseline_quantiles.get(q, mv)
            blended[q] = (1.0 - w_baseline) * mv + w_baseline * bv

        return ForecastResult(
            as_of_date=model_forecast.as_of_date,
            underlying=model_forecast.underlying,
            series_type=model_forecast.series_type,
            quantiles=blended,
            model_id=model_forecast.model_id,
            inference_seed=model_forecast.inference_seed,
            health_score=health_score,
            is_fallback=model_forecast.is_fallback,
            raw_sqret_quantiles=model_forecast.raw_sqret_quantiles,
        )
