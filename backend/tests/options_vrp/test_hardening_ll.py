"""
Tests for forecast hardening — clamp, monotonicity, baseline blend, health gating.
"""
import numpy as np
import pytest

from algaie.models.tsfm.lag_llama.config import LagLlamaConfig
from algaie.models.tsfm.lag_llama.hardening import BaselineBlender, ForecastSanitizer
from algaie.models.tsfm.lag_llama.inference import ForecastResult


def _make_forecast(
    quantiles: dict | None = None,
    health_score: float = 1.0,
) -> ForecastResult:
    return ForecastResult(
        as_of_date="2024-07-15",
        underlying="SPY",
        series_type="sqret",
        quantiles=quantiles or {0.50: 0.18, 0.90: 0.28, 0.95: 0.35, 0.99: 0.48},
        model_id="test",
        inference_seed=42,
        health_score=health_score,
    )


class TestForecastSanitizer:
    def test_clamp_bounds(self):
        cfg = LagLlamaConfig(rv_clamp_min=0.05, rv_clamp_max=0.80)
        san = ForecastSanitizer(cfg)
        fc = _make_forecast({0.50: 0.01, 0.90: 0.90, 0.95: 1.50, 0.99: 2.00})
        result = san.sanitize(fc)
        for q, v in result.quantiles.items():
            assert 0.05 <= v <= 0.80

    def test_monotonicity_enforced(self):
        cfg = LagLlamaConfig()
        san = ForecastSanitizer(cfg)
        # p50 > p90 → violation
        fc = _make_forecast({0.50: 0.30, 0.90: 0.25, 0.95: 0.40, 0.99: 0.50})
        result = san.sanitize(fc)
        sorted_qs = sorted(result.quantiles.keys())
        vals = [result.quantiles[q] for q in sorted_qs]
        for i in range(1, len(vals)):
            assert vals[i] >= vals[i - 1]

    def test_valid_forecast_unchanged(self):
        cfg = LagLlamaConfig()
        san = ForecastSanitizer(cfg)
        fc = _make_forecast()
        result = san.sanitize(fc)
        for q in fc.quantiles:
            assert abs(result.quantiles[q] - fc.quantiles[q]) < 1e-8


class TestBaselineBlender:
    def test_full_model_weight_at_high_health(self):
        cfg = LagLlamaConfig(
            calibration_coverage_min=0.80,
            baseline_blend_weight=0.30,
        )
        blender = BaselineBlender(cfg)
        fc = _make_forecast(health_score=0.95)
        baseline = {0.50: 0.15, 0.90: 0.20, 0.95: 0.25, 0.99: 0.30}
        result = blender.blend(fc, baseline, 0.95)
        # With 30% baseline blend, model contributes 70%
        for q in fc.quantiles:
            expected = 0.70 * fc.quantiles[q] + 0.30 * baseline[q]
            assert abs(result.quantiles[q] - expected) < 1e-8

    def test_more_baseline_at_low_health(self):
        cfg = LagLlamaConfig(
            calibration_coverage_min=0.80,
            baseline_blend_weight=0.30,
        )
        blender = BaselineBlender(cfg)
        fc = _make_forecast(health_score=0.40)
        baseline = {0.50: 0.15, 0.90: 0.20, 0.95: 0.25, 0.99: 0.30}
        result_lo = blender.blend(fc, baseline, 0.40)
        result_hi = blender.blend(fc, baseline, 0.95)
        # Low health should be closer to baseline (smaller model contribution)
        for q in fc.quantiles:
            dist_lo = abs(result_lo.quantiles[q] - baseline[q])
            dist_hi = abs(result_hi.quantiles[q] - baseline[q])
            assert dist_lo <= dist_hi + 1e-8

    def test_zero_health_full_baseline(self):
        cfg = LagLlamaConfig(
            calibration_coverage_min=0.80,
            baseline_blend_weight=0.30,
        )
        blender = BaselineBlender(cfg)
        fc = _make_forecast()
        baseline = {0.50: 0.15, 0.90: 0.20, 0.95: 0.25, 0.99: 0.30}
        result = blender.blend(fc, baseline, 0.0)
        for q in baseline:
            assert abs(result.quantiles[q] - baseline[q]) < 1e-6
