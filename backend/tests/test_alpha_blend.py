"""Tests for R5: baseline-shrinkage alpha blending."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.app.portfolio.alpha_blend import (
    AlphaBlendConfig,
    blend_alpha,
    compute_lambda,
    compute_lambda_series,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def const_cfg() -> AlphaBlendConfig:
    return AlphaBlendConfig(
        enabled=True,
        lambda_mode="constant",
        lambda_const=0.5,
        lambda_min=0.0,
        lambda_max=1.0,
    )


@pytest.fixture
def linear_cfg() -> AlphaBlendConfig:
    return AlphaBlendConfig(
        enabled=True,
        lambda_mode="confidence_linear",
        lambda_min=0.2,
        lambda_max=0.8,
        conf_low=0.0,
        conf_high=1.0,
    )


# ---------------------------------------------------------------------------
# compute_lambda tests
# ---------------------------------------------------------------------------

class TestComputeLambda:
    def test_constant_returns_lambda_const(self, const_cfg):
        assert compute_lambda(0.5, const_cfg) == 0.5
        assert compute_lambda(0.0, const_cfg) == 0.5
        assert compute_lambda(999.0, const_cfg) == 0.5

    def test_constant_clipped_to_bounds(self):
        cfg = AlphaBlendConfig(
            lambda_mode="constant",
            lambda_const=1.5,
            lambda_min=0.0,
            lambda_max=1.0,
        )
        assert compute_lambda(0.5, cfg) == 1.0

    def test_linear_at_conf_low(self, linear_cfg):
        """At conf_low → lambda_min."""
        assert compute_lambda(0.0, linear_cfg) == pytest.approx(0.2)

    def test_linear_at_conf_high(self, linear_cfg):
        """At conf_high → lambda_max."""
        assert compute_lambda(1.0, linear_cfg) == pytest.approx(0.8)

    def test_linear_at_midpoint(self, linear_cfg):
        """At midpoint → midpoint of [lambda_min, lambda_max]."""
        assert compute_lambda(0.5, linear_cfg) == pytest.approx(0.5)

    def test_linear_below_conf_low_clamps(self, linear_cfg):
        """Below conf_low → lambda_min."""
        assert compute_lambda(-1.0, linear_cfg) == pytest.approx(0.2)

    def test_linear_above_conf_high_clamps(self, linear_cfg):
        """Above conf_high → lambda_max."""
        assert compute_lambda(5.0, linear_cfg) == pytest.approx(0.8)

    def test_linear_zero_span(self):
        """conf_low == conf_high → lambda_min."""
        cfg = AlphaBlendConfig(
            lambda_mode="confidence_linear",
            lambda_min=0.3,
            lambda_max=0.9,
            conf_low=0.5,
            conf_high=0.5,
        )
        assert compute_lambda(0.5, cfg) == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# compute_lambda_series tests
# ---------------------------------------------------------------------------

class TestComputeLambdaSeries:
    def test_shape_preserved(self, linear_cfg):
        confs = pd.Series([0.0, 0.25, 0.5, 0.75, 1.0], index=[1, 2, 3, 4, 5])
        result = compute_lambda_series(confs, linear_cfg)
        assert len(result) == 5
        assert list(result.index) == [1, 2, 3, 4, 5]

    def test_values_match_scalar(self, linear_cfg):
        confs = pd.Series([0.0, 0.5, 1.0])
        result = compute_lambda_series(confs, linear_cfg)
        for i, c in enumerate(confs):
            assert result.iloc[i] == pytest.approx(compute_lambda(c, linear_cfg))


# ---------------------------------------------------------------------------
# blend_alpha tests
# ---------------------------------------------------------------------------

class TestBlendAlpha:
    def test_lam_one_returns_model(self):
        """λ=1 → pure model alpha."""
        model = pd.Series([0.01, 0.02, 0.03])
        base = pd.Series([0.10, 0.20, 0.30])
        result = blend_alpha(model, base, lam=1.0)
        pd.testing.assert_series_equal(result, model.rename("alpha_blended"))

    def test_lam_zero_returns_baseline(self):
        """λ=0 → pure baseline alpha."""
        model = pd.Series([0.01, 0.02, 0.03])
        base = pd.Series([0.10, 0.20, 0.30])
        result = blend_alpha(model, base, lam=0.0)
        pd.testing.assert_series_equal(result, base.rename("alpha_blended"))

    def test_lam_half_averages(self):
        """λ=0.5 → arithmetic mean."""
        model = pd.Series([0.0, 0.10])
        base = pd.Series([0.10, 0.0])
        result = blend_alpha(model, base, lam=0.5)
        expected = pd.Series([0.05, 0.05], name="alpha_blended")
        pd.testing.assert_series_equal(result, expected)

    def test_per_row_lambda(self):
        """Per-row λ series blending."""
        model = pd.Series([1.0, 2.0, 3.0])
        base = pd.Series([10.0, 20.0, 30.0])
        lam = pd.Series([0.0, 0.5, 1.0])
        result = blend_alpha(model, base, lam)
        expected = pd.Series([10.0, 11.0, 3.0], name="alpha_blended")
        pd.testing.assert_series_equal(result, expected)

    def test_deterministic(self):
        """Same inputs → same outputs."""
        model = pd.Series(np.random.RandomState(42).randn(100))
        base = pd.Series(np.random.RandomState(99).randn(100))
        r1 = blend_alpha(model, base, lam=0.7)
        r2 = blend_alpha(model, base, lam=0.7)
        pd.testing.assert_series_equal(r1, r2)
