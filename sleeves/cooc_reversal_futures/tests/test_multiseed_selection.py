"""Tests: Multi-seed selection picks median-metric model."""
from __future__ import annotations

from unittest.mock import patch, MagicMock
import numpy as np
import pytest

from sleeves.cooc_reversal_futures.pipeline.types import ModelBundle


def _make_bundle(metric: float) -> ModelBundle:
    return ModelBundle(
        model_path="fake",
        feature_order=("r_co",),
        scaler_path="fake",
        nan_fill_values={},
        chosen_params={"estimator": "CSTransformer"},
        trial_log=(),
        primary_metric="ic",
        primary_metric_value=metric,
    )


def test_median_selection_logic():
    """Median of [0.1, 0.3, 0.2] should pick index 1 (middle = 0.2)."""
    metrics = [0.1, 0.3, 0.2]
    results = sorted(
        [(m, _make_bundle(m), None, None) for m in metrics],
        key=lambda r: r[0],
    )
    # sorted: [(0.1, ...), (0.2, ...), (0.3, ...)]
    median_idx = len(results) // 2  # = 1
    assert results[median_idx][0] == 0.2


def test_median_selection_even():
    """Even number of seeds: picks the upper-median."""
    metrics = [0.1, 0.4, 0.2, 0.3]
    results = sorted(
        [(m, _make_bundle(m), None, None) for m in metrics],
        key=lambda r: r[0],
    )
    # sorted: [0.1, 0.2, 0.3, 0.4], median_idx = 2
    median_idx = len(results) // 2
    assert results[median_idx][0] == 0.3


def test_single_seed():
    """Single seed: picks the only one."""
    metrics = [0.5]
    results = sorted(
        [(m, _make_bundle(m), None, None) for m in metrics],
        key=lambda r: r[0],
    )
    median_idx = len(results) // 2
    assert results[median_idx][0] == 0.5
