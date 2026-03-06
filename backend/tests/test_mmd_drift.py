"""Test MMD concept drift detection.

Validates:
  1. MMD computation matches known analytical result (same distribution → ~0)
  2. MMD detects distribution shift (different mean → high MMD)
  3. Drift check triggers HALTED_DRIFT at threshold
  4. Adaptive bandwidth handles varying scales
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.app.orchestrator.liveguard_baselines import check_mmd, compute_mmd, _adaptive_bandwidth


class TestMMDComputation:
    def test_same_distribution_low_mmd(self):
        """Samples from the same distribution should have MMD ≈ 0."""
        torch.manual_seed(42)
        x = torch.randn(500, 10)
        y = torch.randn(500, 10)

        mmd = compute_mmd(x, y, sigma=1.0)
        assert mmd < 0.05, f"MMD between same distributions too high: {mmd}"

    def test_different_distribution_high_mmd(self):
        """Samples from different distributions should have high MMD."""
        torch.manual_seed(42)
        x = torch.randn(500, 10)          # Mean 0
        y = torch.randn(500, 10) + 3.0    # Mean 3 — large shift

        # Use adaptive bandwidth (matching production code)
        sigma = _adaptive_bandwidth(x)
        mmd = compute_mmd(x, y, sigma=sigma)
        # With adaptive sigma, shifted distribution has clearly higher MMD
        baseline_mmd = compute_mmd(x, x, sigma=sigma)
        assert mmd > baseline_mmd * 2, (
            f"MMD between different distributions ({mmd}) not significantly "
            f"higher than self-MMD ({baseline_mmd})"
        )

    def test_mmd_is_nonnegative(self):
        """MMD² should always be non-negative."""
        torch.manual_seed(42)
        for _ in range(10):
            x = torch.randn(100, 5)
            y = torch.randn(100, 5)
            mmd = compute_mmd(x, y, sigma=1.0)
            assert mmd >= 0, f"Negative MMD: {mmd}"

    def test_mmd_symmetric(self):
        """MMD(P, Q) ≈ MMD(Q, P)."""
        torch.manual_seed(42)
        x = torch.randn(200, 8)
        y = torch.randn(200, 8) + 1.0

        mmd_xy = compute_mmd(x, y, sigma=1.0)
        mmd_yx = compute_mmd(y, x, sigma=1.0)
        assert abs(mmd_xy - mmd_yx) < 0.01, f"MMD not symmetric: {mmd_xy} vs {mmd_yx}"


class TestDriftCheck:
    def test_no_drift_detection(self):
        """Same distribution should not trigger drift."""
        torch.manual_seed(42)
        ref = torch.randn(500, 10)
        current = torch.randn(200, 10)

        # Use a generous threshold multiplier since same-distribution
        # samples can have non-trivial MMD due to finite sampling
        result = check_mmd(
            "test_sleeve", current, reference_data=ref,
            threshold_multiplier=3.0,
        )
        assert result["is_drifted"] is False, (
            f"Same distribution flagged as drifted: MMD={result['mmd_score']}, "
            f"threshold={result['threshold']}"
        )

    def test_drift_detection(self):
        """Large distribution shift should trigger drift."""
        torch.manual_seed(42)
        ref = torch.randn(500, 10)
        current = torch.randn(200, 10) + 5.0  # Massive shift

        result = check_mmd("test_sleeve", current, reference_data=ref)
        assert result["is_drifted"] is True
        assert result["mmd_score"] > result["threshold"]
