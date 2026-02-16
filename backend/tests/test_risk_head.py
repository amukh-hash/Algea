"""Tests for F2 heteroscedastic risk head — Student-t NLL + diagnostics."""
import numpy as np
import torch
import pytest

from sleeves.cooc_reversal_futures.model.losses import StudentTNLL


class TestStudentTNLLCorrelation:
    """On heteroscedastic synthetic data, sigma_pred should correlate with |residual|."""

    def test_heteroscedastic_correlation(self):
        """Train a tiny linear sigma model on heteroscedastic data."""
        torch.manual_seed(42)
        np.random.seed(42)

        N = 200
        # Ground truth scale varies across instruments
        true_sigma = np.abs(np.random.randn(N)) * 0.5 + 0.1
        residuals = np.random.randn(N) * true_sigma

        # Risk head is just a linear model predicting log_sigma
        # True: log_sigma ~ log(true_sigma)
        log_sigma_target = torch.tensor(np.log(true_sigma), dtype=torch.float32)
        residual_t = torch.tensor(residuals, dtype=torch.float32)

        # Trainable log_sigma
        log_sigma = torch.nn.Parameter(torch.zeros(N))
        optimizer = torch.optim.Adam([log_sigma], lr=0.01)
        loss_fn = StudentTNLL(df=5.0, sigma_floor=1e-4)

        for _ in range(200):
            optimizer.zero_grad()
            loss = loss_fn(
                residual_t.unsqueeze(0), log_sigma.unsqueeze(0),
                mask=torch.ones(1, N, dtype=torch.bool),
            )
            loss.backward()
            optimizer.step()

        # Check correlation
        from scipy import stats as sp_stats
        sigma_pred = torch.nn.functional.softplus(log_sigma).detach().numpy()
        corr, pval = sp_stats.spearmanr(sigma_pred, np.abs(residuals))
        assert corr > 0.2, f"Expected positive correlation, got {corr:.3f}"

    def test_no_nans_extreme(self):
        """Extreme residuals and log_sigma should not produce NaN."""
        residual = torch.tensor([[100.0, -100.0, 0.0, 1e-8]])
        log_sigma = torch.tensor([[-20.0, 0.0, 20.0, -5.0]])
        mask = torch.ones(1, 4, dtype=torch.bool)

        loss = StudentTNLL(df=5.0, sigma_floor=1e-4)(residual, log_sigma, mask)
        assert torch.isfinite(loss), f"Non-finite loss: {loss}"

    def test_gradient_stability(self):
        residual = torch.randn(3, 10, requires_grad=True)
        log_sigma = torch.randn(3, 10, requires_grad=True)
        mask = torch.ones(3, 10, dtype=torch.bool)

        loss = StudentTNLL(df=5.0)(residual, log_sigma, mask)
        loss.backward()

        assert residual.grad is not None
        assert log_sigma.grad is not None
        assert torch.all(torch.isfinite(residual.grad))
        assert torch.all(torch.isfinite(log_sigma.grad))


class TestRiskCalibrationDiagnostics:
    """F2 extended diagnostics in risk_calibration."""

    def test_extended_diagnostics_computed(self):
        from sleeves.cooc_reversal_futures.pipeline.risk_calibration import (
            compute_risk_calibration_extended,
        )
        np.random.seed(42)
        N = 500
        r_oc = np.random.randn(N) * 0.02
        # Risk prediction correlated with |r_oc|
        risk_pred = np.abs(r_oc) + np.random.randn(N) * 0.005
        days = np.repeat(np.arange(50), 10)

        report = compute_risk_calibration_extended(
            risk_pred, r_oc,
            sigma_floor=1e-4, sigma_cap=1.0,
            trading_days=days,
        )
        assert hasattr(report, "saturation_fraction_floor")
        assert hasattr(report, "saturation_fraction_cap")
        assert hasattr(report, "mean_daily_sigma_dispersion")
        assert 0.0 <= report.saturation_fraction_floor <= 1.0
        assert 0.0 <= report.saturation_fraction_cap <= 1.0


class TestDerivedScoreStabilizer:
    """score_stabilizer uses sigma_from_log_sigma (F2 update)."""

    def test_stabilizer_numpy(self):
        from sleeves.cooc_reversal_futures.model.score_stabilizer import stabilize_derived_score
        raw_score = np.array([1.0, -1.0, 0.5, -0.5])
        log_sigma_raw = np.array([0.0, 0.0, 0.0, 0.0])

        derived = stabilize_derived_score(raw_score, log_sigma_raw)
        assert np.all(np.isfinite(derived))
        assert derived.shape == raw_score.shape

    def test_stabilizer_torch(self):
        from sleeves.cooc_reversal_futures.model.score_stabilizer import stabilize_derived_score
        raw_score = torch.tensor([1.0, -1.0, 0.5])
        log_sigma_raw = torch.tensor([0.0, -1.0, 1.0])

        derived = stabilize_derived_score(raw_score, log_sigma_raw)
        assert isinstance(derived, torch.Tensor)
        assert torch.all(torch.isfinite(derived))
