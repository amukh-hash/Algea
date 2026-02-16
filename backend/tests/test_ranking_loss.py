"""Tests for F1 ranking loss — ListNetRankLoss and RankingMultiTaskLoss."""
import torch
import pytest

from sleeves.cooc_reversal_futures.model.losses import (
    ListNetRankLoss, StudentTNLL, RankingMultiTaskLoss,
)


class TestListNetRankLoss:
    """ListNet: correct ranking yields lower loss than shuffled."""

    def test_correct_ranking_lower_loss(self):
        torch.manual_seed(42)
        B, N = 4, 8
        mask = torch.ones(B, N, dtype=torch.bool)

        # True alpha target (what we want to rank)
        alpha_target = torch.randn(B, N)

        # Perfect prediction = same ranking
        alpha_pred_correct = alpha_target.clone()
        # Shuffled prediction = wrong ranking
        alpha_pred_shuffled = alpha_target[:, torch.randperm(N)]

        loss_fn = ListNetRankLoss(temp_y=1.0, temp_pred=1.0)
        loss_correct = loss_fn(alpha_pred_correct, alpha_target, mask)
        loss_shuffled = loss_fn(alpha_pred_shuffled, alpha_target, mask)

        assert loss_correct < loss_shuffled, (
            f"Correct ranking loss ({loss_correct:.4f}) should be lower "
            f"than shuffled ({loss_shuffled:.4f})"
        )

    def test_polarity_flip_increases_loss(self):
        torch.manual_seed(42)
        B, N = 4, 8
        mask = torch.ones(B, N, dtype=torch.bool)
        alpha_target = torch.randn(B, N)

        loss_fn = ListNetRankLoss()
        loss_correct = loss_fn(alpha_target, alpha_target, mask)
        loss_flipped = loss_fn(-alpha_target, alpha_target, mask)

        assert loss_flipped > loss_correct, (
            f"Flipped loss ({loss_flipped:.4f}) should exceed correct ({loss_correct:.4f})"
        )

    def test_gradient_flows(self):
        torch.manual_seed(42)
        alpha_pred = torch.randn(2, 6, requires_grad=True)
        alpha_target = torch.randn(2, 6)
        mask = torch.ones(2, 6, dtype=torch.bool)

        loss = ListNetRankLoss()(alpha_pred, alpha_target, mask)
        loss.backward()
        assert alpha_pred.grad is not None
        assert not torch.isnan(alpha_pred.grad).any()

    def test_no_nans(self):
        # Extreme values should not produce NaN
        alpha_pred = torch.tensor([[100.0, -100.0, 0.0, 50.0]])
        alpha_target = torch.tensor([[1.0, -1.0, 0.0, 0.5]])
        mask = torch.ones(1, 4, dtype=torch.bool)

        loss = ListNetRankLoss()(alpha_pred, alpha_target, mask)
        assert not torch.isnan(loss), f"NaN loss: {loss}"
        assert torch.isfinite(loss), f"Infinite loss: {loss}"


class TestStudentTNLL:
    """Student-t NLL gradients and numerics."""

    def test_gradient_flows(self):
        residual = torch.randn(2, 6, requires_grad=True)
        log_sigma = torch.randn(2, 6, requires_grad=True)
        mask = torch.ones(2, 6, dtype=torch.bool)

        loss = StudentTNLL(df=5.0, sigma_floor=1e-4)(residual, log_sigma, mask)
        loss.backward()
        assert residual.grad is not None
        assert log_sigma.grad is not None
        assert not torch.isnan(residual.grad).any()
        assert not torch.isnan(log_sigma.grad).any()

    def test_no_nans_extreme_inputs(self):
        residual = torch.tensor([[100.0, -100.0, 0.0]])
        log_sigma = torch.tensor([[-10.0, 0.0, 10.0]])
        mask = torch.ones(1, 3, dtype=torch.bool)

        loss = StudentTNLL()(residual, log_sigma, mask)
        assert torch.isfinite(loss), f"Non-finite loss: {loss}"


class TestRankingMultiTaskLoss:
    """Full ranking-first composite loss."""

    def test_correct_ranking_lower_total_loss(self):
        torch.manual_seed(42)
        B, N = 4, 8
        mask = torch.ones(B, N, dtype=torch.bool)
        y = torch.randn(B, N)  # label = -r_oc
        log_sigma = torch.randn(B, N)

        loss_fn = RankingMultiTaskLoss(
            rank_loss_weight=1.0, reg_loss_weight=0.1,
            risk_loss_weight=0.5, collapse_weight=0.1,
        )

        # Perfect score: score_raw = y => alpha_pred = -y => matches alpha_target = -y
        loss_perfect = loss_fn(y, log_sigma, y, mask)
        # Anti-score: score_raw = -y
        loss_anti = loss_fn(-y, log_sigma, y, mask)

        assert loss_perfect < loss_anti, (
            f"Perfect ({loss_perfect:.4f}) should be < anti ({loss_anti:.4f})"
        )

    def test_gradient_flows_all_heads(self):
        score_raw = torch.randn(2, 6, requires_grad=True)
        log_sigma = torch.randn(2, 6, requires_grad=True)
        y = torch.randn(2, 6)
        mask = torch.ones(2, 6, dtype=torch.bool)

        loss = RankingMultiTaskLoss()(score_raw, log_sigma, y, mask)
        loss.backward()
        assert score_raw.grad is not None
        assert log_sigma.grad is not None
        assert not torch.isnan(score_raw.grad).any()
        assert not torch.isnan(log_sigma.grad).any()

    def test_no_nans(self):
        score_raw = torch.randn(3, 10)
        log_sigma = torch.randn(3, 10)
        y = torch.randn(3, 10)
        mask = torch.ones(3, 10, dtype=torch.bool)
        mask[0, 8:] = False  # some padding

        loss = RankingMultiTaskLoss()(score_raw, log_sigma, y, mask)
        assert torch.isfinite(loss), f"Non-finite loss: {loss}"

    def test_collapse_penalty_activates(self):
        """Constant score_raw should trigger collapse penalty."""
        B, N = 2, 8
        mask = torch.ones(B, N, dtype=torch.bool)
        y = torch.randn(B, N)

        # Constant outputs: variance = 0
        score_raw = torch.zeros(B, N)
        log_sigma = torch.zeros(B, N)

        loss_fn = RankingMultiTaskLoss(
            rank_loss_weight=0.0, reg_loss_weight=0.0,
            risk_loss_weight=0.0, collapse_weight=1.0,
            collapse_var_threshold=1e-2,
        )
        loss = loss_fn(score_raw, log_sigma, y, mask)
        assert loss > 0, "Collapse penalty should be positive for constant outputs"
