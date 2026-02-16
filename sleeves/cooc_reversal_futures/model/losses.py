"""Loss functions for the CS-Transformer and cross-asset models.

Primary: Huber loss (robust regression on reversal scores).
Optional: pairwise ranking loss for ordering stability.
Existing: gaussian_nll, sizing_utility.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Existing losses
# ---------------------------------------------------------------------------

def gaussian_nll(mu: torch.Tensor, log_sigma: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    sigma2 = torch.exp(2 * log_sigma).clamp_min(1e-8)
    return (0.5 * torch.log(sigma2) + 0.5 * (y - mu) ** 2 / sigma2).mean()


def huber(y_hat: torch.Tensor, y: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    return F.huber_loss(y_hat, y, delta=delta)


def sizing_utility(mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    return mu / sigma.clamp_min(1e-6) ** 2


# ---------------------------------------------------------------------------
# CS-Transformer loss
# ---------------------------------------------------------------------------

class HuberRankLoss(nn.Module):
    """Combined Huber regression + optional pairwise rank loss.

    Parameters
    ----------
    delta : Huber loss delta (transition from quadratic to linear).
    rank_weight : weight for pairwise ranking component (0 = pure Huber).
    """

    def __init__(self, delta: float = 1.0, rank_weight: float = 0.0) -> None:
        super().__init__()
        self.delta = delta
        self.rank_weight = rank_weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute combined loss.

        Parameters
        ----------
        pred : [B, N] — predicted reversal scores.
        target : [B, N] — true reversal scores (y = -r_oc).
        mask : [B, N] — True for valid instruments.

        Returns
        -------
        Scalar loss.
        """
        if mask is None:
            mask = torch.ones_like(pred, dtype=torch.bool)

        # Huber loss (masked)
        huber_val = F.huber_loss(pred, target, reduction="none", delta=self.delta)
        huber_val = (huber_val * mask.float()).sum() / mask.float().sum().clamp(min=1)

        loss = huber_val

        # Optional pairwise ranking loss
        if self.rank_weight > 0:
            rank_loss = self._pairwise_rank_loss(pred, target, mask)
            loss = loss + self.rank_weight * rank_loss

        return loss

    @staticmethod
    def _pairwise_rank_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Pairwise ranking loss within each day."""
        B, N = pred.shape
        total_loss = torch.tensor(0.0, device=pred.device)
        n_pairs = 0

        for b in range(B):
            valid = mask[b].nonzero(as_tuple=True)[0]
            if len(valid) < 2:
                continue
            p = pred[b, valid]
            t = target[b, valid]
            diff_t = t.unsqueeze(0) - t.unsqueeze(1)
            diff_p = p.unsqueeze(0) - p.unsqueeze(1)
            pos_mask = diff_t > 0
            if pos_mask.any():
                pair_loss = F.relu(-diff_p[pos_mask])
                total_loss = total_loss + pair_loss.sum()
                n_pairs += int(pos_mask.sum().item())

        if n_pairs > 0:
            total_loss = total_loss / n_pairs

        return total_loss


class MultiTaskLoss(nn.Module):
    """Two-head loss: score regression + risk prediction + collapse penalty.

    Parameters
    ----------
    score_delta : Huber delta for score head.
    risk_delta : Huber delta for risk head.
    risk_weight : weight for risk loss component.
    collapse_penalty : penalty weight if risk predictions collapse to near-zero.
    rank_weight : optional pairwise ranking weight on derived score.
    """

    def __init__(
        self,
        score_delta: float = 1.0,
        risk_delta: float = 1.0,
        risk_weight: float = 0.5,
        collapse_penalty: float = 0.1,
        rank_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.score_delta = score_delta
        self.risk_delta = risk_delta
        self.risk_weight = risk_weight
        self.collapse_penalty = collapse_penalty
        self.rank_weight = rank_weight

    def forward(
        self,
        score_pred: torch.Tensor,
        risk_pred: torch.Tensor,
        score_target: torch.Tensor,
        risk_target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute multi-task loss.

        Parameters
        ----------
        score_pred : [B, N] — predicted reversal scores.
        risk_pred : [B, N] — predicted risk (|r_oc| proxy, already softplus'd).
        score_target : [B, N] — true y = -r_oc.
        risk_target : [B, N] — true |r_oc|.
        mask : [B, N] — True for valid instruments.

        Returns
        -------
        Scalar loss.
        """
        if mask is None:
            mask = torch.ones_like(score_pred, dtype=torch.bool)
        mf = mask.float()
        n_valid = mf.sum().clamp(min=1)

        # Score head: Huber regression
        score_loss = F.huber_loss(
            score_pred, score_target, reduction="none", delta=self.score_delta,
        )
        score_loss = (score_loss * mf).sum() / n_valid

        # Risk head: Huber regression on |r_oc|
        risk_loss = F.huber_loss(
            risk_pred, risk_target, reduction="none", delta=self.risk_delta,
        )
        risk_loss = (risk_loss * mf).sum() / n_valid

        # Collapse penalty: log(mean(risk)) — pushes risk away from zero
        mean_risk = (risk_pred * mf).sum() / n_valid
        collapse_loss = -torch.log(mean_risk.clamp(min=1e-8))

        loss = score_loss + self.risk_weight * risk_loss + self.collapse_penalty * collapse_loss

        # Optional pairwise ranking loss on derived score
        if self.rank_weight > 0:
            eps = 1e-6
            derived = score_pred / (eps + risk_pred)
            rank_loss = HuberRankLoss._pairwise_rank_loss(derived, score_target, mask)
            loss = loss + self.rank_weight * rank_loss

        return loss


# ---------------------------------------------------------------------------
# F1: ListNet-style ranking loss (polarity-correct)
# ---------------------------------------------------------------------------

class ListNetRankLoss(nn.Module):
    """Listwise ranking loss: cross-entropy over per-day softmax distributions.

    Both `alpha_pred` and `alpha_target` follow the convention:
    higher alpha = more attractive for LONG.

    alpha_target = -y = r_oc  (from alpha_conventions)
    alpha_pred   = some monotone transform of model output

    Loss = mean_over_days[ CE(softmax(alpha_target/temp_y),
                              softmax(alpha_pred/temp_pred)) ]
    """

    def __init__(
        self,
        temp_y: float = 1.0,
        temp_pred: float = 1.0,
    ) -> None:
        super().__init__()
        self.temp_y = temp_y
        self.temp_pred = temp_pred

    def forward(
        self,
        alpha_pred: torch.Tensor,
        alpha_target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute listwise ranking loss.

        Parameters
        ----------
        alpha_pred : [B, N] — predicted alpha (higher = long).
        alpha_target : [B, N] — target alpha (higher = long).
        mask : [B, N] — True for valid instruments.

        Returns
        -------
        Scalar loss averaged across days.
        """
        if mask is None:
            mask = torch.ones_like(alpha_pred, dtype=torch.bool)

        B = alpha_pred.shape[0]
        total_loss = torch.tensor(0.0, device=alpha_pred.device)
        n_days = 0

        for b in range(B):
            valid = mask[b]
            n_valid = valid.sum().item()
            if n_valid < 2:
                continue

            ap = alpha_pred[b][valid]   # [n_valid]
            at = alpha_target[b][valid]  # [n_valid]

            # Target distribution (ground truth ranking)
            target_dist = F.softmax(at / self.temp_y, dim=0)
            # Predicted distribution
            log_pred_dist = F.log_softmax(ap / self.temp_pred, dim=0)

            # Cross-entropy H(target, pred) = -sum(target * log(pred))
            ce = -(target_dist * log_pred_dist).sum()
            total_loss = total_loss + ce
            n_days += 1

        if n_days > 0:
            total_loss = total_loss / n_days

        return total_loss


# ---------------------------------------------------------------------------
# F2: Student-t NLL for heteroscedastic risk head
# ---------------------------------------------------------------------------

class StudentTNLL(nn.Module):
    """Negative log-likelihood of Student-t distribution.

    Models residual = y - score_raw as Student-t(df, 0, sigma).
    sigma = softplus(log_sigma_raw) + sigma_floor.

    Stable implementation avoiding NaNs for small sigma.
    """

    def __init__(
        self,
        df: float = 5.0,
        sigma_floor: float = 1e-4,
    ) -> None:
        super().__init__()
        self.df = df
        self.sigma_floor = sigma_floor
        # Precompute constant terms of log-likelihood
        import math
        self._log_const = (
            math.lgamma((df + 1) / 2)
            - math.lgamma(df / 2)
            - 0.5 * math.log(df * math.pi)
        )

    def forward(
        self,
        residual: torch.Tensor,
        log_sigma_raw: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute Student-t NLL.

        Parameters
        ----------
        residual : [B, N] — y - score_raw.
        log_sigma_raw : [B, N] — raw output of risk head.
        mask : [B, N] — True for valid instruments.

        Returns
        -------
        Scalar NLL averaged over valid entries.
        """
        sigma = F.softplus(log_sigma_raw) + self.sigma_floor

        if mask is None:
            mask = torch.ones_like(residual, dtype=torch.bool)
        mf = mask.float()
        n_valid = mf.sum().clamp(min=1)

        z = residual / sigma
        # NLL = log(sigma) + 0.5*(df+1)*log(1 + z^2/df) - const
        nll = (
            torch.log(sigma)
            + 0.5 * (self.df + 1) * torch.log(1.0 + z ** 2 / self.df)
            - self._log_const
        )

        return (nll * mf).sum() / n_valid


# ---------------------------------------------------------------------------
# F1+F2: Ranking-first composite loss (replaces MultiTaskLoss)
# ---------------------------------------------------------------------------

class RankingMultiTaskLoss(nn.Module):
    """Ranking-first multitask loss for CS-Transformer two-head model.

    Components (all computed per-day, averaged across batch):
    1. ListNet ranking loss on (alpha_pred, alpha_target)
    2. Huber regression stabilizer: Huber(score_raw, y)
    3. Student-t NLL risk loss on residual = y - score_raw
    4. Collapse penalty: low var(score_raw) or var(log_sigma_raw) per day

    alpha_target = -y = r_oc  (constructed by caller via alpha_conventions)
    alpha_pred   = -score_raw  (or -derived; caller decides — must be consistent)
    """

    def __init__(
        self,
        rank_loss_weight: float = 1.0,
        reg_loss_weight: float = 0.1,
        risk_loss_weight: float = 0.5,
        collapse_weight: float = 0.1,
        temp_y: float = 1.0,
        temp_pred: float = 1.0,
        risk_df: float = 5.0,
        sigma_floor: float = 1e-4,
        collapse_var_threshold: float = 1e-4,
        huber_delta: float = 1.0,
    ) -> None:
        super().__init__()
        self.rank_loss_weight = rank_loss_weight
        self.reg_loss_weight = reg_loss_weight
        self.risk_loss_weight = risk_loss_weight
        self.collapse_weight = collapse_weight
        self.collapse_var_threshold = collapse_var_threshold
        self.huber_delta = huber_delta

        self.listnet = ListNetRankLoss(temp_y=temp_y, temp_pred=temp_pred)
        self.student_t = StudentTNLL(df=risk_df, sigma_floor=sigma_floor)

    def forward(
        self,
        score_raw: torch.Tensor,
        log_sigma_raw: torch.Tensor,
        y: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute ranking-first composite loss.

        Parameters
        ----------
        score_raw : [B, N] — raw score head output.
        log_sigma_raw : [B, N] — raw risk head output (log sigma).
        y : [B, N] — label = -r_oc.
        mask : [B, N] — True for valid instruments.

        Returns
        -------
        Scalar loss.
        """
        if mask is None:
            mask = torch.ones_like(score_raw, dtype=torch.bool)
        mf = mask.float()
        n_valid = mf.sum().clamp(min=1)

        loss = torch.tensor(0.0, device=score_raw.device)

        # 1. Ranking loss: alpha_pred = -score_raw, alpha_target = -y = r_oc
        if self.rank_loss_weight > 0:
            alpha_pred = -score_raw
            alpha_target = -y
            rank_loss = self.listnet(alpha_pred, alpha_target, mask)
            loss = loss + self.rank_loss_weight * rank_loss

        # 2. Regression stabilizer: Huber(score_raw, y)
        if self.reg_loss_weight > 0:
            reg_loss = F.huber_loss(
                score_raw, y, reduction="none", delta=self.huber_delta,
            )
            reg_loss = (reg_loss * mf).sum() / n_valid
            loss = loss + self.reg_loss_weight * reg_loss

        # 3. Risk loss: Student-t NLL on residual = y - score_raw
        if self.risk_loss_weight > 0:
            residual = y - score_raw
            risk_loss = self.student_t(residual, log_sigma_raw, mask)
            loss = loss + self.risk_loss_weight * risk_loss

        # 4. Collapse penalty
        if self.collapse_weight > 0:
            collapse = self._collapse_penalty(score_raw, log_sigma_raw, mask)
            loss = loss + self.collapse_weight * collapse

        return loss

    def _collapse_penalty(
        self,
        score_raw: torch.Tensor,
        log_sigma_raw: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Penalize low per-day variance of score and risk head outputs."""
        B = score_raw.shape[0]
        penalty = torch.tensor(0.0, device=score_raw.device)
        n_days = 0

        for b in range(B):
            valid = mask[b]
            n_valid = valid.sum().item()
            if n_valid < 2:
                continue

            score_day = score_raw[b][valid]
            risk_day = log_sigma_raw[b][valid]

            score_var = score_day.var()
            risk_var = risk_day.var()

            # Penalize if variance below threshold: penalty = max(0, threshold - var)
            penalty = penalty + F.relu(
                torch.tensor(self.collapse_var_threshold, device=score_raw.device) - score_var
            )
            penalty = penalty + F.relu(
                torch.tensor(self.collapse_var_threshold, device=score_raw.device) - risk_var
            )
            n_days += 1

        if n_days > 0:
            penalty = penalty / n_days

        return penalty


