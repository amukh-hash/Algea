"""
Offline VRP Environment — Gym wrapper for TD3 behavioral cloning.

Simulates historical VRP option grid traversal with a reward function that
penalizes margin usage and rewards Sharpe ratio stability.

State:  256-dim embedding from RLStateProjector
Action: [size_multiplier ∈ [-1, 1], veto_logit ∈ [-1, 1]]  (Tanh-bounded)
Reward: R_t = PnL_t − λ×MarginUtilized_t − γ×DrawdownPenalty
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class OfflineVRPEnv:
    """Offline VRP environment for behavioral cloning / CQL training.

    Replays historical PnL tracks and option grid embeddings.

    Parameters
    ----------
    pnl_history : np.ndarray
        Shape ``[T]`` — daily PnL track.
    state_embeddings : np.ndarray
        Shape ``[T, 256]`` — pre-computed state embeddings.
    margin_history : np.ndarray
        Shape ``[T]`` — daily margin utilization (0 to 1).
    lambda_margin : float
        Margin penalty coefficient.
    gamma_drawdown : float
        Drawdown penalty coefficient.
    """

    def __init__(
        self,
        pnl_history: np.ndarray,
        state_embeddings: np.ndarray,
        margin_history: Optional[np.ndarray] = None,
        lambda_margin: float = 0.5,
        gamma_drawdown: float = 1.0,
    ):
        self.pnl = pnl_history
        self.states = state_embeddings
        self.margin = margin_history if margin_history is not None else np.zeros_like(pnl_history)
        self.lambda_m = lambda_margin
        self.gamma_dd = gamma_drawdown
        self.T = len(pnl_history)
        self.t = 0
        self.cumulative_pnl = 0.0
        self.peak_pnl = 0.0

        assert len(state_embeddings) == self.T, "state_embeddings must match pnl_history length"

    @property
    def state_dim(self) -> int:
        return self.states.shape[1]

    @property
    def action_dim(self) -> int:
        return 2

    def reset(self) -> np.ndarray:
        self.t = 0
        self.cumulative_pnl = 0.0
        self.peak_pnl = 0.0
        return self.states[0]

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        """Execute one step.

        Parameters
        ----------
        action : np.ndarray
            ``[size_multiplier, veto_logit]`` in [-1, 1] (Tanh output).

        Returns
        -------
        tuple
            (next_state, reward, done, info)
        """
        size_mult = float(np.clip((action[0] + 1) / 2, 0, 1))  # Map [-1,1] → [0,1]
        veto = action[1] < 0  # Negative logit → veto

        # PnL scaled by position size
        raw_pnl = self.pnl[self.t] * (0.0 if veto else size_mult)
        self.cumulative_pnl += raw_pnl
        self.peak_pnl = max(self.peak_pnl, self.cumulative_pnl)
        drawdown = self.peak_pnl - self.cumulative_pnl

        # Reward = PnL - λ×Margin - γ×Drawdown
        margin_penalty = self.lambda_m * self.margin[self.t] * size_mult
        drawdown_penalty = self.gamma_dd * max(drawdown, 0)
        reward = raw_pnl - margin_penalty - drawdown_penalty

        self.t += 1
        done = self.t >= self.T

        next_state = self.states[min(self.t, self.T - 1)]
        info = {
            "raw_pnl": raw_pnl,
            "size_mult": size_mult,
            "veto": veto,
            "drawdown": drawdown,
            "cumulative_pnl": self.cumulative_pnl,
        }

        return next_state, reward, done, info
