"""TD3 (Twin Delayed DDPG) Execution Agent for VRP options.

Implements a continuous-action RL agent that consumes the dense
context embedding from the Spatial-Temporal Transformer as state S_t
and outputs physically bounded actions:
    A_t ∈ [-1, 1]² → [Spread_Width, Delta_Target]

Key features:
  - Deterministically-bounded actor (tanh * mapped range)
  - Twin critics with BFloat16 autocast (FP8 underflow mitigation)
  - Reward: R_t = Premium - Settlement - Slippage*Contracts - λ*Margin
  - Target smoothing + delayed policy updates
"""
from __future__ import annotations

import copy
import logging
from collections import deque
from dataclasses import dataclass
from typing import Deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Action Mapping Constants
# ═══════════════════════════════════════════════════════════════════════

# Spread width: raw tanh [-1, 1] → [5, 50] dollars
SPREAD_WIDTH_MIN = 5.0
SPREAD_WIDTH_MAX = 50.0
SPREAD_WIDTH_MID = (SPREAD_WIDTH_MIN + SPREAD_WIDTH_MAX) / 2  # 27.5
SPREAD_WIDTH_HALF = (SPREAD_WIDTH_MAX - SPREAD_WIDTH_MIN) / 2  # 22.5

# Delta target: raw tanh [-1, 1] → [0.10, 0.30]
DELTA_TARGET_MIN = 0.10
DELTA_TARGET_MAX = 0.30
DELTA_TARGET_MID = (DELTA_TARGET_MIN + DELTA_TARGET_MAX) / 2  # 0.20
DELTA_TARGET_HALF = (DELTA_TARGET_MAX - DELTA_TARGET_MIN) / 2  # 0.10


def map_actions(raw_actions: torch.Tensor) -> torch.Tensor:
    """Map raw tanh-bounded actions to physical operational space.

    Parameters
    ----------
    raw_actions : torch.Tensor
        Shape ``(batch, 2)`` with values in ``[-1, 1]``.

    Returns
    -------
    torch.Tensor
        Shape ``(batch, 2)`` — ``[spread_width, delta_target]``.
    """
    spread_width = SPREAD_WIDTH_HALF * raw_actions[:, 0] + SPREAD_WIDTH_MID
    delta_target = DELTA_TARGET_HALF * raw_actions[:, 1] + DELTA_TARGET_MID
    return torch.stack([spread_width, delta_target], dim=-1)


# ═══════════════════════════════════════════════════════════════════════
# Actor Network
# ═══════════════════════════════════════════════════════════════════════

class TD3Actor(nn.Module):
    """Deterministically-bounded actor network.

    Output is hard-clamped to ``[-1, 1]`` via ``tanh``, then mapped
    to physical spread/delta ranges.

    Parameters
    ----------
    state_dim : int
        Dimension of the state vector (ST-Transformer embedding).
    action_dim : int
        Dimension of the action vector (2: spread_width, delta_target).
    """

    def __init__(self, state_dim: int, action_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh(),  # bounded [-1, 1]
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Produce raw bounded actions in [-1, 1].

        Use ``map_actions()`` to convert to physical operational space.
        """
        return self.net(state)


# ═══════════════════════════════════════════════════════════════════════
# Twin Critic Networks
# ═══════════════════════════════════════════════════════════════════════

class TwinCritic(nn.Module):
    """Twin Q-networks for overestimation mitigation.

    BFloat16 autocast is applied during Q-value computation to prevent
    FP8 underflow on Ada Lovelace (RTX 4070 Super) tensor cores.
    """

    def __init__(self, state_dim: int, action_dim: int = 2):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute Q-values under BFloat16 for FP8 safety.

        Parameters
        ----------
        state : torch.Tensor
            Shape ``(batch, state_dim)``.
        action : torch.Tensor
            Shape ``(batch, action_dim)``.

        Returns
        -------
        (q1, q2) : tuple[torch.Tensor, torch.Tensor]
            Each shape ``(batch, 1)``.
        """
        sa = torch.cat([state, action], dim=-1)

        # BFloat16 autocast prevents Ada Lovelace FP8 tensor cores from
        # rounding microscopically tiny Q-value gradients to zero,
        # which would produce NaN actions.
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=sa.is_cuda):
            q1 = self.q1(sa)
            q2 = self.q2(sa)

        return q1.to(torch.float32), q2.to(torch.float32)


# ═══════════════════════════════════════════════════════════════════════
# Replay Buffer
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Transition:
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """Fixed-size experience replay buffer."""

    def __init__(self, capacity: int = 100_000):
        self._buffer: Deque[Transition] = deque(maxlen=capacity)

    def push(self, t: Transition) -> None:
        self._buffer.append(t)

    def sample(self, batch_size: int) -> list[Transition]:
        idx = np.random.choice(len(self._buffer), batch_size, replace=False)
        return [self._buffer[i] for i in idx]

    def __len__(self) -> int:
        return len(self._buffer)


# ═══════════════════════════════════════════════════════════════════════
# Reward Function
# ═══════════════════════════════════════════════════════════════════════

def compute_reward(
    premium: float,
    settlement: float,
    slippage_per_contract: float,
    n_contracts: int,
    margin_requirement: float,
    margin_penalty_lambda: float = 0.01,
) -> float:
    """Compute the TD3 reward with slippage and margin penalties.

    R_t = Premium - Settlement - (Slippage × Contracts) - (λ × Margin)

    Parameters
    ----------
    premium : float
        Premium received from selling the spread.
    settlement : float
        Settlement cost at expiration.
    slippage_per_contract : float
        Estimated slippage cost per contract.
    n_contracts : int
        Number of contracts traded.
    margin_requirement : float
        Total margin requirement for the position.
    margin_penalty_lambda : float
        Penalty coefficient for margin utilisation.

    Returns
    -------
    float — scalar reward.
    """
    return (
        premium
        - settlement
        - (slippage_per_contract * n_contracts)
        - (margin_penalty_lambda * margin_requirement)
    )


# ═══════════════════════════════════════════════════════════════════════
# Full TD3 Agent
# ═══════════════════════════════════════════════════════════════════════

class TD3Agent:
    """Full TD3 agent with delayed policy updates and target smoothing.

    Parameters
    ----------
    state_dim : int
        Dimension of the state vector (ST-Transformer embedding).
    action_dim : int
        Dimension of the action vector (2: spread_width, delta_target).
    lr : float
        Learning rate for actor and critic optimizers.
    gamma : float
        Discount factor.
    tau : float
        Soft update coefficient for target networks.
    policy_noise : float
        Target policy smoothing noise standard deviation.
    noise_clip : float
        Clipping range for target policy noise.
    policy_delay : int
        Number of critic updates before each actor update.
    buffer_size : int
        Replay buffer capacity.
    batch_size : int
        Training batch size.
    device : str
        Target device string.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 2,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_delay: int = 2,
        buffer_size: int = 100_000,
        batch_size: int = 64,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.batch_size = batch_size

        # Actor
        self.actor = TD3Actor(state_dim, action_dim).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        # Critic
        self.critic = TwinCritic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # Replay
        self.buffer = ReplayBuffer(buffer_size)
        self._update_counter = 0

    def select_action(self, state: np.ndarray, noise_scale: float = 0.1) -> np.ndarray:
        """Select action with optional exploration noise.

        Returns raw actions in [-1, 1]. Use ``map_actions()`` to convert.
        """
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            raw = self.actor(s).cpu().numpy()[0]
        if noise_scale > 0:
            raw = raw + np.random.normal(0, noise_scale, size=raw.shape)
            raw = np.clip(raw, -1.0, 1.0)
        return raw

    def train_step(self) -> dict[str, float]:
        """Single TD3 update step.

        Returns
        -------
        dict with ``critic_loss`` and optionally ``actor_loss``.
        """
        if len(self.buffer) < self.batch_size:
            return {}

        batch = self.buffer.sample(self.batch_size)
        states = torch.FloatTensor(np.array([t.state for t in batch])).to(self.device)
        actions = torch.FloatTensor(np.array([t.action for t in batch])).to(self.device)
        rewards = torch.FloatTensor([t.reward for t in batch]).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array([t.next_state for t in batch])).to(self.device)
        dones = torch.FloatTensor([float(t.done) for t in batch]).unsqueeze(1).to(self.device)

        # ── Target policy smoothing ──────────────────────────────────
        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_actions = (self.actor_target(next_states) + noise).clamp(-1.0, 1.0)

            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target = rewards + (1.0 - dones) * self.gamma * target_q

        # ── Critic update ────────────────────────────────────────────
        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        info: dict[str, float] = {"critic_loss": critic_loss.item()}

        # ── Delayed actor update ─────────────────────────────────────
        self._update_counter += 1
        if self._update_counter % self.policy_delay == 0:
            actor_loss = -self.critic.q1(
                torch.cat([states, self.actor(states)], dim=-1)
            ).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Soft update target networks
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)

            info["actor_loss"] = actor_loss.item()

        return info

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        for sp, tp in zip(source.parameters(), target.parameters()):
            tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)
