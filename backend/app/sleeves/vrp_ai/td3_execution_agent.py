"""TD3 (Twin Delayed DDPG) Execution Agent for VRP options.

Implements the continuous theta/gamma trade-off execution using:
- A deterministically-bounded actor (tanh * max_action)
- Twin critics to mitigate Q-value overestimation
- Delayed policy updates & target-policy smoothing
"""
from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# Networks
# ======================================================================


class TD3Actor(nn.Module):
    """Deterministically-bounded actor network.

    Output is hard-clamped to ``[-max_action, max_action]`` via
    ``tanh`` scaling — satisfying the *Deterministic Bounding*
    operating principle.
    """

    def __init__(self, state_dim: int, action_dim: int, max_action: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh(),
        )
        self.max_action = max_action

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.max_action * self.net(state)


class TwinCritic(nn.Module):
    """Twin Q-networks for overestimation mitigation."""

    def __init__(self, state_dim: int, action_dim: int):
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], dim=1)
        return self.q1(sa), self.q2(sa)


# ======================================================================
# Replay Buffer
# ======================================================================


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

    def sample(self, batch_size: int):
        indices = np.random.choice(len(self._buffer), batch_size, replace=False)
        batch = [self._buffer[i] for i in indices]
        states = torch.FloatTensor(np.array([t.state for t in batch]))
        actions = torch.FloatTensor(np.array([t.action for t in batch]))
        rewards = torch.FloatTensor([t.reward for t in batch]).unsqueeze(1)
        next_states = torch.FloatTensor(np.array([t.next_state for t in batch]))
        dones = torch.FloatTensor([float(t.done) for t in batch]).unsqueeze(1)
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self._buffer)


# ======================================================================
# TD3 Agent
# ======================================================================


class TD3Agent:
    """Full TD3 agent with delayed policy updates and target smoothing.

    Parameters
    ----------
    state_dim : int
        Dimension of the state vector.
    action_dim : int
        Dimension of the action vector.
    max_action : float
        Maximum magnitude of any action component.
    lr : float
        Learning rate for both actor and critic.
    gamma : float
        Discount factor.
    tau : float
        Polyak averaging coefficient for target networks.
    policy_noise : float
        Std-dev of smoothing noise added to target policy.
    noise_clip : float
        Clip magnitude for target policy noise.
    policy_delay : int
        Number of critic updates per actor update.
    buffer_size : int
        Replay buffer capacity.
    batch_size : int
        Mini-batch size.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_delay: int = 2,
        buffer_size: int = 100_000,
        batch_size: int = 64,
    ):
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.batch_size = batch_size

        # Networks
        self.actor = TD3Actor(state_dim, action_dim, max_action)
        self.actor_target = copy.deepcopy(self.actor)

        self.critic = TwinCritic(state_dim, action_dim)
        self.critic_target = copy.deepcopy(self.critic)

        # Optimisers
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # Replay buffer
        self.buffer = ReplayBuffer(buffer_size)

        self._total_it = 0

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray, noise_scale: float = 0.1) -> np.ndarray:
        """Select action with optional exploration noise."""
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0)
            a = self.actor(s).cpu().numpy().flatten()
        noise = np.random.normal(0, noise_scale * self.max_action, size=a.shape)
        return np.clip(a + noise, -self.max_action, self.max_action)

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def train_step(self) -> dict:
        """Single TD3 update step.

        Returns
        -------
        info : dict
            ``critic_loss`` and optionally ``actor_loss`` scalars.
        """
        if len(self.buffer) < self.batch_size:
            return {}

        self._total_it += 1
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        with torch.no_grad():
            # Target policy smoothing
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_actions = (self.actor_target(next_states) + noise).clamp(
                -self.max_action, self.max_action
            )
            # Twin target Q
            tq1, tq2 = self.critic_target(next_states, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * torch.min(tq1, tq2)

        # Critic update
        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        info: dict = {"critic_loss": critic_loss.item()}

        # Delayed actor update
        if self._total_it % self.policy_delay == 0:
            actor_loss = -self.critic.q1(
                torch.cat(
                    [states, self.actor(states)], dim=1
                )
            ).mean()

            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            info["actor_loss"] = actor_loss.item()

            # Polyak update targets
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)

        return info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        for sp, tp in zip(source.parameters(), target.parameters()):
            tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)
