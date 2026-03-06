"""
Train TD3 Agent — Offline RL via Conservative Q-Learning (CQL).

Trains the TD3 Actor-Critic + RLStateProjector on historical VRP PnL
tracks using the OfflineVRPEnv environment.

Device: cuda:0 (3090 Ti, 24GB) — offline batch training.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from algae.models.rl.td3 import TD3Actor, TD3Critic
from backend.app.ml_platform.models.rl_policy.model import RLStateProjector

logger = logging.getLogger(__name__)


class CQLTrainer:
    """Conservative Q-Learning trainer for offline TD3.

    CQL adds a penalty to Q-values of out-of-distribution (OOD) actions,
    preventing the policy from exploiting Q-function overestimation on
    unseen state-action pairs.

    Parameters
    ----------
    state_dim : int
        Dimension of state embeddings (256).
    action_dim : int
        Dimension of action space (2: size_multiplier + veto_logit).
    cql_alpha : float
        CQL regularization coefficient.
    """

    def __init__(
        self,
        state_dim: int = 256,
        action_dim: int = 2,
        lr_actor: float = 1e-4,
        lr_critic: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        cql_alpha: float = 1.0,
        device: str = "cuda:0",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.tau = tau
        self.cql_alpha = cql_alpha

        # Networks
        self.actor = TD3Actor(state_dim, action_dim).to(self.device)
        self.actor_target = TD3Actor(state_dim, action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = TD3Critic(state_dim, action_dim).to(self.device)
        self.critic_target = TD3Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), lr=lr_critic)

    def _soft_update(self, target: nn.Module, source: nn.Module) -> None:
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)

    def train_step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        update_actor: bool = True,
    ) -> dict[str, float]:
        """Single training step with CQL regularization.

        Returns metrics dict.
        """
        # ── Critic update with CQL ──────────────────────────────────
        with torch.no_grad():
            noise = (torch.randn_like(actions) * 0.2).clamp(-0.5, 0.5)
            next_actions = (self.actor_target(next_states) + noise).clamp(-1, 1)
            q1_target, q2_target = self.critic_target(next_states, next_actions)
            q_target = torch.min(q1_target, q2_target)
            target_q = rewards + (1 - dones) * self.gamma * q_target

        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        # CQL regularization: penalize high Q-values on random actions
        random_actions = torch.FloatTensor(states.shape[0], actions.shape[1]).uniform_(-1, 1).to(self.device)
        q1_rand, q2_rand = self.critic(states, random_actions)
        q1_data, q2_data = self.critic(states, actions)

        cql_penalty = (
            torch.logsumexp(q1_rand, dim=0).mean() - q1_data.mean()
            + torch.logsumexp(q2_rand, dim=0).mean() - q2_data.mean()
        )
        critic_loss += self.cql_alpha * cql_penalty

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        metrics = {"critic_loss": critic_loss.item(), "cql_penalty": cql_penalty.item()}

        # ── Actor update (delayed) ──────────────────────────────────
        if update_actor:
            actor_actions = self.actor(states)
            q1_policy, _ = self.critic(states, actor_actions)
            actor_loss = -q1_policy.mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()

            self._soft_update(self.actor_target, self.actor)
            self._soft_update(self.critic_target, self.critic)
            metrics["actor_loss"] = actor_loss.item()

        return metrics


def train_td3_agent(
    pnl_path: str,
    states_path: str,
    output_path: str = "backend/artifacts/model_weights/td3_policy.pt",
    raw_feature_dim: int = 64,
    margin_path: str | None = None,
    device: str = "cuda:0",
    epochs: int = 100,
    batch_size: int = 256,
    policy_delay: int = 2,
) -> dict:
    """Train TD3 + RLStateProjector offline.

    Parameters
    ----------
    pnl_path : str
        ``.npy`` file of daily PnL track.
    states_path : str
        ``.npy`` file of raw feature vectors ``[T, raw_feature_dim]``.
    output_path : str
        Where to save ``{"actor": ..., "projector": ...}``.
    raw_feature_dim : int
        Input dimension for the RLStateProjector.
    """
    from algae.training.env_statarb import OfflineVRPEnv

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    logger.info("Training TD3 on %s", dev)

    pnl = np.load(pnl_path)
    raw_states = np.load(states_path)
    margin = np.load(margin_path) if margin_path else None
    T = len(pnl)

    # Build state projector and project raw features
    projector = RLStateProjector(raw_dim=raw_feature_dim, embed_dim=256).to(dev)
    projector.train()

    raw_tensor = torch.tensor(raw_states, dtype=torch.float32, device=dev)
    with torch.no_grad():
        state_embeddings = projector(raw_tensor).cpu().numpy()

    # Build environment
    env = OfflineVRPEnv(
        pnl_history=pnl,
        state_embeddings=state_embeddings,
        margin_history=margin,
    )

    # Collect demonstration data via environment rollout
    logger.info("Collecting demonstrations from %d timesteps", T)
    states_list, actions_list, rewards_list, next_states_list, dones_list = [], [], [], [], []

    state = env.reset()
    for t in range(T - 1):
        # Behavioral policy: proportional to PnL sign (simple heuristic)
        if pnl[t] > 0:
            action = np.array([0.6, 0.3])  # Moderate size, no veto
        elif pnl[t] < -abs(pnl[:t+1].std() if t > 0 else 1.0):
            action = np.array([-0.5, -0.5])  # Reduce size, lean veto
        else:
            action = np.array([0.0, 0.1])   # Neutral

        next_state, reward, done, _ = env.step(action)

        states_list.append(state)
        actions_list.append(action)
        rewards_list.append(reward)
        next_states_list.append(next_state)
        dones_list.append(float(done))

        state = next_state
        if done:
            break

    # Convert to tensors
    all_states = torch.tensor(np.array(states_list), dtype=torch.float32, device=dev)
    all_actions = torch.tensor(np.array(actions_list), dtype=torch.float32, device=dev)
    all_rewards = torch.tensor(np.array(rewards_list), dtype=torch.float32, device=dev).unsqueeze(1)
    all_next_states = torch.tensor(np.array(next_states_list), dtype=torch.float32, device=dev)
    all_dones = torch.tensor(np.array(dones_list), dtype=torch.float32, device=dev).unsqueeze(1)

    n_samples = len(all_states)
    logger.info("Collected %d demonstration transitions", n_samples)

    # Train CQL
    trainer = CQLTrainer(state_dim=256, action_dim=2, device=device)
    best_critic_loss = float("inf")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        # Random mini-batches
        total_metrics = {"critic_loss": 0, "cql_penalty": 0, "actor_loss": 0}
        n_batches = max(1, n_samples // batch_size)

        for batch_idx in range(n_batches):
            idx = np.random.choice(n_samples, size=min(batch_size, n_samples), replace=False)
            metrics = trainer.train_step(
                states=all_states[idx],
                actions=all_actions[idx],
                rewards=all_rewards[idx],
                next_states=all_next_states[idx],
                dones=all_dones[idx],
                update_actor=(batch_idx % policy_delay == 0),
            )
            for k, v in metrics.items():
                total_metrics[k] += v

        for k in total_metrics:
            total_metrics[k] /= n_batches

        if epoch % 10 == 0:
            logger.info(
                "Epoch %d/%d — critic=%.4f cql=%.4f actor=%.4f",
                epoch + 1, epochs,
                total_metrics["critic_loss"],
                total_metrics["cql_penalty"],
                total_metrics.get("actor_loss", 0),
            )

        if total_metrics["critic_loss"] < best_critic_loss:
            best_critic_loss = total_metrics["critic_loss"]
            torch.save({
                "actor": trainer.actor.state_dict(),
                "projector": projector.state_dict(),
            }, out)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("TD3 training complete → %s", out)
    return {"status": "ok", "checkpoint": str(out), "best_critic_loss": best_critic_loss}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--pnl", required=True, help="Daily PnL track .npy")
    parser.add_argument("--states", required=True, help="Raw features .npy [T, D]")
    parser.add_argument("--output", default="backend/artifacts/model_weights/td3_policy.pt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    train_td3_agent(
        pnl_path=args.pnl,
        states_path=args.states,
        output_path=args.output,
        device=args.device,
        epochs=args.epochs,
    )
