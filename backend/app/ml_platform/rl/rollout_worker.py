from __future__ import annotations

from .env_base import TradingEnvBase
from .seeds import seed_all


def run_rollout(env: TradingEnvBase, steps: int, seed: int) -> list[dict]:
    rng = seed_all(seed)
    transitions: list[dict] = []
    state = env.reset()
    for _ in range(steps):
        action = {"size_multiplier": rng.random(), "veto": False}
        result = env.step(action)
        transitions.append({"state": state, "action": action, "reward": result.reward, "next_state": result.state, "done": result.done})
        state = result.state
        if result.done:
            break
    return transitions
