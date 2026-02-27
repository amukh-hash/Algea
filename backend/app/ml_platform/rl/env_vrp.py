from __future__ import annotations

from .env_base import StepResult, TradingEnvBase


class VRPSizingEnv(TradingEnvBase):
    def __init__(self, seed: int = 7, horizon: int = 8):
        super().__init__(seed=seed)
        self.horizon = horizon

    def _state(self) -> dict[str, float]:
        x = float((self.seed * 31 + self.t * 17) % 100) / 100.0
        return {
            "forecast_edge": x - 0.5,
            "uncertainty": 0.2 + 0.1 * (self.t % 3),
            "drift": 0.1 * (self.t % 2),
            "exposure": 0.3,
            "liquidity": 0.8,
            "regime": float((self.seed + self.t) % 4) / 4.0,
        }

    def step(self, action: dict) -> StepResult:
        s = self._state()
        mult = max(0.0, min(1.0, float(action.get("size_multiplier", 0.0))))
        veto = bool(action.get("veto", False))
        effective = 0.0 if veto else mult
        reward = (s["forecast_edge"] * effective) - (0.2 * s["uncertainty"] * effective) - (0.05 * effective)
        self.t += 1
        done = self.t >= self.horizon
        return StepResult(state=self._state(), reward=float(reward), done=done, info={"effective": effective})
