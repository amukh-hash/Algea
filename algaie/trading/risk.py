from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PortfolioTargetConfig:
    top_k: int = 50
    weight_method: str = "softmax"
    softmax_temp: float = 1.0
    max_weight_per_name: float = 0.1
    max_names: int = 50
    min_dollar_position: float = 0.0
    cash_buffer_pct: float = 0.05


class PortfolioTargetBuilder:
    def __init__(self, config: PortfolioTargetConfig) -> None:
        self.config = config

    def build_targets(
        self,
        signals: pd.DataFrame,
        eligibility: pd.DataFrame,
        prices: pd.DataFrame,
        asof: pd.Timestamp,
        total_equity: Optional[float] = None,
    ) -> pd.DataFrame:
        signals = signals.copy()
        eligibility = eligibility.copy()
        signals["date"] = pd.to_datetime(signals["date"])
        eligibility["date"] = pd.to_datetime(eligibility["date"])
        merged = signals.merge(eligibility, on=["date", "ticker"], how="inner")
        merged = merged[merged["is_eligible"]]
        daily = merged[merged["date"] == asof].copy()
        if daily.empty:
            return pd.DataFrame(columns=["date", "ticker", "target_weight", "score", "rank"])
        daily = daily.nsmallest(self.config.top_k, "rank")
        daily = daily.nsmallest(self.config.max_names, "rank")
        weights = self._compute_weights(daily)
        daily["target_weight"] = weights
        if total_equity is not None and self.config.min_dollar_position > 0:
            min_weight = self.config.min_dollar_position / total_equity
            daily = daily[daily["target_weight"] >= min_weight]
            if daily.empty:
                return pd.DataFrame(columns=["date", "ticker", "target_weight", "score", "rank"])
        daily = self._cap_weights(daily)
        daily["target_weight"] = daily["target_weight"] * (1 - self.config.cash_buffer_pct)
        return daily[["date", "ticker", "target_weight", "score", "rank"]]

    def _compute_weights(self, daily: pd.DataFrame) -> pd.Series:
        if self.config.weight_method == "softmax":
            scores = daily["score"].astype(float).to_numpy()
            scores = scores - np.max(scores)
            exp_scores = np.exp(scores / max(self.config.softmax_temp, 1e-6))
            weights = exp_scores / exp_scores.sum()
            return pd.Series(weights, index=daily.index)
        if self.config.weight_method == "rank":
            weights = 1.0 / daily["rank"].astype(float)
            return weights / weights.sum()
        if self.config.weight_method == "zscore":
            scores = daily["score"].astype(float)
            z = (scores - scores.mean()) / scores.std(ddof=0)
            clipped = z.clip(-2, 2)
            relu = clipped.clip(lower=0)
            if relu.sum() == 0:
                relu = pd.Series(np.ones(len(relu)), index=relu.index)
            return relu / relu.sum()
        raise ValueError(f"Unknown weight_method: {self.config.weight_method}")

    def _cap_weights(self, daily: pd.DataFrame) -> pd.DataFrame:
        capped = daily.copy()
        for _ in range(5):
            over = capped["target_weight"] > self.config.max_weight_per_name
            if not over.any():
                break
            capped.loc[over, "target_weight"] = self.config.max_weight_per_name
            total = capped["target_weight"].sum()
            if total > 0:
                capped["target_weight"] = capped["target_weight"] / total
        return capped
