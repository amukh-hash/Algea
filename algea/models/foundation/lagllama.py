from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from algea.models.foundation.base import FoundationModel, FoundationModelOutput
from algea.models.foundation.registry import register


@dataclass(frozen=True)
class LagLlamaConfig:
    context_length: int = 60
    horizon_short: int = 5
    horizon_long: int = 10


class LagLlamaStub(FoundationModel):
    def __init__(self, config: LagLlamaConfig) -> None:
        self.config = config

    def infer(self, canonical_daily: pd.DataFrame) -> pd.DataFrame:
        """TODO: Implement LagLlama priors generation with Chronos-compatible output."""
        raise NotImplementedError("LagLlama stub not implemented")

    def infer_priors(self, canonical_df: pd.DataFrame, asof: pd.Timestamp | None = None) -> FoundationModelOutput:
        raise NotImplementedError("LagLlama stub not implemented")

    def train(self, train_data: pd.DataFrame, val_data: pd.DataFrame | None = None) -> str:
        raise NotImplementedError("LagLlama stub not implemented")


register("lagllama", LagLlamaStub)
