from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class FoundationModelOutput:
    priors: pd.DataFrame


class FoundationModel(ABC):
    @abstractmethod
    def infer_priors(self, canonical_df: pd.DataFrame, asof: pd.Timestamp | None = None) -> FoundationModelOutput:
        raise NotImplementedError

    @abstractmethod
    def train(self, train_data: pd.DataFrame, val_data: pd.DataFrame | None = None) -> Any:
        raise NotImplementedError
