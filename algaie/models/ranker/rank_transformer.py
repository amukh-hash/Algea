from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class RankerConfig:
    feature_columns: tuple[str, ...] = ("ret_1d", "ret_5d", "vol_20d", "dollar_vol")


class SimpleRanker:
    def __init__(self, config: RankerConfig) -> None:
        self.config = config

    def score(self, features: pd.DataFrame) -> pd.Series:
        return features[list(self.config.feature_columns)].sum(axis=1)
