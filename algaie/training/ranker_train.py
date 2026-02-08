from __future__ import annotations

from pathlib import Path

import pandas as pd

from algaie.models.ranker.rank_transformer import RankerConfig, SimpleRanker


def train_ranker_model(features: pd.DataFrame, destination: Path) -> None:
    model = SimpleRanker(RankerConfig())
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("ranker model placeholder", encoding="utf-8")
    _ = model
