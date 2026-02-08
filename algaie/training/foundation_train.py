from __future__ import annotations

from pathlib import Path

import pandas as pd

from algaie.models.foundation.chronos2 import FoundationModelConfig, SimpleChronos2


def train_foundation_model(canonical_daily: pd.DataFrame, destination: Path) -> None:
    model = SimpleChronos2(FoundationModelConfig())
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("foundation model placeholder", encoding="utf-8")
    _ = model
