from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from algaie.data.priors.validate import validate_priors_frame
from algaie.models.foundation.chronos2 import FoundationModelConfig, SimpleChronos2


def build_priors(
    canonical_daily: pd.DataFrame,
    asof: date,
    config: FoundationModelConfig | None = None,
) -> pd.DataFrame:
    model = SimpleChronos2(config or FoundationModelConfig())
    canonical_daily = canonical_daily.copy()
    canonical_daily["date"] = pd.to_datetime(canonical_daily["date"])
    snapshot = canonical_daily[canonical_daily["date"] <= pd.Timestamp(asof)].copy()
    priors = model.infer_priors(snapshot, pd.Timestamp(asof)).priors
    validate_priors_frame(priors)
    return priors


def write_priors(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(destination, index=False)
