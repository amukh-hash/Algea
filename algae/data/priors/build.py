from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from algae.data.common import ensure_datetime, write_dataframe
from algae.data.priors.validate import validate_priors_frame
from algae.models.foundation.chronos2 import FoundationModelConfig, SimpleChronos2


def build_priors(
    canonical_daily: pd.DataFrame,
    asof: date,
    config: FoundationModelConfig | None = None,
) -> pd.DataFrame:
    model = SimpleChronos2(config or FoundationModelConfig())
    canonical_daily = ensure_datetime(canonical_daily.copy())
    snapshot = canonical_daily[canonical_daily["date"] <= pd.Timestamp(asof)].copy()
    priors = model.infer_priors(snapshot, pd.Timestamp(asof)).priors
    validate_priors_frame(priors)
    return priors


def write_priors(frame: pd.DataFrame, destination: Path) -> None:
    write_dataframe(frame, destination)
