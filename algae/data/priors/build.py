from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from algae.data.common import ensure_datetime, write_dataframe
from algae.data.priors.validate import validate_priors_frame
from algae.models.foundation.base import ModelProvider
from algae.models.foundation.chronos2 import FoundationModelConfig, SimpleChronos2


def build_priors(
    canonical_daily: pd.DataFrame,
    asof: date,
    config: FoundationModelConfig | None = None,
    provider: ModelProvider | None = None,
) -> pd.DataFrame:
    """Build distributional priors for all tickers as of ``asof``.

    Parameters
    ----------
    canonical_daily : pd.DataFrame
        Full canonical daily OHLCV history.
    asof : date
        Decision date.  Only data strictly **before** this date is used
        to prevent look-ahead bias.
    config : FoundationModelConfig | None
        Foundation model configuration.
    provider : ModelProvider | None
        Injectable model provider.  Pass ``StatisticalFallbackProvider()``
        in tests to avoid HuggingFace downloads.
    """
    model = SimpleChronos2(config or FoundationModelConfig(), provider=provider)
    canonical_daily = ensure_datetime(canonical_daily.copy())
    # Strict filtering: only data before asof (no day-of data)
    snapshot = canonical_daily[canonical_daily["date"] < pd.Timestamp(asof)].copy()
    priors = model.infer_priors(snapshot, pd.Timestamp(asof)).priors
    validate_priors_frame(priors)
    return priors


def write_priors(frame: pd.DataFrame, destination: Path) -> None:
    write_dataframe(frame, destination)
