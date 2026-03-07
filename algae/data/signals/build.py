from __future__ import annotations

from pathlib import Path

import pandas as pd

from algae.data.common import ensure_datetime, write_dataframe
from algae.data.signals.validate import validate_signal_frame


def build_signals(features: pd.DataFrame, priors: pd.DataFrame) -> pd.DataFrame:
    """Build alpha signals by merging features with distributional priors.

    Both inputs must contain ``date`` and ``ticker`` columns.
    The priors frame must contain ``p_mu10`` (10-day expected return prior).
    """
    features = ensure_datetime(features.copy())
    priors = ensure_datetime(priors.copy()) if "date" in priors.columns else priors.copy()

    # Guard: verify p_mu10 exists before merge
    if "p_mu10" not in priors.columns:
        raise KeyError(
            f"priors frame missing 'p_mu10' column. "
            f"Available columns: {list(priors.columns)}"
        )

    # If priors don't have a date column, broadcast merge on ticker only
    merge_on = ["ticker"]
    if "date" in priors.columns:
        merge_on = ["date", "ticker"]

    merged = features.merge(priors, on=merge_on, how="inner")
    merged["score"] = merged["ret_5d"].fillna(0) + merged["p_mu10"].fillna(0)
    merged["rank"] = merged.groupby("date")["score"].rank(ascending=False, method="first")
    signals = merged[["date", "ticker", "score", "rank"]].copy()
    validate_signal_frame(signals)
    return signals


def write_signals(frame: pd.DataFrame, destination: Path) -> None:
    write_dataframe(frame, destination)
