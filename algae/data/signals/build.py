from __future__ import annotations

from pathlib import Path

import pandas as pd

from algae.data.common import ensure_datetime, write_dataframe
from algae.data.signals.validate import validate_signal_frame


def build_signals(features: pd.DataFrame, priors: pd.DataFrame) -> pd.DataFrame:
    features = ensure_datetime(features.copy())
    priors = ensure_datetime(priors.copy())
    merged = features.merge(priors, on=["date", "ticker"], how="inner")
    merged["score"] = merged["ret_5d"].fillna(0) + merged["p_mu10"].fillna(0)
    merged["rank"] = merged.groupby("date")["score"].rank(ascending=False, method="first")
    signals = merged[["date", "ticker", "score", "rank"]].copy()
    validate_signal_frame(signals)
    return signals


def write_signals(frame: pd.DataFrame, destination: Path) -> None:
    write_dataframe(frame, destination)
