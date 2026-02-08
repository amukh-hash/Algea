from __future__ import annotations

from pathlib import Path

import pandas as pd

from algaie.data.signals.validate import validate_signal_frame


def build_signals(features: pd.DataFrame, priors: pd.DataFrame) -> pd.DataFrame:
    features = features.copy()
    priors = priors.copy()
    features["date"] = pd.to_datetime(features["date"])
    priors["date"] = pd.to_datetime(priors["date"])
    merged = features.merge(priors, on=["date", "ticker"], how="inner")
    merged["score"] = merged["ret_5d"].fillna(0) + merged["p_mu10"].fillna(0)
    merged["rank"] = merged.groupby("date")["score"].rank(ascending=False, method="first")
    signals = merged[["date", "ticker", "score", "rank"]].copy()
    validate_signal_frame(signals)
    return signals


def write_signals(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(destination, index=False)
