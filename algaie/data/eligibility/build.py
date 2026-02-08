from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd

from algaie.data.eligibility.validate import validate_eligibility_frame


def build_eligibility(
    canonical_daily: pd.DataFrame,
    asof: date,
    min_price: float = 5.0,
    min_volume: float = 0.0,
) -> pd.DataFrame:
    canonical_daily = canonical_daily.copy()
    canonical_daily["date"] = pd.to_datetime(canonical_daily["date"])
    snapshot = canonical_daily[canonical_daily["date"] <= pd.Timestamp(asof)].copy()
    latest = snapshot.groupby("ticker").tail(1)
    eligible = (latest["close"] >= min_price) & (latest["volume"] >= min_volume)
    frame = pd.DataFrame(
        {
            "date": latest["date"],
            "ticker": latest["ticker"],
            "is_eligible": eligible.values,
            "reason_codes": ["price_volume"] * len(latest),
        }
    )
    validate_eligibility_frame(frame)
    return frame


def write_eligibility(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(destination, index=False)


def load_canonical_files(paths: Iterable[Path]) -> pd.DataFrame:
    frames = [pd.read_parquet(path) for path in paths]
    return pd.concat(frames, ignore_index=True)
