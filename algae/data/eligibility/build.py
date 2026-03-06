from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd

from algae.data.common import ensure_datetime, write_dataframe
from algae.data.eligibility.validate import validate_eligibility_frame


def build_eligibility(
    canonical_daily: pd.DataFrame,
    asof: date,
    min_price: float = 5.0,
    min_volume: float = 0.0,
) -> pd.DataFrame:
    canonical_daily = ensure_datetime(canonical_daily.copy())
    snapshot = canonical_daily[canonical_daily["date"] <= pd.Timestamp(asof)].copy()
    eligible = (snapshot["close"] >= min_price) & (snapshot["volume"] >= min_volume)
    frame = pd.DataFrame(
        {
            "date": snapshot["date"],
            "ticker": snapshot["ticker"],
            "is_eligible": eligible.values,
            "reason_codes": ["price_volume"] * len(snapshot),
        }
    )
    validate_eligibility_frame(frame)
    return frame


def write_eligibility(frame: pd.DataFrame, destination: Path) -> None:
    write_dataframe(frame, destination)


def load_canonical_files(paths: Iterable[Path]) -> pd.DataFrame:
    frames = [pd.read_parquet(path) for path in paths]
    return pd.concat(frames, ignore_index=True)
