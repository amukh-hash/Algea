from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_canonical_daily(df: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(destination, index=False)
