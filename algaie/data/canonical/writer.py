from __future__ import annotations

from pathlib import Path

import pandas as pd

from algaie.data.common import write_dataframe


def write_canonical_daily(df: pd.DataFrame, destination: Path) -> None:
    write_dataframe(df, destination)
