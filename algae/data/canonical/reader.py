from __future__ import annotations

from pathlib import Path

import polars as pl


def read_canonical_daily(path: Path) -> pl.DataFrame:
    return pl.scan_parquet(path).collect()
