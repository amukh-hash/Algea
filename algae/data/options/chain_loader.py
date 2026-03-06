"""
EOD option chain loader — reads/writes Parquet partitioned by date + underlying.

Layout:
    {root}/options_chains/date=YYYY-MM-DD/underlying=SPY/chain.parquet
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import List, Optional

import pandas as pd

from algae.data.options.schema import validate_chain


class OptionChainLoader:
    """Persist and retrieve validated option chain snapshots."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root) / "options_chains"

    # -- paths ---------------------------------------------------------

    def _partition_dir(self, dt: date, underlying: str) -> Path:
        return self.root / f"date={dt.isoformat()}" / f"underlying={underlying}"

    def _parquet_path(self, dt: date, underlying: str) -> Path:
        return self._partition_dir(dt, underlying) / "chain.parquet"

    # -- public API ----------------------------------------------------

    def save(
        self,
        df: pd.DataFrame,
        dt: date,
        underlying: str,
        *,
        validate: bool = True,
    ) -> Path:
        """Validate and persist a chain snapshot.  Returns the written path."""
        if validate:
            validate_chain(df)
        path = self._parquet_path(dt, underlying)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False, engine="pyarrow")
        return path

    def load(
        self,
        dt: date,
        underlying: str,
        *,
        validate: bool = True,
    ) -> pd.DataFrame:
        """Load a chain snapshot, optionally re-validating."""
        path = self._parquet_path(dt, underlying)
        if not path.exists():
            raise FileNotFoundError(f"No chain for {underlying} on {dt}: {path}")
        df = pd.read_parquet(path, engine="pyarrow")
        if validate:
            validate_chain(df)
        return df

    def available_dates(self, underlying: str) -> List[date]:
        """Return sorted list of dates that have chain snapshots."""
        pattern = self.root / "date=*" / f"underlying={underlying}"
        dates: List[date] = []
        # glob for partition dirs
        for p in sorted(self.root.glob(f"date=*/underlying={underlying}/chain.parquet")):
            # extract date from parent path
            date_part = p.parent.parent.name  # "date=YYYY-MM-DD"
            if date_part.startswith("date="):
                dates.append(date.fromisoformat(date_part[5:]))
        return sorted(dates)

    def available_underlyings(self, dt: Optional[date] = None) -> List[str]:
        """Return sorted list of underlyings available (optionally for a date)."""
        if dt is not None:
            base = self.root / f"date={dt.isoformat()}"
            pattern = base / "underlying=*" / "chain.parquet"
        else:
            pattern = self.root / "date=*" / "underlying=*" / "chain.parquet"
        underlyings: set[str] = set()
        for p in self.root.glob(
            f"date={'*' if dt is None else dt.isoformat()}"
            + "/underlying=*/chain.parquet"
        ):
            # parent.name is "underlying=SPY"
            und_part = p.parent.name
            if und_part.startswith("underlying="):
                underlyings.add(und_part[11:])
        return sorted(underlyings)
