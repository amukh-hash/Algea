"""Bronze data ingestion: fetch daily bars per contract/root and persist."""
from __future__ import annotations

import abc
import hashlib
import io
import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Sequence

import pandas as pd

from .types import BronzeManifest

# ---------------------------------------------------------------------------
# Required schema for daily bars
# ---------------------------------------------------------------------------

_REQUIRED_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")


def _enforce_schema(df: pd.DataFrame, root: str) -> pd.DataFrame:
    """Validate and coerce column dtypes for raw bars."""
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"[{root}] Missing required columns: {missing}")
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(int)
    return df


# ---------------------------------------------------------------------------
# Abstract provider
# ---------------------------------------------------------------------------

class FuturesDataProvider(abc.ABC):
    """Abstract interface for fetching daily futures bars."""

    @abc.abstractmethod
    def fetch_daily_bars(
        self, root: str, start: date, end: date
    ) -> pd.DataFrame:
        """Return a DataFrame with columns: timestamp, open, high, low, close, volume.

        All timestamps must be timezone-aware (UTC).
        """


# ---------------------------------------------------------------------------
# CSV / Parquet provider
# ---------------------------------------------------------------------------

class CsvDataProvider(FuturesDataProvider):
    """Read daily bars from local CSV or Parquet files.

    Expected directory layout::

        base_dir/{root}.csv   (or {root}.parquet)

    Each file must contain the required schema columns.
    """

    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)

    def fetch_daily_bars(
        self, root: str, start: date, end: date
    ) -> pd.DataFrame:
        parquet_path = self.base_dir / f"{root}.parquet"
        csv_path = self.base_dir / f"{root}.csv"

        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
        elif csv_path.exists():
            df = pd.read_csv(csv_path)
        else:
            raise FileNotFoundError(
                f"No data file for root {root} in {self.base_dir}"
            )

        df = _enforce_schema(df, root)
        mask = (df["timestamp"].dt.date >= start) & (df["timestamp"].dt.date <= end)
        return df.loc[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Bronze ingestion
# ---------------------------------------------------------------------------

def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_meta_sidecar(
    meta_path: Path,
    root: str,
    vendor: str,
    params: Dict[str, Any],
    retrieval_ts: str,
    checksum: str,
) -> None:
    meta = {
        "root": root,
        "vendor": vendor,
        "params": params,
        "retrieval_ts": retrieval_ts,
        "sha256": checksum,
    }
    meta_path.write_text(json.dumps(meta, sort_keys=True, indent=2))


def ingest_bronze(
    provider: FuturesDataProvider,
    roots: Sequence[str],
    start: date,
    end: date,
    bronze_dir: str | Path,
    vendor: str = "local",
) -> BronzeManifest:
    """Fetch daily bars for each root and persist as bronze parquet + sidecar.

    Parameters
    ----------
    provider : FuturesDataProvider
        Data source.
    roots : sequence of root symbols (sorted deterministically).
    start, end : date range (inclusive).
    bronze_dir : output directory.
    vendor : vendor tag for sidecar metadata.

    Returns
    -------
    BronzeManifest
    """
    bronze_dir = Path(bronze_dir)
    sorted_roots = sorted(roots)
    retrieval_ts = datetime.now(timezone.utc).isoformat()
    params: Dict[str, Any] = {
        "start": start.isoformat(),
        "end": end.isoformat(),
    }

    paths: Dict[str, str] = {}
    checksums: Dict[str, str] = {}

    for root in sorted_roots:
        root_dir = bronze_dir / root
        root_dir.mkdir(parents=True, exist_ok=True)

        df = provider.fetch_daily_bars(root, start, end)

        # Skip roots with no data (e.g. IBKR can't find contract defs)
        if df.empty:
            import logging
            logging.getLogger(__name__).warning(
                "Skipping root %s — no data returned by provider", root,
            )
            continue

        df = df.sort_values("timestamp").reset_index(drop=True)

        parquet_path = root_dir / "bars.parquet"
        buf = io.BytesIO()
        df.to_parquet(buf, index=False, engine="pyarrow")
        parquet_bytes = buf.getvalue()
        parquet_path.write_bytes(parquet_bytes)

        checksum = _sha256_bytes(parquet_bytes)
        checksums[root] = checksum
        paths[root] = str(parquet_path)

        _write_meta_sidecar(
            root_dir / "_meta.json",
            root=root,
            vendor=vendor,
            params=params,
            retrieval_ts=retrieval_ts,
            checksum=checksum,
        )

    ingested_roots = tuple(sorted(paths.keys()))
    return BronzeManifest(
        roots=ingested_roots,
        paths=paths,
        checksums=checksums,
        vendor=vendor,
        retrieval_ts=retrieval_ts,
        params=params,
    )
