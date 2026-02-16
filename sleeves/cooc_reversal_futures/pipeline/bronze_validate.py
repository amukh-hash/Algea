"""Bronze bar validation: monotonic ts, no dups, OHLC sanity, volume, gaps."""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from .types import BronzeValidationReport


def _business_day_gaps(
    dates: pd.Series,
) -> List[Tuple[str, str, int]]:
    """Identify calendar gaps larger than 1 business day."""
    if len(dates) < 2:
        return []
    unique_dates = sorted(dates.unique())
    gaps: List[Tuple[str, str, int]] = []
    for i in range(1, len(unique_dates)):
        d0 = pd.Timestamp(unique_dates[i - 1])
        d1 = pd.Timestamp(unique_dates[i])
        bdays = len(pd.bdate_range(d0, d1, inclusive="neither"))
        if bdays > 0:
            gaps.append((str(d0.date()), str(d1.date()), int(bdays)))
    return gaps


def validate_bronze_bars(
    df: pd.DataFrame,
    root: str,
) -> BronzeValidationReport:
    """Run all bronze quality checks on a daily-bar DataFrame.

    Parameters
    ----------
    df : DataFrame with columns timestamp, open, high, low, close, volume.
    root : root symbol (for reporting).

    Returns
    -------
    BronzeValidationReport
    """
    violations: List[str] = []

    # --- monotonic timestamps ---
    ts = df["timestamp"]
    monotonic = bool(ts.is_monotonic_increasing)
    if not monotonic:
        violations.append("timestamps not monotonically increasing")

    # --- no duplicates ---
    no_dups = bool(not ts.duplicated().any())
    if not no_dups:
        n_dups = int(ts.duplicated().sum())
        violations.append(f"{n_dups} duplicate timestamps")

    # --- OHLC sanity ---
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    ohlc_ok = bool(
        (l <= o).all()
        and (l <= c).all()
        and (l <= h).all()
        and (h >= o).all()
        and (h >= c).all()
    )
    if not ohlc_ok:
        bad_low = int(((l > o) | (l > c) | (l > h)).sum())
        bad_high = int(((h < o) | (h < c)).sum())
        violations.append(f"OHLC violations: {bad_low} low, {bad_high} high")

    # --- non-negative volume ---
    vol_ok = bool((df["volume"] >= 0).all())
    if not vol_ok:
        n_neg = int((df["volume"] < 0).sum())
        violations.append(f"{n_neg} negative volume rows")

    # --- gap report ---
    trading_dates = df["timestamp"].dt.date
    gap_report = _business_day_gaps(trading_dates)

    ok = monotonic and no_dups and ohlc_ok and vol_ok

    return BronzeValidationReport(
        root=root,
        ok=ok,
        monotonic_ts=monotonic,
        no_duplicates=no_dups,
        ohlc_sane=ohlc_ok,
        non_negative_volume=vol_ok,
        gap_report=tuple(tuple(g) for g in gap_report),  # type: ignore[arg-type]
        row_count=len(df),
        violations=tuple(violations),
    )


def persist_validation_report(
    report: BronzeValidationReport,
    output_dir: str | Path,
) -> Path:
    """Write validation report as JSON and return the path."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{report.root}_validation.json"
    path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return path
