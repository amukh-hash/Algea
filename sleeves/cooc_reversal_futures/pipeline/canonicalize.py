"""Canonicalize bronze bars → silver bars → gold daily frame.

CHANGE LOG (2026-02-14):
  - D2: build_gold_frame now asserts unique (root, trading_day) before computing
    returns.  Uses session_open_price/session_close_price if available, else
    falls back to open/close.  trading_day coerced to date.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

from ..contract_master import CONTRACT_MASTER, ContractSpec
from ..roll import _MONTH_CODE_MAP, active_contract_for_day, days_to_expiry_estimate
from .types import CanonicalizationManifest

ET = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# Timezone normalization
# ---------------------------------------------------------------------------

def normalize_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Convert timestamps to America/New_York and assign ``trading_day``.

    For daily bars whose UTC timestamps fall at midnight (00:00), the
    ``trading_day`` is taken from the *UTC date* (which represents the
    session date).  For intraday timestamps, the NY-localized date is
    used instead.
    """
    out = df.copy()
    ts = pd.to_datetime(out["timestamp"], utc=True)
    ts_ny = ts.dt.tz_convert(ET)
    out["timestamp"] = ts_ny

    # Daily bars typically have midnight-UTC timestamps; use UTC date to
    # avoid shifting the trading day backward when converting to ET.
    is_midnight = (ts.dt.hour == 0) & (ts.dt.minute == 0) & (ts.dt.second == 0)
    utc_dates = ts.dt.date
    ny_dates = ts_ny.dt.date
    out["trading_day"] = np.where(is_midnight, utc_dates, ny_dates)
    return out


# ---------------------------------------------------------------------------
# Contract map
# ---------------------------------------------------------------------------

def build_contract_map(
    roots: Sequence[str],
    start: date,
    end: date,
    contract_master: Dict[str, ContractSpec] | None = None,
) -> pd.DataFrame:
    """Build a deterministic ``(root, trading_day) -> active_contract`` map.

    Uses :func:`active_contract_for_day` from the roll module.

    Returns
    -------
    DataFrame with columns: root, trading_day, active_contract, days_to_expiry
    """
    cm = contract_master or CONTRACT_MASTER
    trading_days: List[date] = [
        d.date() for d in pd.bdate_range(start, end)
    ]

    rows: List[Dict[str, object]] = []
    for root in sorted(roots):
        spec = cm[root]
        for day in trading_days:
            contract = active_contract_for_day(root, day, spec)
            # Parse contract month/year from symbol suffix
            code_char = contract[len(root)]
            yr_str = contract[len(root) + 1:]
            month = _MONTH_CODE_MAP[code_char]
            year = 2000 + int(yr_str)
            dte = days_to_expiry_estimate(day, month, year)
            rows.append({
                "root": root,
                "trading_day": day,
                "active_contract": contract,
                "days_to_expiry": dte,
            })

    df = pd.DataFrame(rows)
    df = df.sort_values(["root", "trading_day"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Silver bars
# ---------------------------------------------------------------------------

def build_silver_bars(
    bronze_bars: pd.DataFrame,
    contract_map: pd.DataFrame,
) -> pd.DataFrame:
    """Merge active contract info into normalized daily bars per root.

    Parameters
    ----------
    bronze_bars : must have columns: root, trading_day, open, high, low, close, volume
    contract_map : from :func:`build_contract_map`

    Returns
    -------
    DataFrame with columns: trading_day, root, active_contract, days_to_expiry,
    open, high, low, close, volume.
    """
    merged = bronze_bars.merge(
        contract_map,
        on=["root", "trading_day"],
        how="inner",
    )
    keep = [
        "trading_day", "root", "active_contract", "days_to_expiry",
        "open", "high", "low", "close", "volume",
        # D1/D2: keep session columns if present
        "session_open_price", "session_close_price",
        "session_open_ts", "session_close_ts",
    ]
    merged = merged[[c for c in keep if c in merged.columns]]
    merged = merged.sort_values(["root", "trading_day"]).reset_index(drop=True)
    return merged


# ---------------------------------------------------------------------------
# Gold frame (derived returns)
# ---------------------------------------------------------------------------

def build_gold_frame(silver_bars: pd.DataFrame) -> pd.DataFrame:
    """Compute derived returns from silver bars.

    Uses ``session_open_price`` / ``session_close_price`` if available,
    otherwise falls back to ``open`` / ``close`` for backward compatibility.

    ``r_co[D] = open_price[D] / prev_session_close[D-1] - 1``
    ``r_oc[D] = close_price[D] / open_price[D] - 1``

    Shift logic is strictly per-root, sorted by (root, trading_day).

    Raises
    ------
    ValueError
        If duplicate (root, trading_day) rows are found.
    """
    df = silver_bars.copy()

    # --- Coerce trading_day to date ---
    if hasattr(df["trading_day"].dtype, "tz") or pd.api.types.is_datetime64_any_dtype(df["trading_day"]):
        df["trading_day"] = pd.to_datetime(df["trading_day"]).dt.date
    # If object dtype containing Timestamps, also convert
    if df["trading_day"].dtype == object:
        sample = df["trading_day"].iloc[0] if len(df) > 0 else None
        if hasattr(sample, "date") and callable(getattr(sample, "date", None)):
            df["trading_day"] = df["trading_day"].apply(
                lambda x: x.date() if hasattr(x, "date") and callable(x.date) else x
            )

    df = df.sort_values(["root", "trading_day"]).reset_index(drop=True)

    # --- D2: Duplicate guard ---
    dups = df.duplicated(subset=["root", "trading_day"], keep=False)
    if dups.any():
        n_dups = dups.sum()
        examples = df[dups][["root", "trading_day"]].head(10).to_string()
        raise ValueError(
            f"build_gold_frame: {n_dups} duplicate (root, trading_day) rows found. "
            f"Deduplicate upstream before computing returns.\n"
            f"Example duplicates:\n{examples}"
        )

    # --- D2: Use session prices if available, else fallback ---
    open_col = "session_open_price" if "session_open_price" in df.columns else "open"
    close_col = "session_close_price" if "session_close_price" in df.columns else "close"

    # Per-root lagged session close
    df["prev_close"] = df.groupby("root")[close_col].shift(1)

    df["ret_co"] = df[open_col] / df["prev_close"] - 1.0
    df["ret_oc"] = df[close_col] / df[open_col] - 1.0

    # Drop rows without a valid prev_close (first day per root)
    df = df.dropna(subset=["ret_co"]).reset_index(drop=True)
    df = df.drop(columns=["prev_close"])

    # --- Canonical aliases (backward-compatible) ---
    df["instrument"] = df["root"]
    df["r_co"] = df["ret_co"]
    df["r_oc"] = df["ret_oc"]

    return df


# ---------------------------------------------------------------------------
# Persist + manifest
# ---------------------------------------------------------------------------

def persist_canonicalized(
    silver: pd.DataFrame,
    gold: pd.DataFrame,
    contract_map: pd.DataFrame,
    output_dir: str | Path,
) -> CanonicalizationManifest:
    """Write silver, gold, and contract_map to *output_dir* and return manifest."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    silver_path = out / "silver_bars.parquet"
    gold_path = out / "gold_frame.parquet"
    cmap_path = out / "contract_map.parquet"

    silver.to_parquet(silver_path, index=False)
    gold.to_parquet(gold_path, index=False)
    contract_map.to_parquet(cmap_path, index=False)

    roots = tuple(sorted(silver["root"].unique()))
    trading_days = int(silver["trading_day"].nunique())

    return CanonicalizationManifest(
        roots=roots,
        silver_path=str(silver_path),
        gold_path=str(gold_path),
        contract_map_path=str(cmap_path),
        trading_days=trading_days,
        row_count=len(gold),
    )
