"""
Multi-Asset Overnight Gap Extraction for Chronos2 Deep Optimization.

Extracts overnight BPS gaps for ES, NQ, RTY, YM:
  - ES: Databento 1-min bars (2021-2026, 5 years)
  - NQ/RTY/YM:
      * yfinance daily cache (2021-2025) — approximate close/open
      * Databento 1-min bars (Jan-Mar 2026) — exact 15:58/09:30

Builds per-asset 32-day context windows then safely interleaves them
(preserving chronological ordering WITHOUT mixing assets within windows).

Output:
    data_lake/chronos2_training/
        {asset}_context_windows.npy  — per-asset [N, 32]
        {asset}_targets.npy          — per-asset [N]
        {asset}_metadata.npz         — dates for audit
        X_multi.npy                  — interleaved [~5000, 32]
        y_multi.npy                  — interleaved [~5000]
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)

EASTERN_TZ = ZoneInfo("America/New_York")
CONTEXT_LEN = 32

# ============================================================================
# Data source paths
# ============================================================================

ES_DBN_PATH = Path(
    r"C:\Users\crick\ResolveLabs\Algae\data_lake\futures"
    r"\OHLCV1m.ES.FUT[5yr]\glbx-mdp3-20210227-20260226.ohlcv-1m.dbn"
    r"\glbx-mdp3-20210227-20260226.ohlcv-1m.dbn"
)

NQ_RTY_YM_DBN_PATH = Path(
    r"C:\Users\crick\ResolveLabs\Algae\DataBento Kronos\Part 2"
    r"\glbx-mdp3-20260101-20260302.ohlcv-1m.dbn"
    r"\glbx-mdp3-20260101-20260302.ohlcv-1m.dbn"
)

YFINANCE_CACHE = Path(r"C:\Users\crick\ResolveLabs\Algae\data_lake\futures\yfinance_cache")


# ============================================================================
# Step 1: Extract gaps from Databento 1-min bars
# ============================================================================

def extract_gaps_from_dbn(dbn_path: Path, symbol_filter: str = None) -> pd.DataFrame:
    """Extract 15:58 Close / 09:30 Open pairs from Databento 1-min .dbn."""
    import databento as db

    logger.info("Loading .dbn: %s", dbn_path)
    store = db.DBNStore.from_file(str(dbn_path))
    df = store.to_df()
    logger.info("Raw: %d rows", len(df))

    # Filter spreads
    if "symbol" in df.columns:
        df = df[~df["symbol"].str.contains("-", na=False)].copy()

    # Filter to specific symbol root if requested (e.g., "NQ", "RTY", "YM")
    if symbol_filter and "symbol" in df.columns:
        mask = df["symbol"].str.startswith(symbol_filter)
        df = df[mask].copy()
        logger.info("Filtered to %s: %d rows", symbol_filter, len(df))

    # Convert to Eastern
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(EASTERN_TZ)
    df = df.sort_index()

    # Build front-month chain
    if "symbol" in df.columns:
        df["trade_date"] = df.index.date
        daily_vol = df.groupby(["trade_date", "symbol"])["volume"].sum()
        front_month = daily_vol.groupby(level="trade_date").idxmax()
        front_map = {date: sym for date, sym in front_month.values}
        df["is_front"] = df.apply(
            lambda row: front_map.get(row["trade_date"]) == row["symbol"],
            axis=1,
        )
        df = df[df["is_front"]].copy()
        df = df.drop(columns=["trade_date", "is_front"], errors="ignore")

    # Extract 15:58 close and 09:30 open
    close_mask = df.index.time == time(15, 58)
    close_bars = df.loc[close_mask, ["close"]].copy()
    close_bars["trade_date"] = close_bars.index.date
    close_bars = close_bars.groupby("trade_date").last()
    close_bars = close_bars.rename(columns={"close": "close_1558"})

    open_mask = df.index.time == time(9, 30)
    open_bars = df.loc[open_mask, ["open"]].copy()
    open_bars["trade_date"] = open_bars.index.date
    open_bars = open_bars.groupby("trade_date").first()
    open_bars = open_bars.rename(columns={"open": "open_0930"})

    # Match pairs
    close_dates = sorted(close_bars.index)
    open_dates = sorted(open_bars.index)

    records = []
    for open_date in open_dates:
        prev_closes = [d for d in close_dates if d < open_date]
        if not prev_closes:
            continue
        prev_close_date = prev_closes[-1]
        if (open_date - prev_close_date).days > 4:
            continue

        close_price = float(close_bars.loc[prev_close_date, "close_1558"])
        open_price = float(open_bars.loc[open_date, "open_0930"])
        if close_price <= 0:
            continue

        gap_bps = (open_price - close_price) / close_price * 10_000
        records.append({
            "date": open_date,
            "close_1558": close_price,
            "open_0930": open_price,
            "gap_bps": gap_bps,
        })

    result = pd.DataFrame(records)
    logger.info("Extracted %d gap pairs from .dbn", len(result))
    return result


# ============================================================================
# Step 2: Extract gaps from yfinance daily cache
# ============================================================================

def extract_gaps_from_yfinance(parquet_path: Path, start_date: str = "2021-01-01") -> pd.DataFrame:
    """Extract overnight BPS gaps from yfinance daily OHLCV cache.

    Uses daily Close ≈ 16:00 (proxy for 15:58) and Open ≈ 09:30 RTH.
    """
    df = pd.read_parquet(parquet_path)

    # Convert timestamp to date
    if "timestamp" in df.columns:
        df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    elif df.index.dtype == "datetime64[ns, UTC]":
        df["date"] = df.index.date
    else:
        raise ValueError(f"Cannot determine dates from {parquet_path}")

    # Filter to start_date onward
    from datetime import date as dt_date
    start = pd.Timestamp(start_date).date()
    df = df[df["date"] >= start].sort_values("date").reset_index(drop=True)

    records = []
    for i in range(1, len(df)):
        prev_row = df.iloc[i - 1]
        curr_row = df.iloc[i]

        # Gap check: weekends/holidays
        gap_days = (curr_row["date"] - prev_row["date"]).days
        if gap_days > 4:
            continue

        close_price = float(prev_row["close"])  # ≈ 16:00 (proxy for 15:58)
        open_price = float(curr_row["open"])    # ≈ 09:30

        if close_price <= 0:
            continue

        gap_bps = (open_price - close_price) / close_price * 10_000
        records.append({
            "date": curr_row["date"],
            "close_1558": close_price,
            "open_0930": open_price,
            "gap_bps": gap_bps,
        })

    result = pd.DataFrame(records)
    logger.info("Extracted %d gap pairs from yfinance (%s)", len(result), parquet_path.stem)
    return result


# ============================================================================
# Step 3: Combine yfinance (historical) + Databento (recent) for an asset
# ============================================================================

def extract_asset_gaps(
    asset: str,
    yf_path: Path | None,
    dbn_path: Path | None,
    dbn_symbol_filter: str | None,
    start_date: str = "2021-01-01",
    dbn_cutover: str = "2026-01-01",
) -> pd.DataFrame:
    """Extract and merge gaps from yfinance + Databento for a single asset."""
    parts = []

    # Historical from yfinance
    if yf_path and yf_path.exists():
        yf_gaps = extract_gaps_from_yfinance(yf_path, start_date)
        # Cut off before Databento starts
        cutover = pd.Timestamp(dbn_cutover).date()
        yf_gaps = yf_gaps[yf_gaps["date"] < cutover]
        logger.info("  %s yfinance: %d gaps (before %s)", asset, len(yf_gaps), dbn_cutover)
        parts.append(yf_gaps)

    # Recent from Databento
    if dbn_path and dbn_path.exists():
        dbn_gaps = extract_gaps_from_dbn(dbn_path, symbol_filter=dbn_symbol_filter)
        logger.info("  %s Databento: %d gaps", asset, len(dbn_gaps))
        parts.append(dbn_gaps)

    if not parts:
        raise FileNotFoundError(f"No data sources found for {asset}")

    combined = pd.concat(parts, ignore_index=True)
    combined = combined.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
    logger.info("  %s combined: %d gaps (%s → %s)",
                asset, len(combined),
                combined["date"].iloc[0], combined["date"].iloc[-1])
    return combined


# ============================================================================
# Step 4: Build per-asset context windows
# ============================================================================

def build_per_asset_windows(gaps_df: pd.DataFrame, context_len: int = CONTEXT_LEN):
    """Build [N, 32] context windows and [N] targets for one asset."""
    gap_series = gaps_df["gap_bps"].values.astype(np.float32)
    dates = gaps_df["date"].values

    n = len(gap_series) - context_len
    if n <= 0:
        raise ValueError(f"Not enough data: {len(gap_series)} < {context_len}")

    windows = np.zeros((n, context_len), dtype=np.float32)
    targets = np.zeros(n, dtype=np.float32)
    target_dates = []

    for i in range(n):
        windows[i] = gap_series[i : i + context_len]
        targets[i] = gap_series[i + context_len]
        target_dates.append(dates[i + context_len])

    assert not np.any(np.isnan(windows)), "NaN in windows"
    assert not np.any(np.isnan(targets)), "NaN in targets"

    return windows, targets, np.array(target_dates)


# ============================================================================
# Step 5: Full pipeline
# ============================================================================

def run_multiplexed_extraction(output_dir: str = "data_lake/chronos2_training"):
    """Extract, window, and interleave all 4 futures assets."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Asset configs: (symbol, yfinance_path, dbn_path, dbn_symbol_filter, dbn_cutover)
    # ES uses full 5-year .dbn (no yfinance needed)
    # NQ/RTY/YM use yfinance (2021-2025) + Databento (2026)
    asset_configs = [
        {
            "asset": "ES",
            "yf_path": None,  # Full .dbn covers 2021-2026
            "dbn_path": ES_DBN_PATH,
            "dbn_symbol_filter": None,
            "dbn_cutover": "2021-01-01",
        },
        {
            "asset": "NQ",
            "yf_path": YFINANCE_CACHE / "NQ_daily.parquet",
            "dbn_path": NQ_RTY_YM_DBN_PATH,
            "dbn_symbol_filter": "NQ",
            "dbn_cutover": "2026-01-01",
        },
        {
            "asset": "RTY",
            "yf_path": YFINANCE_CACHE / "RTY_daily.parquet",
            "dbn_path": NQ_RTY_YM_DBN_PATH,
            "dbn_symbol_filter": "RTY",
            "dbn_cutover": "2026-01-01",
        },
        {
            "asset": "YM",
            "yf_path": YFINANCE_CACHE / "YM_daily.parquet",
            "dbn_path": NQ_RTY_YM_DBN_PATH,
            "dbn_symbol_filter": "YM",
            "dbn_cutover": "2026-01-01",
        },
    ]

    all_windows, all_targets, all_dates = [], [], []

    for cfg in asset_configs:
        asset = cfg["asset"]
        logger.info("=" * 60)
        logger.info("Processing %s...", asset)

        # Extract gaps
        gaps_df = extract_asset_gaps(
            asset=asset,
            yf_path=cfg.get("yf_path"),
            dbn_path=cfg.get("dbn_path"),
            dbn_symbol_filter=cfg.get("dbn_symbol_filter"),
            dbn_cutover=cfg.get("dbn_cutover", "2026-01-01"),
        )

        # Build per-asset windows (CRITICAL: each window is pure single-asset)
        windows, targets, target_dates = build_per_asset_windows(gaps_df)

        # Save per-asset
        np.save(out / f"{asset}_context_windows.npy", windows)
        np.save(out / f"{asset}_targets.npy", targets)
        np.savez_compressed(
            out / f"{asset}_metadata.npz",
            target_dates=np.array([str(d) for d in target_dates]),
            gap_bps=gaps_df["gap_bps"].values,
        )

        logger.info(
            "  %s: %d windows [%d, %d], gap_bps: mean=%.1f std=%.1f",
            asset, len(windows), *windows.shape,
            gaps_df["gap_bps"].mean(), gaps_df["gap_bps"].std(),
        )

        all_windows.append(windows)
        all_targets.append(targets)
        all_dates.append(np.array([str(d) for d in target_dates]))

    # Interleave: concatenate then sort by target date
    X_cat = np.concatenate(all_windows, axis=0)
    y_cat = np.concatenate(all_targets, axis=0)
    d_cat = np.concatenate(all_dates, axis=0)

    sort_idx = np.argsort(d_cat)
    X_multi = X_cat[sort_idx]
    y_multi = y_cat[sort_idx]

    np.save(out / "X_multi.npy", X_multi)
    np.save(out / "y_multi.npy", y_multi)

    logger.info("=" * 60)
    logger.info(
        "MULTIPLEXED DATA: X=%s, y=%s",
        X_multi.shape, y_multi.shape,
    )
    logger.info(
        "BPS stats: mean=%.1f std=%.1f min=%.1f max=%.1f",
        y_multi.mean(), y_multi.std(), y_multi.min(), y_multi.max(),
    )


if __name__ == "__main__":
    run_multiplexed_extraction()
