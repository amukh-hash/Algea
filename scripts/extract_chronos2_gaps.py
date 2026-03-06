"""
Extract ES Futures Overnight Gap Dataset for Chronos2 LoRA Training.

Loads the Databento 1-minute .dbn file, extracts:
    - 15:58 ET Close (last 1-min bar before RTH close)
    - 09:30 ET Open (first RTH bar)

and computes the overnight gap target in **Basis Points (BPS)**:

    y = (Open_09:30 − Close_15:58) / Close_15:58 × 10,000

A 0.5% gap → 50.0 BPS (healthy gradient range).

Builds 32-day rolling context windows of BPS gaps for the Chronos2
model, which predicts the next-day gap distribution.

Output
------
    data_lake/chronos2_training/
        overnight_gaps.npy   — [N_days] raw BPS gap series
        context_windows.npy  — [N_samples, 32] sliding context windows
        targets.npy          — [N_samples] next-day BPS gap target
        metadata.npz         — dates, close_prices, open_prices for audit

Data Integrity
--------------
    - Uses real Databento 1-min bars (no yfinance approximations)
    - Continuous front-month chain via highest-volume contract per day
    - Strict timezone handling (Eastern Time)
    - No future data leakage: context window ends at day T, target is day T+1
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

# Databento .dbn file path (updated for new project location)
DEFAULT_DBN_PATH = Path(
    r"C:\Users\crick\ResolveLabs\Algae\data_lake\futures"
    r"\OHLCV1m.ES.FUT[5yr]\glbx-mdp3-20210227-20260226.ohlcv-1m.dbn"
    r"\glbx-mdp3-20210227-20260226.ohlcv-1m.dbn"
)

CONTEXT_LEN = 32  # Days of gap history per sample


# ═══════════════════════════════════════════════════════════════════════════
# Step 1: Load .dbn and Build Continuous 1-Min Front-Month Chain
# ═══════════════════════════════════════════════════════════════════════════

def load_es_bars(dbn_path: Path) -> pd.DataFrame:
    """Load 1-min ES bars from Databento .dbn and build continuous chain.

    Returns
    -------
    pd.DataFrame
        1-min OHLCV bars in Eastern Time, front-month only.
    """
    import databento as db

    logger.info("Loading .dbn file: %s (this may take 30-60 seconds)", dbn_path)
    store = db.DBNStore.from_file(str(dbn_path))
    df = store.to_df()
    logger.info("Raw: %d rows", len(df))

    # Filter out calendar spreads
    if "symbol" in df.columns:
        df = df[~df["symbol"].str.contains("-", na=False)].copy()
        logger.info("After spread filter: %d rows", len(df))

    # Convert to Eastern Time
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(EASTERN_TZ)
    df = df.sort_index()

    # Build continuous front-month chain (highest volume per day)
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
        logger.info("Front-month chain: %d rows", len(df))

    return df


# ═══════════════════════════════════════════════════════════════════════════
# Step 2: Extract 15:58 Close and 09:30 Open
# ═══════════════════════════════════════════════════════════════════════════

def extract_close_open_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Extract the 15:58 ET Close and next-day 09:30 ET Open.

    Parameters
    ----------
    df : pd.DataFrame
        1-min OHLCV bars in Eastern Time.

    Returns
    -------
    pd.DataFrame
        Columns: date, close_1558, open_0930, gap_bps
    """
    # Extract 15:58 bars (last 1-min bar before RTH close at 16:00)
    close_mask = df.index.time == time(15, 58)
    close_bars = df.loc[close_mask, ["close"]].copy()
    close_bars["trade_date"] = close_bars.index.date
    # If multiple bars at 15:58, take the last one
    close_bars = close_bars.groupby("trade_date").last()
    close_bars = close_bars.rename(columns={"close": "close_1558"})

    # Extract 09:30 bars (first RTH bar)
    open_mask = df.index.time == time(9, 30)
    open_bars = df.loc[open_mask, ["open"]].copy()
    open_bars["trade_date"] = open_bars.index.date
    # If multiple bars at 09:30, take the first one
    open_bars = open_bars.groupby("trade_date").first()
    open_bars = open_bars.rename(columns={"open": "open_0930"})

    logger.info(
        "Found %d close bars (15:58) and %d open bars (09:30)",
        len(close_bars), len(open_bars),
    )

    # Merge: for each open_0930 on date D, find close_1558 on date D-1
    # Shift close_bars forward by 1 trading day
    close_dates = sorted(close_bars.index)
    open_dates = sorted(open_bars.index)

    records = []
    for i, open_date in enumerate(open_dates):
        # Find the most recent close date before this open date
        prev_closes = [d for d in close_dates if d < open_date]
        if not prev_closes:
            continue

        prev_close_date = prev_closes[-1]

        # Validate: the gap should be at most 4 calendar days (weekend + holiday)
        gap_days = (open_date - prev_close_date).days
        if gap_days > 4:
            logger.debug(
                "Skipping %s: gap to prev close (%s) is %d days",
                open_date, prev_close_date, gap_days,
            )
            continue

        close_price = float(close_bars.loc[prev_close_date, "close_1558"])
        open_price = float(open_bars.loc[open_date, "open_0930"])

        if close_price <= 0:
            continue

        # BPS gap: (Open - Close) / Close × 10,000
        gap_bps = (open_price - close_price) / close_price * 10_000

        records.append({
            "date": open_date,
            "prev_close_date": prev_close_date,
            "close_1558": close_price,
            "open_0930": open_price,
            "gap_bps": gap_bps,
        })

    result = pd.DataFrame(records)
    logger.info("Extracted %d valid overnight gap pairs", len(result))

    # Validation statistics
    if len(result) > 0:
        gaps = result["gap_bps"]
        logger.info(
            "Gap BPS stats: mean=%.1f  std=%.1f  min=%.1f  max=%.1f  "
            "median=%.1f  |gap|>50bps: %d/%d (%.1f%%)",
            gaps.mean(), gaps.std(), gaps.min(), gaps.max(), gaps.median(),
            (gaps.abs() > 50).sum(), len(gaps),
            100 * (gaps.abs() > 50).sum() / len(gaps),
        )

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Step 3: Build 32-Day Context Windows
# ═══════════════════════════════════════════════════════════════════════════

def build_context_windows(
    gaps_df: pd.DataFrame,
    context_len: int = CONTEXT_LEN,
) -> tuple[np.ndarray, np.ndarray, list]:
    """Build sliding context windows from the BPS gap series.

    For each sample at index t (where t >= context_len):
        context = gaps[t-context_len : t]     (32 historical BPS gaps)
        target  = gaps[t]                     (next-day BPS gap)

    NO LOOKAHEAD: context window ends at day T-1, target is day T.

    Parameters
    ----------
    gaps_df : pd.DataFrame
        Must have column ``gap_bps`` sorted chronologically.
    context_len : int
        Number of historical days in each context window.

    Returns
    -------
    contexts : np.ndarray
        Shape ``[num_samples, context_len]``.
    targets : np.ndarray
        Shape ``[num_samples]``.
    dates : list
        Target dates for each sample.
    """
    gap_series = gaps_df["gap_bps"].values.astype(np.float32)
    all_dates = gaps_df["date"].values

    num_samples = len(gap_series) - context_len
    if num_samples <= 0:
        raise ValueError(
            f"Not enough data: {len(gap_series)} gaps, need > {context_len}"
        )

    contexts = np.zeros((num_samples, context_len), dtype=np.float32)
    targets = np.zeros(num_samples, dtype=np.float32)
    dates = []

    for i in range(num_samples):
        contexts[i] = gap_series[i : i + context_len]
        targets[i] = gap_series[i + context_len]
        dates.append(all_dates[i + context_len])

    logger.info(
        "Built %d context windows: [%d, %d] → target",
        num_samples, num_samples, context_len,
    )

    # Validate: no NaN, no Inf
    assert not np.any(np.isnan(contexts)), "NaN in contexts!"
    assert not np.any(np.isnan(targets)), "NaN in targets!"
    assert not np.any(np.isinf(contexts)), "Inf in contexts!"
    assert not np.any(np.isinf(targets)), "Inf in targets!"

    return contexts, targets, dates


# ═══════════════════════════════════════════════════════════════════════════
# Step 4: Full Pipeline
# ═══════════════════════════════════════════════════════════════════════════

def run_extraction(
    dbn_path: str = str(DEFAULT_DBN_PATH),
    output_dir: str = "data_lake/chronos2_training",
    context_len: int = CONTEXT_LEN,
) -> dict:
    """End-to-end overnight gap extraction pipeline.

    Returns
    -------
    dict
        Pipeline summary with shapes, paths, and validation stats.
    """
    dbn = Path(dbn_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Load 1-min bars
    df = load_es_bars(dbn)

    # 2. Extract 15:58 Close / 09:30 Open pairs
    gaps_df = extract_close_open_pairs(df)

    # 3. Build context windows
    contexts, targets, dates = build_context_windows(gaps_df, context_len)

    # 4. Save
    np.save(out / "overnight_gaps.npy", gaps_df["gap_bps"].values.astype(np.float32))
    np.save(out / "context_windows.npy", contexts)
    np.save(out / "targets.npy", targets)
    np.savez_compressed(
        out / "metadata.npz",
        dates=np.array([str(d) for d in gaps_df["date"].values]),
        close_prices=gaps_df["close_1558"].values.astype(np.float64),
        open_prices=gaps_df["open_0930"].values.astype(np.float64),
        gap_bps=gaps_df["gap_bps"].values.astype(np.float32),
        context_dates=np.array([str(d) for d in dates]),
    )

    # Also save the raw gaps DataFrame for audit
    gaps_df.to_parquet(out / "overnight_gaps.parquet")

    summary = {
        "status": "ok",
        "total_trading_days": len(gaps_df),
        "context_windows_shape": list(contexts.shape),
        "targets_shape": list(targets.shape),
        "date_range": [str(gaps_df["date"].iloc[0]), str(gaps_df["date"].iloc[-1])],
        "gap_bps_mean": float(gaps_df["gap_bps"].mean()),
        "gap_bps_std": float(gaps_df["gap_bps"].std()),
        "gap_bps_min": float(gaps_df["gap_bps"].min()),
        "gap_bps_max": float(gaps_df["gap_bps"].max()),
        "output_dir": str(out),
    }
    logger.info("Extraction complete: %s", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract ES overnight gap dataset for Chronos2 LoRA",
    )
    parser.add_argument(
        "--dbn", type=str, default=str(DEFAULT_DBN_PATH),
        help="Path to Databento .dbn file",
    )
    parser.add_argument(
        "--output", type=str, default="data_lake/chronos2_training",
        help="Output directory",
    )
    parser.add_argument(
        "--context-len", type=int, default=CONTEXT_LEN,
        help="Context window length in trading days",
    )
    args = parser.parse_args()

    result = run_extraction(
        dbn_path=args.dbn,
        output_dir=args.output,
        context_len=args.context_len,
    )
    print(result)
