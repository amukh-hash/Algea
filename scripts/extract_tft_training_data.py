"""Extract and package /ES 1-min Databento bars into 5-min .npz training samples.

Loads the raw `.dbn` file (1-minute OHLCV), filters to outright front-month
contracts, resamples to 5-minute bars, and applies the strict 09:20 EST
temporal mask required by the TFT Core Reversal model.

**Data Flow:**
  1. Load .dbn → DataFrame (2.8M rows, 1-min bars)
  2. Filter out calendar spreads (contain '-')
  3. Build continuous front-month chain (highest-volume contract per day)
  4. Resample 1-min → 5-min (standard OHLCV aggregation)
  5. For each trading day:
     a. Slice overnight: 18:00 → 09:20 EST (184 bars of 5-min)
     b. Label: (Close_16:00 - Open_09:30) / Open_09:30
     c. Save as .npz per day

**09:20 Hard Cutoff Invariant:**
  If any bar after 09:20 leaks into ts_features, the model will overfit
  to the RTH open and lose money in live execution.

Usage
-----
    python scripts/extract_tft_training_data.py \\
        --output data_lake/tft_training/
"""
from __future__ import annotations

import argparse
import logging
import os
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

# Default path to the downloaded Databento .dbn file
DEFAULT_DBN_PATH = Path(
    r"C:\Users\crick\Documents\Workshop\Algaie\data_lake\futures"
    r"\OHLCV1m.ES.FUT[5yr]\glbx-mdp3-20210227-20260226.ohlcv-1m.dbn"
    r"\glbx-mdp3-20210227-20260226.ohlcv-1m.dbn"
)

# Overnight feature window constants
OVERNIGHT_START = time(18, 0)   # Previous day's globex open
FEATURE_CUTOFF = time(9, 20)    # HARD STOP — no bars after this
EXPECTED_BARS = 184             # 18:00→09:20 at 5-min = 184 bars


# ═══════════════════════════════════════════════════════════════════════
# Step 1: Load .dbn and Build Continuous 5-Min Bars
# ═══════════════════════════════════════════════════════════════════════

def load_and_resample(dbn_path: Path) -> pd.DataFrame:
    """Load 1-min .dbn, filter to front-month outrights, resample to 5-min.

    Parameters
    ----------
    dbn_path : Path
        Path to the Databento .dbn file.

    Returns
    -------
    DataFrame with 5-min OHLCV bars, EST timezone, sorted chronologically.
    """
    import databento as db

    logger.info("Loading .dbn file: %s", dbn_path)
    store = db.DBNStore.from_file(str(dbn_path))
    df = store.to_df()
    logger.info("Raw: %d rows, columns: %s", len(df), df.columns.tolist())

    # ── Filter out calendar spreads (contain '-') ────────────────────
    if "symbol" in df.columns:
        before = len(df)
        df = df[~df["symbol"].str.contains("-", na=False)].copy()
        logger.info(
            "Filtered spreads: %d → %d rows (removed %d spread rows)",
            before, len(df), before - len(df),
        )

    # ── Convert to EST ───────────────────────────────────────────────
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(EASTERN_TZ)
    df = df.sort_index()

    # ── Build continuous front-month chain ────────────────────────────
    # For each date, keep only the contract with highest total volume
    if "symbol" in df.columns:
        df["trade_date"] = df.index.date
        daily_vol = df.groupby(["trade_date", "symbol"])["volume"].sum()
        front_month = daily_vol.groupby(level="trade_date").idxmax()

        # Build a lookup: date → front-month symbol
        front_map = {date: sym for date, sym in front_month.values}

        # Filter to only front-month rows
        df["is_front"] = df.apply(
            lambda row: front_map.get(row["trade_date"]) == row["symbol"],
            axis=1,
        )
        before = len(df)
        df = df[df["is_front"]].copy()
        logger.info(
            "Front-month chain: %d → %d rows", before, len(df),
        )
        df = df.drop(columns=["trade_date", "is_front"], errors="ignore")

    # ── Resample 1-min → 5-min ───────────────────────────────────────
    ohlcv_5m = df.resample("5min").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna(subset=["open", "close"])

    logger.info(
        "Resampled 1m → 5m: %d bars (%s → %s)",
        len(ohlcv_5m), ohlcv_5m.index[0], ohlcv_5m.index[-1],
    )
    return ohlcv_5m


# ═══════════════════════════════════════════════════════════════════════
# Step 2: Feature Computation
# ═══════════════════════════════════════════════════════════════════════

def compute_overnight_features(bars: pd.DataFrame) -> np.ndarray:
    """Compute [log_return, volume_normalized, vwap_normalized].

    Parameters
    ----------
    bars : DataFrame with exactly 184 rows of OHLCV.

    Returns
    -------
    ndarray of shape [184, 3] — float32.
    """
    close = bars["close"].values.astype(np.float64)
    volume = bars["volume"].values.astype(np.float64)
    high = bars["high"].values.astype(np.float64)
    low = bars["low"].values.astype(np.float64)

    # Log returns
    log_returns = np.zeros(len(close), dtype=np.float64)
    log_returns[1:] = np.log(close[1:] / np.clip(close[:-1], 1e-8, None))

    # Volume normalized (z-score within window)
    vol_std = volume.std()
    vol_norm = (volume - volume.mean()) / max(vol_std, 1e-8)

    # VWAP normalized: typical price relative to close
    typical_price = (high + low + close) / 3.0
    vwap_norm = (typical_price - close) / np.clip(np.abs(close), 1e-8, None)

    return np.stack([log_returns, vol_norm, vwap_norm], axis=1).astype(np.float32)


def get_static_features(date: pd.Timestamp) -> np.ndarray:
    """[day_of_week, is_opex, macro_event_id] as float32."""
    dow = date.weekday()

    # OpEx: third Friday of the month
    month_start = date.replace(day=1)
    fridays = pd.date_range(month_start, periods=5, freq="WOM-3FRI")
    is_opex = 1 if date.date() in [f.date() for f in fridays] else 0

    return np.array([dow, is_opex, 0], dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════
# Step 3: Extraction Loop (09:20 Hard Cutoff)
# ═══════════════════════════════════════════════════════════════════════

def extract_training_samples(df: pd.DataFrame, output_dir: Path) -> int:
    """Extract one .npz per valid trading day.

    Returns count of valid samples written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Identify unique trading dates from RTH bars (09:30-16:00)
    rth_mask = (df.index.time >= time(9, 30)) & (df.index.time <= time(16, 0))
    trading_dates = sorted(set(df.loc[rth_mask].index.date))
    logger.info("Found %d potential trading dates", len(trading_dates))

    n_written = 0
    n_skipped_bars = 0
    n_skipped_label = 0
    n_lookahead_blocked = 0

    for trade_date in trading_dates:
        trade_dt = pd.Timestamp(trade_date, tz=EASTERN_TZ)

        # ── Slice Overnight Feature Window ───────────────────────────
        prev_day = trade_dt - timedelta(days=1)
        while prev_day.weekday() >= 5:
            prev_day -= timedelta(days=1)

        window_start = prev_day.replace(hour=18, minute=0, second=0)
        window_end = trade_dt.replace(hour=9, minute=20, second=0)

        overnight = df.loc[window_start:window_end].copy()

        # ── LOOKAHEAD GUARD ──────────────────────────────────────────
        # Only flag bars in the MORNING pre-open window (09:25-09:30)
        # that would constitute actual RTH open lookahead.
        # Evening bars (18:00+) are legitimate overnight features.
        contaminated = overnight[
            (overnight.index.time >= time(9, 25)) &
            (overnight.index.time < time(12, 0))  # Morning only
        ]
        if len(contaminated) > 0:
            n_lookahead_blocked += 1
            overnight = overnight[
                ~((overnight.index.time >= time(9, 25)) &
                  (overnight.index.time < time(12, 0)))
            ]

        # ── Bar count check (±5 tolerance for DST / holidays) ────────
        if len(overnight) > EXPECTED_BARS:
            overnight = overnight.iloc[-EXPECTED_BARS:]
        elif len(overnight) < EXPECTED_BARS - 5:
            n_skipped_bars += 1
            continue
        elif len(overnight) < EXPECTED_BARS:
            # Pad with forward-fill for minor gaps
            overnight = overnight.resample("5min").ffill()
            overnight = overnight.iloc[-EXPECTED_BARS:]

        if len(overnight) != EXPECTED_BARS:
            n_skipped_bars += 1
            continue

        # ── Compute Label ────────────────────────────────────────────
        rth_start = trade_dt.replace(hour=9, minute=30, second=0)
        rth_end = trade_dt.replace(hour=16, minute=0, second=0)

        open_bars = df.loc[
            (df.index >= rth_start) &
            (df.index < rth_start + timedelta(minutes=5))
        ]
        close_bars = df.loc[
            (df.index >= rth_end - timedelta(minutes=5)) &
            (df.index <= rth_end)
        ]

        if open_bars.empty or close_bars.empty:
            n_skipped_label += 1
            continue

        open_price = float(open_bars.iloc[0]["open"])
        close_price = float(close_bars.iloc[-1]["close"])

        if open_price <= 0:
            n_skipped_label += 1
            continue

        oc_return = (close_price - open_price) / open_price

        # ── Features ─────────────────────────────────────────────────
        ts_features = compute_overnight_features(overnight)

        # Gap proxy: 09:20 LTP vs prev 16:00 close
        ltp_0920 = float(overnight.iloc[-1]["close"])
        prev_close_bars = df.loc[
            (df.index >= prev_day.replace(hour=15, minute=55)) &
            (df.index <= prev_day.replace(hour=16, minute=0))
        ]
        prev_close = float(prev_close_bars.iloc[-1]["close"]) if not prev_close_bars.empty else ltp_0920
        gap_proxy = (ltp_0920 - prev_close) / max(abs(prev_close), 1e-8)

        static_features = get_static_features(trade_dt)
        obs_features = np.array(
            [gap_proxy, 0.0, 0.0, 0.0, 0.0], dtype=np.float32,
        )

        # ── Shape Validation ─────────────────────────────────────────
        assert ts_features.shape == (184, 3), f"ts: {ts_features.shape}"

        # ── Save ─────────────────────────────────────────────────────
        np.savez_compressed(
            output_dir / f"{trade_date}.npz",
            ts_features=ts_features,
            static_features=static_features,
            obs_features=obs_features,
            oc_return=np.float32(oc_return),
        )
        n_written += 1

    logger.info(
        "DONE: %d samples written, %d skipped (bars), "
        "%d skipped (label), %d lookahead blocks",
        n_written, n_skipped_bars, n_skipped_label, n_lookahead_blocked,
    )
    return n_written


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract TFT training data: 1-min .dbn → 5-min .npz",
    )
    parser.add_argument(
        "--dbn", type=str, default=str(DEFAULT_DBN_PATH),
        help="Path to Databento .dbn file (1-min OHLCV)",
    )
    parser.add_argument(
        "--output", type=str, default="data_lake/tft_training",
        help="Output directory for .npz files",
    )
    args = parser.parse_args()

    df_5m = load_and_resample(Path(args.dbn))
    n = extract_training_samples(df_5m, Path(args.output))
    logger.info("Pipeline complete. %d training samples in %s", n, args.output)
    logger.info(
        "Next: python scripts/train_tft_gap.py "
        "--data-dir %s --device cuda:0", args.output,
    )


if __name__ == "__main__":
    main()
