"""
Data preparation for Chronos-2 training.

1. Splits raw_ohlcv.parquet → per-ticker parquet files.
2. Builds a real UniverseFrame using UniverseBuilder (is_observable / is_tradable / tier / weight).
3. Outputs a summary of the processed data.

Usage:
    python backend/scripts/prepare_training_data.py
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import pandas as pd

# Ensure project root is on path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from algaie.data.eligibility.universe_builder import UniverseBuilder, UniverseConfig
from backend.scripts._cli_utils import normalise_ohlcv_columns

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── Paths ──────────────────────────────────────────────────────────────────
RAW_OHLCV = ROOT / "backend" / "data" / "artifacts" / "universe" / "raw_ohlcv.parquet"
PER_TICKER_DIR = ROOT / "backend" / "data" / "canonical" / "per_ticker"
UNIVERSE_OUT = ROOT / "backend" / "data" / "canonical" / "universe_frame.parquet"


def split_ohlcv_to_per_ticker(ohlcv_path: Path, out_dir: Path) -> int:
    """Split long-form OHLCV into one parquet file per ticker. Returns count."""
    logger.info(f"Reading {ohlcv_path} ...")
    df = pd.read_parquet(ohlcv_path)
    df = normalise_ohlcv_columns(df)

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["symbol", "date"])

    out_dir.mkdir(parents=True, exist_ok=True)
    symbols = df["symbol"].unique()
    logger.info(f"Splitting {len(df):,} rows across {len(symbols):,} tickers → {out_dir}")

    t0 = time.time()
    for i, sym in enumerate(symbols):
        ticker_df = df[df["symbol"] == sym].copy()
        ticker_df.to_parquet(out_dir / f"{sym}.parquet", index=False)
        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            logger.info(f"  [{i+1}/{len(symbols)}] {elapsed:.1f}s")

    elapsed = time.time() - t0
    logger.info(f"Done — {len(symbols):,} files in {elapsed:.1f}s")
    return len(symbols)


def build_universe_frame(ohlcv_path: Path, out_path: Path) -> pd.DataFrame:
    """Build a proper UniverseFrame from raw OHLCV data."""
    logger.info(f"Building UniverseFrame from {ohlcv_path} ...")
    df = pd.read_parquet(ohlcv_path)
    df = normalise_ohlcv_columns(df)

    config = UniverseConfig(
        min_price=5.0,
        min_dollar_vol=1_000_000.0,
        min_history_days=60,
        tier_breakpoints=[50_000_000.0, 10_000_000.0],
    )
    builder = UniverseBuilder(config)
    universe = builder.build(df)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    universe.to_parquet(out_path, index=False)

    # Summary stats
    n_dates = universe["date"].nunique()
    n_symbols = universe["symbol"].nunique()
    n_observable = universe["is_observable"].sum()
    n_tradable = universe["is_tradable"].sum()
    total = len(universe)

    logger.info(f"UniverseFrame: {total:,} rows, {n_dates:,} dates, {n_symbols:,} symbols")
    logger.info(f"  Observable: {n_observable:,} ({100*n_observable/total:.1f}%)")
    logger.info(f"  Tradable:   {n_tradable:,} ({100*n_tradable/total:.1f}%)")
    logger.info(f"  Tier dist:  {universe['tier'].value_counts().to_dict()}")
    logger.info(f"Saved → {out_path}")

    return universe


def main() -> None:
    if not RAW_OHLCV.exists():
        logger.error(f"raw_ohlcv.parquet not found at {RAW_OHLCV}")
        sys.exit(1)

    # Step 1: Split per-ticker
    n_tickers = split_ohlcv_to_per_ticker(RAW_OHLCV, PER_TICKER_DIR)

    # Step 2: Build universe
    universe = build_universe_frame(RAW_OHLCV, UNIVERSE_OUT)

    # Step 3: Summary
    logger.info("=" * 60)
    logger.info("DATA PREP COMPLETE")
    logger.info(f"  Per-ticker files: {PER_TICKER_DIR} ({n_tickers} files)")
    logger.info(f"  Universe frame:   {UNIVERSE_OUT}")
    logger.info(f"  Observable mask entries: {universe['is_observable'].sum():,}")
    logger.info(f"  Tradable mask entries:   {universe['is_tradable'].sum():,}")
    logger.info("=" * 60)
    logger.info("Ready for ChronosDataset training!")


if __name__ == "__main__":
    main()
