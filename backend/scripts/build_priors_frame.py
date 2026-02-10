"""
Build the canonical selector priors-frame from cached teacher outputs.

Joins 10d and 30d priors caches, computes forward targets, derived features,
z-scores, and agreement features.  Writes partitioned parquet:

    backend/data/selector/priors_frame/date=YYYY-MM-DD/part.parquet

Usage
-----
::

    python backend/scripts/build_priors_frame.py \\
        --target-horizon 5 \\
        --start-date 2024-01-02 --end-date 2024-01-31 \\
        --teacher-10d-run RUN-2026-02-09-175844 \\
        --teacher-30d-run RUN-2026-02-09-181337 \\
        --context-len 252
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import math
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from algaie.data.priors.feature_utils import build_features
from algaie.data.priors.selector_schema import (
    ALL_FRAME_COLS,
    META_COLS,
    MODEL_FEATURE_COLS,
    TARGET_COLS,
    TEACHER_PRIORS_COLS,
    feature_version_hash,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Trading calendar helpers
# ═══════════════════════════════════════════════════════════════════════════

def load_trading_dates_for_ticker(
    symbol: str,
    ticker_dir: Path,
) -> Optional[pd.Series]:
    """Load sorted trading dates for a ticker."""
    fpath = ticker_dir / f"{symbol}.parquet"
    if not fpath.exists():
        return None
    df = pd.read_parquet(fpath, columns=["date", "close"])
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values("date")
    return df.set_index("date")["close"]


def compute_forward_return(
    close_series: pd.Series,
    anchor_date: date,
    horizon: int,
) -> Optional[float]:
    """Compute log(close[t+H] / close[t]) where t+H is H trading days ahead.

    Parameters
    ----------
    close_series : pd.Series
        Index = date, values = close price.
    anchor_date : date
        The anchor date t.
    horizon : int
        Number of trading days ahead.

    Returns
    -------
    float or None if insufficient forward data.
    """
    if anchor_date not in close_series.index:
        return None
    future_dates = close_series.index[close_series.index > anchor_date]
    if len(future_dates) < horizon:
        return None
    price_t = close_series[anchor_date]
    price_tH = close_series[future_dates[horizon - 1]]
    if price_t <= 0 or price_tH <= 0:
        return None
    return float(np.log(price_tH / price_t))


def compute_forward_vol(
    close_series: pd.Series,
    anchor_date: date,
    horizon: int,
    annualize: bool = True,
) -> Optional[float]:
    """Compute realised volatility over [t+1, t+H].

    Uses daily log returns, then annualises by sqrt(252) if requested.
    """
    if anchor_date not in close_series.index:
        return None
    future_dates = close_series.index[close_series.index > anchor_date]
    if len(future_dates) < horizon:
        return None
    # Include anchor close for computing first return
    window_dates = [anchor_date] + list(future_dates[:horizon])
    window_prices = [close_series[d] for d in window_dates]
    if any(p <= 0 for p in window_prices):
        return None
    log_rets = [
        np.log(window_prices[i + 1] / window_prices[i])
        for i in range(len(window_prices) - 1)
    ]
    if len(log_rets) < 2:
        return None
    vol = float(np.std(log_rets, ddof=1))
    if annualize:
        vol *= np.sqrt(252)
    return vol


# ═══════════════════════════════════════════════════════════════════════════
# Main builder
# ═══════════════════════════════════════════════════════════════════════════

def load_cache_partition(
    cache_root: Path,
    teacher_horizon_tag: str,
    date_str: str,
) -> Optional[pd.DataFrame]:
    """Load one partition from the priors cache."""
    part_path = cache_root / f"teacher={teacher_horizon_tag}" / f"date={date_str}" / "part.parquet"
    if not part_path.exists():
        return None
    return pd.read_parquet(part_path)


def build_frame_for_date(
    date_val: date,
    cache_root: Path,
    ticker_dir: Path,
    target_horizon: int,
    teacher_10d_run: str,
    teacher_30d_run: str,
    context_len: int,
) -> Optional[pd.DataFrame]:
    """Build the full feature frame for a single date.

    Steps:
    1. Load 10d and 30d cache partitions.
    2. Inner join on (date, symbol).
    3. Compute forward targets.
    4. Compute derived + z-score + agreement features.
    5. Attach metadata.
    """
    date_str = date_val.isoformat()

    # 1. Load caches
    df_10 = load_cache_partition(cache_root, "10d", date_str)
    df_30 = load_cache_partition(cache_root, "30d", date_str)

    if df_10 is None or df_30 is None:
        return None

    # 2. Inner join
    join_cols = ["date", "symbol"]
    meta_10 = ["run_id", "context_len", "horizon_days"]
    meta_30 = ["run_id", "context_len", "horizon_days"]
    df = df_10.merge(df_30.drop(columns=meta_30, errors="ignore"),
                     on=join_cols, how="inner", suffixes=("", "_dup"))

    # Drop any accidental duplicates from merge
    dup_cols = [c for c in df.columns if c.endswith("_dup")]
    if dup_cols:
        df = df.drop(columns=dup_cols)

    if len(df) == 0:
        return None

    # 3. Compute forward targets
    # Cache close series for efficiency
    close_cache: Dict[str, pd.Series] = {}
    y_ret_list = []
    y_vol_list = []

    for _, row in df.iterrows():
        sym = row["symbol"]
        if sym not in close_cache:
            series = load_trading_dates_for_ticker(sym, ticker_dir)
            close_cache[sym] = series

        series = close_cache[sym]
        if series is None:
            y_ret_list.append(np.nan)
            y_vol_list.append(np.nan)
            continue

        y_ret = compute_forward_return(series, date_val, target_horizon)
        y_vol = compute_forward_vol(series, date_val, target_horizon)
        y_ret_list.append(y_ret if y_ret is not None else np.nan)
        y_vol_list.append(y_vol if y_vol is not None else np.nan)

    df["y_ret"] = y_ret_list
    df["y_vol"] = y_vol_list

    # Drop rows where we couldn't compute targets
    n_before = len(df)
    df = df.dropna(subset=["y_ret"])
    if len(df) == 0:
        return None
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        logger.debug(f"  Dropped {n_dropped} rows with missing targets")

    # Ensure date column is present for z-scoring
    df["date"] = date_val

    # 4. Derived + z-score + agreement features
    # Item C: z-scores computed ONLY on post-join survivors (both teachers present).
    # This is guaranteed because build_features() is called after the inner join above.
    n_pre_zscore = len(df)
    df = build_features(df)
    assert len(df) == n_pre_zscore, "build_features must not drop rows"

    # Item B: end-of-horizon magnitude sanity check
    if "q50_10" in df.columns and "q50_30" in df.columns:
        abs_10 = df["q50_10"].abs().median()
        abs_30 = df["q50_30"].abs().median()
        if abs_30 < abs_10 * 0.5:
            logger.warning(
                f"  ⚠ Horizon magnitude check: median|q50_30|={abs_30:.5f} < "
                f"0.5 * median|q50_10|={abs_10:.5f}. "
                f"Expected 30d returns to be broadly larger in abs terms."
            )
        else:
            logger.debug(f"  Horizon check OK: |q50_10|={abs_10:.5f}, |q50_30|={abs_30:.5f}")

    # 5. Attach metadata
    df["run_id_teacher_10d"] = teacher_10d_run
    df["run_id_teacher_30d"] = teacher_30d_run
    df["context_len"] = context_len
    df["horizon_ret"] = target_horizon
    df["feature_version_hash"] = feature_version_hash()

    # Item E: symbols metadata
    syms_sorted = sorted(df["symbol"].unique().tolist())
    df["universe_count"] = len(syms_sorted)
    df["symbols_hash"] = hashlib.sha256(
        ",".join(syms_sorted).encode()
    ).hexdigest()[:12]

    return df


def main():
    parser = argparse.ArgumentParser(description="Build selector priors frame")
    parser.add_argument("--target-horizon", type=int, default=5,
                        help="Forward return horizon in trading days")
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--teacher-10d-run", default="RUN-2026-02-09-175844")
    parser.add_argument("--teacher-30d-run", default="RUN-2026-02-09-181337")
    parser.add_argument("--context-len", type=int, default=252)
    parser.add_argument("--cache-root", default=None)
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()

    cache_root = Path(args.cache_root) if args.cache_root else (
        ROOT / "backend" / "data" / "selector" / "priors_cache"
    )
    output_root = Path(args.output_root) if args.output_root else (
        ROOT / "backend" / "data" / "selector" / "priors_frame"
    )

    ticker_dir = ROOT / "backend" / "data" / "canonical" / "per_ticker"

    start = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end = datetime.strptime(args.end_date, "%Y-%m-%d").date()

    # Discover dates from 10d cache (must exist for both teachers)
    cache_10d_dir = cache_root / "teacher=10d"
    cache_30d_dir = cache_root / "teacher=30d"

    if not cache_10d_dir.exists() or not cache_30d_dir.exists():
        logger.error("Cache directories not found. Run build_priors_cache.py first.")
        sys.exit(1)

    # Get dates from cache
    dates_10d = {
        d.name.replace("date=", "")
        for d in cache_10d_dir.iterdir()
        if d.is_dir() and d.name.startswith("date=")
    }
    dates_30d = {
        d.name.replace("date=", "")
        for d in cache_30d_dir.iterdir()
        if d.is_dir() and d.name.startswith("date=")
    }
    available_dates = sorted(dates_10d & dates_30d)

    # Filter to requested range
    available_dates = [
        d for d in available_dates
        if start <= datetime.strptime(d, "%Y-%m-%d").date() <= end
    ]
    logger.info(f"Dates with both caches in range: {len(available_dates)}")

    processed = 0
    skipped = 0
    for i, date_str in enumerate(available_dates):
        out_dir = output_root / f"date={date_str}"
        out_file = out_dir / "part.parquet"

        if out_file.exists():
            skipped += 1
            continue

        date_val = datetime.strptime(date_str, "%Y-%m-%d").date()
        logger.info(f"[{i+1}/{len(available_dates)}] {date_str}")

        df = build_frame_for_date(
            date_val=date_val,
            cache_root=cache_root,
            ticker_dir=ticker_dir,
            target_horizon=args.target_horizon,
            teacher_10d_run=args.teacher_10d_run,
            teacher_30d_run=args.teacher_30d_run,
            context_len=args.context_len,
        )

        if df is not None and len(df) > 0:
            out_dir.mkdir(parents=True, exist_ok=True)
            df.to_parquet(out_file, index=False)
            processed += 1
            logger.info(f"  → {len(df)} rows written")
        else:
            logger.warning(f"  → No valid rows for {date_str}")

    logger.info(f"Done. Processed={processed}, Skipped={skipped}")


if __name__ == "__main__":
    main()
