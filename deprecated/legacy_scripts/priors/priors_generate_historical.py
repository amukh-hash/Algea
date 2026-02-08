import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))

import logging
import argparse
import pandas as pd
import os
from typing import List
from backend.app.ops import bootstrap, pathmap, config
from backend.app.teacher import priors, chronos_runner
from backend.app.data import security_master, calendar
from backend.app.data.ingest import ohlcv_daily as ingest_daily

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2016-01-01")
    parser.add_argument("--end", default="2026-02-03")
    parser.add_argument("--context", type=int, default=config.CONTEXT_LEN)
    parser.add_argument("--horizon", type=int, default=config.PRIOR_HORIZON)
    args = parser.parse_args()

    bootstrap.ensure_dirs()

    spec = {
        "model": "chronos-t5-small",
        "context": args.context,
        "horizon": args.horizon
    }

    # Chronos should be pre-calibrated and frozen before running priors generation.
    runner = chronos_runner.ChronosRunner(
        model_id=spec["model"],
        context_len=spec["context"],
        horizon=spec["horizon"]
    )

    start_dt = pd.to_datetime(args.start)
    end_dt = pd.to_datetime(args.end)
    trading_days = calendar.get_trading_days(start_dt, end_dt)
    if not trading_days:
        logger.error("No trading days found in range.")
        return

    sec_master = security_master.load_security_master()
    default_symbols = sec_master["symbol"].dropna().astype(str).unique().tolist()

    def load_union_manifest_symbols() -> List[str]:
        manifest_root = pathmap.get_paths().manifests
        if not os.path.exists(manifest_root):
            return []
        manifests = [p for p in os.listdir(manifest_root) if p.startswith("universe_asof=")]
        if not manifests:
            return []
        symbols = []
        for m in manifests:
            try:
                df = pd.read_parquet(os.path.join(manifest_root, m))
                symbols.extend(df["symbol"].dropna().astype(str).tolist())
            except Exception:
                continue
        return sorted(set(symbols))

    union_symbols = load_union_manifest_symbols()

    def load_series(symbol: str, asof_date: pd.Timestamp) -> pd.DataFrame:
        df = ingest_daily.load_ohlcv(symbol, end_date=asof_date)
        if df.empty:
            return df
        return df.tail(spec["context"])

    for current in trading_days:
        date_str = pd.Timestamp(current).strftime("%Y-%m-%d")
        logger.info(f"Generating Priors for {date_str}...")

        manifest_path = pathmap.resolve("manifest", date=current)
        if os.path.exists(manifest_path):
            manifest = pd.read_parquet(manifest_path)
            symbols = manifest.loc[manifest["eligible"] == True, "symbol"].astype(str).tolist()
        else:
            symbols = union_symbols or default_symbols

        if not symbols:
            logger.warning(f"  No symbols for {date_str}. Skipping.")
            continue

        try:
            df = priors.generate_priors_for_date(current, symbols, spec, runner, load_fn=load_series)
            coverage = len(df["symbol"].unique()) / max(len(symbols), 1)
            if coverage < config.PRIORS_MIN_COVERAGE:
                logger.warning(f"  Coverage {coverage:.2%} below threshold, skipping write.")
                continue

            path = priors.write_priors(current, df, spec)
            logger.info(f"  Written to {path}")

        except Exception as e:
            logger.error(f"  Failed for {date_str}: {e}")
            if config.FAIL_ON_MISSING_DIRS:
                raise


if __name__ == "__main__":
    main()
