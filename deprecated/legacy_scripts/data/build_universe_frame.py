
import sys
import os
import argparse
import logging
import json
import pandas as pd
import polars as pl
from pathlib import Path
from backend.app.ops import bootstrap, run_recorder, pathmap, artifact_registry
from backend.app.data.universe_config import UniverseRules
from backend.app.data.universe_frame import UniverseBuilder
from backend.app.data import schema_contracts as sc

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Build Rolling UniverseFrame (v2)")
    parser.add_argument("--ohlcv_path", default="backend/data_canonical/ohlcv_adj") # Directory of partitioned parquet or single file? usually partitioned per ticker
    # Actually, daily_parquet is per-ticker. 
    # Or raw_ohlcv.parquet global file?
    # Prefer raw_ohlcv global file for polars efficiency if available.
    # Otherwise scan parquet dir.
    # checking default paths... usually "backend/data/artifacts/universe/raw_ohlcv.parquet" or similar.
    # Let's default to finding raw_ohlcv if exists, or scan daily_parquet.
    parser.add_argument("--metadata_path", default="backend/data_canonical/security_master.parquet")
    parser.add_argument("--out_dir", default="backend/data_canonical/universe")
    parser.add_argument("--start_year", type=int, default=2006)
    parser.add_argument("--end_year", type=int, default=2025)
    args = parser.parse_args()

    bootstrap.ensure_dirs()
    run_id = run_recorder.init_run(
        pipeline_type="data_universe_build", 
        trigger="manual",
        config=vars(args),
        data_versions={}, # Todo: track input versions
        tags=["universe", "v2", "rolling"]
    )

    try:
        # 1. Load Data
        logger.info("Loading Data...")
        
        # Metadata
        if os.path.exists(args.metadata_path):
             meta_pl = pl.read_parquet(args.metadata_path)
             # Normalize columns: symbol -> ticker, ensure name
             if "symbol" in meta_pl.columns and "ticker" not in meta_pl.columns:
                 meta_pl = meta_pl.rename({"symbol": "ticker"})
             if "name" not in meta_pl.columns:
                 meta_pl = meta_pl.with_columns(pl.col("ticker").alias("name"))
                 
             logger.info(f"Loaded Metadata: {len(meta_pl)} rows. Columns: {meta_pl.columns}")
        else:
             logger.warning(f"Metadata not found at {args.metadata_path}. Proceeding without metadata filters.")
             meta_pl = None

        # OHLCV
        # Option A: Single large file (fastest for Polars)
        # Option B: Scan directory of 1000s of files (slower but works if memory issue)
        # Check standard path
        raw_path = Path("backend/data/artifacts/universe/raw_ohlcv.parquet")
        if raw_path.exists():
             logger.info(f"Scanning OHLCV from {raw_path}...")
             ohlcv_lf = pl.scan_parquet(raw_path)
        else:
             # Scan daily_parquet directory?
             daily_dir = Path("backend/data_canonical/daily_parquet")
             if daily_dir.exists():
                 logger.info(f"Scanning OHLCV from {daily_dir}/*.parquet...")
                 ohlcv_lf = pl.scan_parquet(str(daily_dir / "*.parquet"))
             else:
                 raise FileNotFoundError("No OHLCV data found (checked raw_ohlcv.parquet and daily_parquet dir)")

        # Filter years to reduce memory pressure if needed?
        # Actually UniverseBuilder needs history for rolling windows.
        # So we should load at least (start_year - 1) data.
        # But for "build", we want to generate valid universe for the requested range.
        
        # Eager Load (if fits in RAM)
        # 2000 tickers * 10 years * 250 days = 5M rows. Tiny.
        logger.info("Collecting OHLCV into memory...")
        ohlcv_df = ohlcv_lf.collect()
        
        # Ensure Types
        # cast to proper types
        ohlcv_df = ohlcv_df.with_columns([
            pl.col("date").cast(pl.Date),
            pl.col("close").cast(pl.Float32),
            pl.col("volume").cast(pl.Float32),
        ])
        
        logger.info(f"OHLCV Shape: {ohlcv_df.shape}")

        # 2. Build Universe
        logger.info("Building Rolling UniverseFrame...")
        builder = UniverseBuilder()
        frame = builder.build(ohlcv_df, meta_pl)
        
        # Filter Output Range
        # We built history for rolling, but only save requested years
        # But maybe we want to save everything?
        # Let's consistency check: usually artifacts store everything?
        # Partitioning by year handles scale.
        # Filter to >= start_year just to clean up early IPO noise?
        # builder returns comprehensive frame.
        
        start_date = pd.Timestamp(f"{args.start_year}-01-01").date()
        frame = frame.filter(pl.col("date") >= start_date)

        logger.info(f"UniverseFrame Built: {len(frame)} rows. Saving...")

        # 3. Save
        builder.save(frame, args.out_dir, version="v2")
        
        # 3b. Write Metadata (Enhanced)
        print("Writing metadata...")
        # Resolve actual root output path (builder.save appends universe_frame_v2)
        root = Path(args.out_dir) / "universe_frame_v2"
        
        metadata = {
            "schema_signature": sc.schema_signature(frame),
            "n_rows": len(frame),
            "n_symbols": frame.select("symbol").n_unique(),
            "date_range": [
                str(frame["date"].min()),
                str(frame["date"].max())
            ],
            "config": vars(args),
            "columns": frame.columns
        }
        
        with open(root / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Metadata saved to {root}/metadata.json")

        # 4. Diagnostics
        # Summary JSON
        summary_path = Path(args.out_dir) / "universe_frame_v2" / "universe_summary.json"
        
        # Aggregations
        # Group by date
        daily_stats = frame.group_by("date").agg([
             pl.col("symbol").count().alias("total_symbols"),
             pl.col("is_observable").sum().alias("n_observable"),
             pl.col("is_tradable").sum().alias("n_tradable"),
             (pl.col("tier") == "A").sum().alias("tier_A"),
             (pl.col("tier") == "B").sum().alias("tier_B"),
             (pl.col("tier") == "C").sum().alias("tier_C"),
        ]).sort("date")
        
        # Compute Global Stats
        def get_quantiles(col):
            return {
                "min": int(daily_stats[col].min()),
                "p10": int(daily_stats[col].quantile(0.1)),
                "median": int(daily_stats[col].median()),
                "max": int(daily_stats[col].max())
            }

        # Threshold for low breadth (e.g. 50? user said 300 in one place, 50 in preflight. usage: preflight said 50)
        # We'll just report it relative to say 50
        min_breadth = 50
        n_low = daily_stats.filter(pl.col("n_tradable") < min_breadth).height
        frac_low = n_low / daily_stats.height if daily_stats.height > 0 else 0.0

        summary = {
            "global": {
                "n_observable": get_quantiles("n_observable"),
                "n_tradable": get_quantiles("n_tradable"),
                "days_total": daily_stats.height,
                "days_low_breadth": n_low,
                "frac_low_breadth": frac_low,
                "low_breadth_threshold": min_breadth
            },
            "daily": daily_stats.to_dicts()
        }
        
        with open(summary_path, "w") as f:
            json.dump(summary, f, default=str)
            
        logger.info(f"Saved extended summary to {summary_path}")

        run_recorder.finalize_run(run_id, "SUCCESS")

    except Exception as e:
        logger.error(f"Failed: {e}", exc_info=True)
        run_recorder.finalize_run(run_id, "FAILED")
        sys.exit(1)

if __name__ == "__main__":
    main()
