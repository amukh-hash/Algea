
import logging
import argparse
import pandas as pd
import os
from dateutil.relativedelta import relativedelta
from backend.app.ops import bootstrap, pathmap, config
from backend.app.data import universe, security_master
from backend.app.data.ingest_daily import load_ohlcv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2016-01-01")
    parser.add_argument("--end", default="2026-02-03")
    parser.add_argument("--freq", default="monthly")
    args = parser.parse_args()
    
    bootstrap.ensure_dirs()
    
    # Load Master
    sec_master = security_master.load_security_master()
    
    start_dt = pd.to_datetime(args.start)
    end_dt = pd.to_datetime(args.end)
    current = start_dt
    
    while current <= end_dt:
        date_str = current.strftime("%Y-%m-%d")
        logger.info(f"Building Manifest for {date_str}...")
        
        # In a real run, we'd load metrics efficiently.
        # For this script (backfill), we might have to mock or scan.
        # Scanning 5000+ parquet files is SLOW.
        
        # Optimization: Use legacy 'universe_YYYY-MM-DD.parquet' if available to speed up?
        # Check pathmap legacy root
        pm = pathmap.get_paths()
        legacy_path = os.path.join(pm.legacy_artifacts_root, f"universe/universe_{date_str}.parquet")
        
        metrics = []
        if os.path.exists(legacy_path) and config.ALLOW_LEGACY_READ:
            # Load legacy snapshot to get eligible tickers + basic metrics
            logger.info("  Loading legacy snapshot to hydrate metrics...")
            leg_df = pd.read_parquet(legacy_path)
            # Legacy cols: ticker, dollar_vol (approx adv?), close
            # We need: symbol, asset_type, close_adj, adv20, vol20, ipo_date, sector
            
            # Merge with Security Master for static fields
            merged = leg_df.merge(sec_master, left_on="ticker", right_on="symbol", how="left")
            
            # Map legacy metrics
            # legacy 'dollar_vol' is median 20d dollar vol usually
            # we need adv20 (shares). adv20 = dollar_vol / close (approx)
            
            merged["adv20"] = merged["dollar_vol"] / merged["close"]
            merged["close_adj"] = merged["close"] # Assuming legacy was adjusted
            merged["vol20"] = 0.02 # Dummy vol if missing? Or calculated?
            # Legacy didn't store vol? Feature frame does. 
            # We'll set dummy for manifest backfill unless we recalc.
            
            metrics_df = merged[[
                "symbol", "asset_type", "close_adj", "adv20", "vol20", "ipo_date", "sector"
            ]].copy()
            
        else:
             # Full recalc (Heavy)
             # Skip for now or implement slow loop?
             # We'll assert legacy exists for this migration plan step.
             logger.warning("  Legacy snapshot not found. Skipping date (Heavy recalc not implemented).")
             current += relativedelta(months=1)
             continue

        # Apply Rules (Strict B5)
        rules = {
            "min_price": 5.0,
            "min_adv": 25e6, # $25M
            "min_ipo_days": 252
        }
        
        manifest = universe.apply_universe_rules(metrics_df, rules)
        
        # Write
        out = universe.write_universe_manifest(current, manifest, rules)
        logger.info(f"  Written {out}")
        
        current += relativedelta(months=1)

if __name__ == "__main__":
    main()
