
import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))

import logging
import argparse
import pandas as pd
import os
from dateutil.relativedelta import relativedelta
from backend.app.ops import bootstrap, pathmap, config
from backend.app.data import universe_selector as universe, security_master
from backend.app.data.ingest.ohlcv_daily import load_ohlcv

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
        
        # Real Calculation using universe.py logic
        # Optimize: Pass only active symbols?
        # For now, pass all master symbols, the function loads OHLCV per symbol.
        # This allows accurate Point-in-Time calculation.
        
        # Define base rules
        rules = {
            "min_price": 5.0,
            "min_adv": 25e6, # $25M
            "min_ipo_days": 252,
            "top_n": 1000  # Broad universe
        }
        
        base_symbols = sec_master["symbol"].tolist()
        
        # Build Manifest
        # usage: build_universe_manifest(asof_date, base_symbols, rules)
        # Note: This loads OHLCV for every symbol. It might be slow.
        # But it's the correct "functional" path.
        manifest = universe.build_universe_manifest(date_str, base_symbols, rules)
        
        # Write
        if not manifest.empty:
            out = universe.write_universe_manifest(current, manifest, rules)
            logger.info(f"  Written {out} | Eligible: {manifest['eligible'].sum()}")
        else:
            logger.warning(f"  Empty manifest for {date_str}")
        
        current += relativedelta(months=1)

if __name__ == "__main__":
    main()
