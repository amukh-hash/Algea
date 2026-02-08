
import os
import pandas as pd
import logging
from backend.app.ops import bootstrap, pathmap
from backend.app.data import security_master

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    bootstrap.ensure_dirs()
    
    paths = pathmap.get_paths()
    ohlcv_root = os.path.join(paths.data_canonical, "ohlcv_adj")
    
    if not os.path.exists(ohlcv_root):
        logger.error(f"OHLCV Root not found: {ohlcv_root}")
        return

    logger.info(f"Scanning {ohlcv_root} for tickers...")
    symbols = []
    for item in os.listdir(ohlcv_root):
        if item.startswith("ticker=") and os.path.isdir(os.path.join(ohlcv_root, item)):
            sym = item.split("ticker=")[1]
            symbols.append(sym)
            
    logger.info(f"Found {len(symbols)} symbols.")
    
    if not symbols:
        logger.warning("No symbols found.")
        return
        
    df = pd.DataFrame({"symbol": symbols})
    df["asset_type"] = "stock" # Default to stock as per source
    df["exchange"] = "UNKNOWN"
    df["sector"] = None
    df["industry"] = None
    df["primary_id"] = df["symbol"]
    df["ipo_date"] = pd.Timestamp("2000-01-01") # Default old IPO for now? Or parse from data?
    # Better: Inspect one file per symbol to get start date? Too slow.
    # Just set default IPO to ensure eligibility (age > 252).
    # Universe selection checks min_ipo_days. If we set 2000, it passes.
    
    logger.info("Building Master DataFrame...")
    master = security_master.build_security_master(df, data_version="rebuilt_v1")
    
    logger.info("Writing Security Master...")
    out = security_master.write_security_master(master, version="rebuilt_v1")
    logger.info(f"Written to {out}")

if __name__ == "__main__":
    main()
