
import logging
import argparse
import pandas as pd
import alpaca_trade_api as tradeapi
import os
from backend.app.ops import bootstrap, config
from backend.app.data.security_master import build_security_master, write_security_master

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    bootstrap.ensure_dirs()
    
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    
    if not api_key:
        logger.error("ALPACA_API_KEY missing")
        return

    api = tradeapi.REST(api_key, secret_key, api_version='v2')
    logger.info("Fetching assets from Alpaca...")
    assets = api.list_assets(status='active', asset_class='us_equity')
    
    # Helper to map Alpaca Asset to dict
    data = []
    for a in assets:
        data.append({
            "symbol": a.symbol,
            "exchange": a.exchange,
            "asset_type": "COMMON" if a.exchange in ['NYSE', 'NASDAQ', 'ARCA', 'AMEX'] else "OTHER", # Simple heuristic
             # Alpaca doesn't give sector/industry in list_assets easily
            "ipo_date": None, # Will be filled by logic or defaults
            "primary_id": a.id # Use Alpaca UUID as primary ID? Or Symbol? Schema says string.
        })
        
    raw_df = pd.DataFrame(data)
    
    # Build Canonical
    logger.info("Building Canonical Security Master...")
    master = build_security_master(raw_df, data_version="alpaca_v1")
    
    # Write
    path = write_security_master(master, version="v1")
    logger.info(f"Written Security Master to {path}")

if __name__ == "__main__":
    main()
