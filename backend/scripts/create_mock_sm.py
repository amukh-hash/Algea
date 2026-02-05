import pandas as pd
import os
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
from backend.app.ops import pathmap

# 1. Mock Security Master
# Use pathmap to get the authoritative path
sm_path = pathmap.resolve("security_master")
symbols = ["AAPL", "MSFT", "GOOGL", "SPY"]

df_sm = pd.DataFrame([
    {"symbol": "AAPL", "asset_name": "Apple Inc", "exchange": "NASDAQ", "is_active": True, "asset_type": "stock", "ipo_date": "1980-12-12", "sector": "Technology"},
    {"symbol": "MSFT", "asset_name": "Microsoft Corp", "exchange": "NASDAQ", "is_active": True, "asset_type": "stock", "ipo_date": "1986-03-13", "sector": "Technology"},
    {"symbol": "GOOGL", "asset_name": "Alphabet Inc", "exchange": "NASDAQ", "is_active": True, "asset_type": "stock", "ipo_date": "2004-08-19", "sector": "Technology"},
    {"symbol": "SPY", "asset_name": "SPDR S&P 500", "exchange": "ARCA", "is_active": True, "asset_type": "etf", "ipo_date": "1993-01-22", "sector": "Financial"},
])

os.makedirs(os.path.dirname(sm_path), exist_ok=True)
df_sm.to_parquet(sm_path)
print(f"Created mock security master at {sm_path}")

# 2. Mock Legacy Universe Snapshots include
pm = pathmap.get_paths()
legacy_root = pm.legacy_artifacts_root
universe_dir = os.path.join(legacy_root, "universe")
os.makedirs(universe_dir, exist_ok=True)

start_date = datetime(2016, 1, 1)
end_date = datetime(2025, 12, 31)
current = start_date

print(f"Generating mock legacy snapshots in {universe_dir}...")
count = 0
while current <= end_date:
    date_str = current.strftime("%Y-%m-%d")
    path = os.path.join(universe_dir, f"universe_{date_str}.parquet")
    
    # Needs: ticker, dollar_vol, close
    mock_data = []
    for s in symbols:
        mock_data.append({
            "ticker": s,
            "dollar_vol": 50000000.0,
            "close": 150.0 + np.random.randn()
        })
    
    pd.DataFrame(mock_data).to_parquet(path)
    
    current += relativedelta(months=1) 
    count += 1

print(f"Generated {count} mock legacy snapshots.")
