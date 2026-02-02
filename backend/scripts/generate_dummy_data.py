import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

DATA_DIR = "backend/data_cache_alpaca"

def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    tickers = ["AAPL", "SPY", "QQQ", "NVDA", "MSFT"]

    # 1 year of 1m data
    dates = pd.date_range("2023-01-01", "2023-01-31", freq="1min", tz="UTC") # 1 month for speed

    print(f"Generating dummy data for {tickers} in {DATA_DIR}...")

    for ticker in tickers:
        fpath = os.path.join(DATA_DIR, f"{ticker}_1m.parquet")
        if not os.path.exists(fpath):
            n = len(dates)
            base_price = 100.0 if ticker != "SPY" else 400.0

            # Random walk
            returns = np.random.normal(0, 0.0001, n)
            prices = base_price * np.exp(np.cumsum(returns))

            df = pd.DataFrame({
                "open": prices,
                "high": prices * 1.0001,
                "low": prices * 0.9999,
                "close": prices,
                "volume": np.abs(np.random.normal(1000, 500, n))
            }, index=dates)

            # Reset index to make 'timestamp' a column if needed, or keeping index works for pandas read_parquet
            # My scripts usually handle pandas index if parquet has it.
            # But let's verify.
            # build_marketframe uses pl.read_parquet.
            # If saved with index=True, pandas parquet often sets index.
            # Let's save with reset_index to be safe and explicit 'timestamp' col.
            df = df.reset_index().rename(columns={"index": "timestamp"})

            df.to_parquet(fpath)
            print(f"Generated {fpath}")
        else:
            print(f"Exists: {fpath}")

if __name__ == "__main__":
    main()
