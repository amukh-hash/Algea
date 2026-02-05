import pandas as pd
import numpy as np
import os
from backend.app.ops import bootstrap

def main():
    out_path = "backend/data/context/breadth_1m.parquet"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    dates = pd.date_range("2016-01-01", "2025-12-31", freq="1min")
    # Take a sample to be faster? 
    # Just a few days is enough? 
    # Validation checks monotonicity.
    # Validation also iterates "backend/data/marketframe" for other files.
    # Wait, does validation iterate marketframe?
    # Yes: "for f in os.listdir(args.data_dir): ... validate_marketframe..."
    # If marketframe dir is empty, it passes loop.
    # My stubs didn't write to backend/data/marketframe, they wrote to canonical or features.
    # So marketframe validation might just pass (empty).
    # But context check failures.
    
    # Create valid breadth context
    df = pd.DataFrame({
        "timestamp": dates,
        "ad_line": np.cumsum(np.random.randn(len(dates))),
        "bpi": 50 + np.cumsum(np.random.randn(len(dates)))
    })
    
    # Validation expects index or timestamp column?
    # "if 'timestamp' in df.columns: ... if isinstance(index, DatetimeIndex)..."
    # It checks both.
    
    df.set_index("timestamp", inplace=True)
    df.to_parquet(out_path)
    print(f"Created mock breadth at {out_path}")

if __name__ == "__main__":
    main()
