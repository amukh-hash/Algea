
import pandas as pd
import os

try:
    path = "backend/data_canonical/security_master.parquet"
    if not os.path.exists(path):
        print(f"Error: {path} not found")
        exit(1)
        
    df = pd.read_parquet(path)
    count = len(df)
    print(f"Total Universe Count: {count}")
    
    # Also check asset types if possible
    if "asset_type" in df.columns:
        print("Asset Types breakdown:")
        print(df["asset_type"].value_counts())
        
except Exception as e:
    print(f"Error: {e}")
