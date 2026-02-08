import polars as pl
import os

gold_path = r"C:\Users\Aishik\Documents\Workshop\Algaie\legacy\v2\Legacy_Algaie_2\backend\data\processed\orthogonal_features_final.parquet"
silver_path = r"C:\Users\Aishik\Documents\Workshop\Algaie\LocalDatabase\data_cache_alpaca_curated(120)\AAPL_1m.parquet"

print("--- Gold Schema ---")
try:
    df_g = pl.read_parquet(gold_path)
    print(df_g.columns)
    print(df_g.head(3))
except Exception as e:
    print(f"Gold Error: {e}")

print("\n--- Silver Schema ---")
try:
    df_s = pl.read_parquet(silver_path)
    print(df_s.columns)
    print(df_s.head(3))
except Exception as e:
    print(f"Silver Error: {e}")
