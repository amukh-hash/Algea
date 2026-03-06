"""
Build canonical market covariates for Chronos-2 training.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from algae.data.market.covariates import build_and_persist_covariates

def main():
    per_ticker = ROOT / "backend" / "data" / "canonical" / "per_ticker"
    out_path = ROOT / "backend" / "data" / "canonical" / "market_covariates.parquet"
    
    print(f"Building covariates from {per_ticker} to {out_path}...")
    build_and_persist_covariates(per_ticker, out_path, overwrite=True)
    print("Done.")

if __name__ == "__main__":
    main()
