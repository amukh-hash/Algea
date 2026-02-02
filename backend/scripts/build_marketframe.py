import os
import sys
import json
import argparse
import polars as pl
from tqdm import tqdm

# Ensure backend in path
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from backend.app.data import marketframe, manifest

DATA_DIR = os.path.join("backend", "data_cache_alpaca")
CONTEXT_DIR = os.path.join("backend", "data", "context")
OUTPUT_DIR = os.path.join("backend", "data", "marketframe")
CONTEXT_FILE = "breadth_1m.parquet"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, help="Build for specific ticker only")
    parser.add_argument("--data_dir", default=DATA_DIR)
    parser.add_argument("--context_path", default=os.path.join(CONTEXT_DIR, CONTEXT_FILE))
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.ticker:
        tickers = [args.ticker]
    else:
        # Load trade universe? Or full context universe?
        # Usually we build MarketFrame for the trade universe + maybe some liquid proxies.
        # Let's use context universe to be safe/complete.
        tickers = manifest.get_context_universe()

    print(f"Building MarketFrames for {len(tickers)} tickers...")

    context_path = args.context_path
    if not os.path.exists(context_path):
        print(f"Context file not found at {context_path}. Run build_breadth_context.py first.")
        return

    stats = {
        "success": [],
        "failed": [],
        "total_rows": 0
    }

    for ticker in tqdm(tickers):
        try:
            mf, meta = marketframe.build_marketframe(ticker, args.data_dir, context_path)

            out_path = os.path.join(args.output_dir, f"marketframe_{ticker}_1m.parquet")
            mf.write_parquet(out_path)

            # Save minimal metadata? Or just rely on parquet?
            # We aggregate stats.
            stats["success"].append(ticker)
            stats["total_rows"] += meta["rows"]

        except FileNotFoundError:
            # warn silently
            pass
        except Exception as e:
            print(f"Error building {ticker}: {e}")
            stats["failed"].append(ticker)

    print(f"Build Complete. Success: {len(stats['success'])}, Failed: {len(stats['failed'])}")

    # Save build manifest/stats
    with open(os.path.join(args.output_dir, "build_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

if __name__ == "__main__":
    main()
