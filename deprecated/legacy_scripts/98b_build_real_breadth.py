
import pandas as pd
import numpy as np
import logging
import os
from tqdm import tqdm
from backend.app.ops import bootstrap, pathmap

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    bootstrap.ensure_dirs()
    
    paths = pathmap.get_paths()
    ohlcv_root = os.path.join(paths.data_canonical, "ohlcv_adj")
    
    if not os.path.exists(ohlcv_root):
        logger.error("OHLCV Root not found.")
        return

    logger.info("Scanning for breadth calculation...")
    tickers = [p.split("ticker=")[1] for p in os.listdir(ohlcv_root) if p.startswith("ticker=")]
    logger.info(f"Found {len(tickers)} tickers.")
    
    date_stats = {} # date -> {up: 0, down: 0, tot: 0}
    
    for ticker in tqdm(tickers):
        try:
            p = os.path.join(ohlcv_root, f"ticker={ticker}", "data.parquet")
            df = pd.read_parquet(p, columns=["date", "close_adj"])
            if df.empty: continue
            
            # Force UTC
            if df["date"].dt.tz is None:
                df["date"] = df["date"].dt.tz_localize("UTC")
            else:
                df["date"] = df["date"].dt.tz_convert("UTC")
            
            df = df.sort_values("date")
            
            # 1 = Up, -1 = Down, 0 = Flat
            vals = np.sign(df["close_adj"].diff()).fillna(0).astype(int)
            dates = df["date"]
            
            for d, v in zip(dates, vals):
                if d not in date_stats:
                    date_stats[d] = {"up": 0, "down": 0, "flat": 0, "tot": 0}
                
                date_stats[d]["tot"] += 1
                if v > 0: date_stats[d]["up"] += 1
                elif v < 0: date_stats[d]["down"] += 1
                else: date_stats[d]["flat"] += 1
                
        except Exception:
            pass
            
    # Convert to DF
    logger.info("Aggregating Breadth stats...")
    rows = []
    keys = sorted(date_stats.keys()) # Sort keys which are all UTC now
    
    for d in keys:
        s = date_stats[d]
        net = (s["up"] - s["down"])
        tot = s["tot"]
        metric = net / tot if tot > 0 else 0.0
        
        rows.append({
            "date": d,
            "market_breadth_ad": metric,
            "advancers": s["up"],
            "decliners": s["down"],
            "total_issues": s["tot"]
        })
        
    bdf = pd.DataFrame(rows)
    # Ensure date col is valid
    if not bdf.empty:
        bdf["date"] = pd.to_datetime(bdf["date"])
    
    bdf["data_version"] = "real_calc_v1"
    
    out_path = pathmap.resolve("breadth")
    bdf.to_parquet(out_path)
    logger.info(f"Written real breadth to {out_path} ({len(bdf)} rows)")

if __name__ == "__main__":
    main()
