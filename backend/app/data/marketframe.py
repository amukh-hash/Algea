
import os
import pandas as pd
import logging
from typing import Optional, List
from backend.app.ops import pathmap, config
from backend.app.data import calendar, security_master
from backend.app.data.ingest import ohlcv_daily

logger = logging.getLogger(__name__)

def build_marketframe(start_date, end_date, symbols: List[str] = None) -> pd.DataFrame:
    """
    Constructs a MarketFrame: Aligned daily OHLCV + Covariates + Breadth.
    Strictly aligned to NYSE session dates.
    """
    trading_days = calendar.get_trading_days(start_date, end_date)
    if not trading_days:
        raise ValueError("No trading days found for marketframe build.")

    if not symbols:
        try:
            master = security_master.load_security_master()
            symbols = master["symbol"].dropna().astype(str).unique().tolist()
        except Exception:
            paths = pathmap.get_paths()
            ohlcv_root = os.path.join(paths.data_canonical, "ohlcv_adj")
            if not os.path.exists(ohlcv_root):
                raise FileNotFoundError("No canonical OHLCV partitions found.")
            symbols = [p.split("ticker=")[-1] for p in os.listdir(ohlcv_root) if p.startswith("ticker=")]

    cov_path = pathmap.resolve("covariates")
    breadth_path = pathmap.resolve("breadth")
    if not os.path.exists(cov_path):
        raise FileNotFoundError(f"Missing covariates file: {cov_path}")
    if not os.path.exists(breadth_path):
        raise FileNotFoundError(f"Missing breadth file: {breadth_path}")

    cov_df = pd.read_parquet(cov_path)
    cov_df["date"] = pd.to_datetime(cov_df["date"])
    breadth_df = pd.read_parquet(breadth_path)
    breadth_df["date"] = pd.to_datetime(breadth_df["date"])

    frames: List[pd.DataFrame] = []
    for symbol in symbols:
        ohlcv = ohlcv_daily.load_ohlcv(symbol, start_date=start_date, end_date=end_date)
        if ohlcv.empty:
            continue
        ohlcv = ohlcv.copy()
        ohlcv["date"] = pd.to_datetime(ohlcv["date"])
        ohlcv = ohlcv.sort_values("date")
        ohlcv = ohlcv.set_index("date").reindex(pd.DatetimeIndex(trading_days)).reset_index()
        ohlcv = ohlcv.rename(columns={"index": "date"})
        ohlcv["symbol"] = symbol

        merged = ohlcv.merge(cov_df, on="date", how="left")
        merged = merged.merge(breadth_df, on="date", how="left")
        frames.append(merged)

    if not frames:
        raise ValueError("MarketFrame build produced no data.")

    return pd.concat(frames, ignore_index=True)
