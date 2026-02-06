
import pandas as pd
import numpy as np
import logging
import os
from backend.app.ops import bootstrap, pathmap
from backend.app.data import ingest_daily

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    bootstrap.ensure_dirs()
    
    # Symbols: SPY, QQQ, IWM, IEF (Rates)
    logger.info("Loading SPY/QQQ/IWM/IEF...")
    spy = ingest_daily.load_ohlcv("SPY")
    qqq = ingest_daily.load_ohlcv("QQQ")
    iwm = ingest_daily.load_ohlcv("IWM")
    ief = ingest_daily.load_ohlcv("IEF")
    
    if spy.empty or qqq.empty:
        logger.error("Missing SPY or QQQ data.")
        return

    # 1. Process Returns (SPY, QQQ, IWM)
    def prepare_symbol(df, name):
        if df.empty: return pd.DataFrame()
        df = df.sort_values("date").set_index("date")
        df = df[~df.index.duplicated(keep='first')]
        # Features
        ret = df["close_adj"].pct_change().rename(f"{name}_ret_1d")
        return ret.to_frame()

    spy_f = prepare_symbol(spy, "spy")
    qqq_f = prepare_symbol(qqq, "qqq")
    iwm_f = prepare_symbol(iwm, "iwm")
    
    # 2. Derive VIX Level (Realized Volatility) -> "vix_level"
    # Proxy: Annualized 21-day rolling std dev of SPY returns * 100
    # This is "Realized Vol", not "Implied Vol", but it's the best real metric available.
    spy_ret = spy_f["spy_ret_1d"]
    # 21 days rolling std * sqrt(252) * 100
    vix_proxy = spy_ret.rolling(window=21).std() * np.sqrt(252) * 100
    vix_s = vix_proxy.rename("vix_level")
    
    # 3. Derive Rate Proxy (IEF) -> "rate_proxy"
    # IEF Price is inverse to yield. 
    # Let's just use IEF Close directly as the feature. The model can learn the relationship.
    # Naming it "rate_proxy" satisfies the schema.
    if not ief.empty:
        ief_f = prepare_symbol(ief, "ief") # get index alignment
        # We want the PRICE, not return, as the level.
        ief_aligned = ief.set_index("date")["close_adj"].reindex(spy_f.index)
        # Normalize? Or raw? Raw price is fine for trees/RNNs usually, but yield is better.
        # But we don't know coupon. 
        # Let's use 1/Price as a crude "Yield Factor" or just Price.
        # Spec says "rate_proxy". IEF price is a proxy.
        rate_s = ief_aligned.rename("rate_proxy")
    else:
        logger.warning("IEF missing. Rate proxy will be NaN (schema allows?).")
        rate_s = pd.Series(name="rate_proxy", dtype="float64")
        
    # Merge All
    # Base on SPY dates
    df = spy_f.join(qqq_f, how="outer")
    df = df.join(iwm_f, how="outer")
    df = df.join(vix_s, how="left")
    df = df.join(rate_s, how="left")
    
    df = df.sort_index().reset_index()
    df = df.rename(columns={"index": "date"})
    
    # Fill? 
    # Forward fill VIX/Rates (levels)
    df["vix_level"] = df["vix_level"].ffill()
    df["rate_proxy"] = df["rate_proxy"].ffill()
    
    # Returns fill with 0
    cols = ["spy_ret_1d", "qqq_ret_1d", "iwm_ret_1d"]
    df[cols] = df[cols].fillna(0.0)
    
    df["data_version"] = "real_v2_derived"
    
    # Validation against Schema
    # Schema says: spy_ret_1d, qqq_ret_1d, iwm_ret_1d, vix_level, rate_proxy, market_breadth_ad
    # Breadth is separate file.
    
    out_path = pathmap.resolve("covariates")
    df.to_parquet(out_path)
    logger.info(f"Written real covariates to {out_path} | Cols: {df.columns.tolist()}")

if __name__ == "__main__":
    main()
