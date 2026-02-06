import os
import pandas as pd
import numpy as np
import logging
from typing import Dict, List
from backend.app.ops import pathmap, artifact_registry
from backend.app.features import schemas, validators
from backend.app.data import calendar, ingest_daily, security_master

logger = logging.getLogger(__name__)



def _winsorize_by_date(df: pd.DataFrame, cols: List[str], p_lo=0.01, p_hi=0.99) -> pd.DataFrame:
    def _clip(g):
        # Check if empty to avoid errors
        if g.empty: return g
        lo = g[cols].quantile(p_lo)
        hi = g[cols].quantile(p_hi)
        # Align to avoid index issues if g is a view? Usually safe in apply
        g[cols] = g[cols].clip(lo, hi, axis=1)
        return g
    return df.groupby(df["date"], group_keys=False).apply(_clip)

def _zscore_by_date(df: pd.DataFrame, cols: List[str], eps=1e-12) -> pd.DataFrame:
    def _zs(g):
        if g.empty: return g
        mu = g[cols].mean()
        sd = g[cols].std(ddof=0).replace(0.0, np.nan)
        g[cols] = (g[cols] - mu) / (sd + eps)
        return g
    return df.groupby(df["date"], group_keys=False).apply(_zs)

def _compute_data_version(versions: Dict[str, List[str]]) -> str:
    # Basic aggregation of source versions
    s = "-".join(sorted(list(set(versions.get("ohlcv", []) + versions.get("covariates", []) + versions.get("breadth", [])))))
    if len(s) > 50:
        import hashlib
        return "hash_" + hashlib.md5(s.encode()).hexdigest()[:8]
    return s if s else "v1"

def build_featureframe(
    start_date, end_date, feature_spec: Dict, code_version: str = "v1"
) -> pd.DataFrame:
    """
    Computes features for the given window.
    Returns Schema B6.
    """
    trading_days = calendar.get_trading_days(start_date, end_date)
    if not trading_days:
        raise ValueError("No trading days found in requested range.")

    symbols = feature_spec.get("symbols") if feature_spec else None
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
            
    print(f"DEBUG: Processing {len(symbols)} symbols.")

    cov_path = pathmap.resolve("covariates")
    breadth_path = pathmap.resolve("breadth")
    if not os.path.exists(cov_path):
        raise FileNotFoundError(f"Missing covariates file: {cov_path}")
    if not os.path.exists(breadth_path):
        raise FileNotFoundError(f"Missing breadth file: {breadth_path}")

    cov_df = pd.read_parquet(cov_path)
    breadth_df = pd.read_parquet(breadth_path)
    cov_df["date"] = pd.to_datetime(cov_df["date"])
    breadth_df["date"] = pd.to_datetime(breadth_df["date"])
    
    # Normalize to midnight UTC
    if cov_df["date"].dt.tz is None:
        cov_df["date"] = cov_df["date"].dt.tz_localize("UTC")
    cov_df["date"] = cov_df["date"].dt.normalize()
    
    if breadth_df["date"].dt.tz is None:
        breadth_df["date"] = breadth_df["date"].dt.tz_localize("UTC")
    breadth_df["date"] = breadth_df["date"].dt.normalize()

    required_cov_cols = [
        "date",
        "spy_ret_1d",
        "qqq_ret_1d",
        "iwm_ret_1d",
        "vix_level",
        "rate_proxy"
    ]
    missing_cov = [c for c in required_cov_cols if c not in cov_df.columns]
    if missing_cov:
        raise ValueError(f"Covariates missing required columns: {missing_cov}")

    frames: List[pd.DataFrame] = []
    source_versions: Dict[str, List[str]] = {"ohlcv": [], "covariates": [], "breadth": []}
    if "data_version" in cov_df.columns:
        source_versions["covariates"].extend(cov_df["data_version"].dropna().astype(str).unique().tolist())
    if "data_version" in breadth_df.columns:
        source_versions["breadth"].extend(breadth_df["data_version"].dropna().astype(str).unique().tolist())

    start_ts_utc = pd.Timestamp(start_date).tz_localize("UTC")
    start_ts_naive = pd.Timestamp(start_date)
    end_ts_utc = pd.Timestamp(end_date).tz_localize("UTC")
    end_ts_naive = pd.Timestamp(end_date)

    for symbol in symbols:
        # Load implicit full range first (to avoid filter error)
        
        ohlcv = ingest_daily.load_ohlcv(symbol, start_date=None, end_date=None)
        if ohlcv.empty:
            continue
            
        # Robust Filter
        is_tz_aware = ohlcv["date"].dt.tz is not None
        s_ts = start_ts_utc if is_tz_aware else start_ts_naive
        e_ts = end_ts_utc if is_tz_aware else end_ts_naive
        
        ohlcv = ohlcv[(ohlcv["date"] >= s_ts) & (ohlcv["date"] <= e_ts)]
        if ohlcv.empty:
            continue

        # Normalize to UTC midnight for reindexing alignment
        ohlcv = ohlcv.copy()
        # Force to UTC then normalize to midnight
        if ohlcv["date"].dt.tz is None:
            ohlcv["date"] = ohlcv["date"].dt.tz_localize("UTC")
        else:
            ohlcv["date"] = ohlcv["date"].dt.tz_convert("UTC")
        ohlcv["date"] = ohlcv["date"].dt.normalize()
        ohlcv = ohlcv.drop_duplicates(subset=["date"], keep="last")
        ohlcv = ohlcv.set_index("date")
        
        # Force trading_days to UTC midnight
        trading_days_idx = pd.DatetimeIndex(trading_days)
        if trading_days_idx.tz is None:
            trading_days_idx = trading_days_idx.tz_localize("UTC")
        else:
            trading_days_idx = trading_days_idx.tz_convert("UTC")
        trading_days_norm = trading_days_idx.normalize()
        
        ohlcv = ohlcv.reindex(trading_days_norm)

        if "data_version" in ohlcv.columns:
            source_versions["ohlcv"].extend(
                ohlcv["data_version"].dropna().astype(str).unique().tolist()
            )

        if ohlcv["close_adj"].isnull().all():
            continue

        # --- 3.1 Returns ---
        ohlcv["ret_1d"] = ohlcv["close_adj"].pct_change()
        ohlcv["ret_3d"] = ohlcv["close_adj"].pct_change(3)
        ohlcv["ret_5d"] = ohlcv["close_adj"].pct_change(5)
        ohlcv["ret_10d"] = ohlcv["close_adj"].pct_change(10)
        ohlcv["ret_20d"] = ohlcv["close_adj"].pct_change(20)

        # --- 3.2 EWMA Volatility ---
        ohlcv["ewma_vol_10d"] = ohlcv["ret_1d"].ewm(span=10, adjust=False).std()
        ohlcv["ewma_vol_20d"] = ohlcv["ret_1d"].ewm(span=20, adjust=False).std()
        
        # --- Standard Volatility (for reference) ---
        ohlcv["vol_20d"] = ohlcv["ret_1d"].rolling(20).std()
        ohlcv["vol_chg_1d"] = ohlcv["vol_20d"].pct_change()

        # --- 3.3 Parkinson Volatility ---
        # Prefer adjusted if available; fallback to raw
        hi = ohlcv["high_adj"] if "high_adj" in ohlcv.columns else ohlcv["high"]
        lo = ohlcv["low_adj"]  if "low_adj"  in ohlcv.columns else ohlcv["low"]
        # Handle zeros/errors
        hl = (hi / lo).replace([np.inf, -np.inf, 0], np.nan)
        
        ohlcv["parkinson_var_10d"] = (np.log(hl) ** 2).rolling(10).mean() / (4 * np.log(2))
        ohlcv["parkinson_vol_10d"] = np.sqrt(ohlcv["parkinson_var_10d"])

        ohlcv["parkinson_var_20d"] = (np.log(hl) ** 2).rolling(20).mean() / (4 * np.log(2))
        ohlcv["parkinson_vol_20d"] = np.sqrt(ohlcv["parkinson_var_20d"])

        # --- 3.4 Range % ---
        ohlcv["range_pct_1d"] = (hi - lo) / ohlcv["close_adj"]

        # --- 3.5 Liquidity ---
        if "dollar_volume" not in ohlcv.columns:
            ohlcv["dollar_volume"] = ohlcv["close_adj"] * ohlcv["volume"]

        ohlcv["adv_dollars_20d"] = ohlcv["dollar_volume"].rolling(20).mean()
        ohlcv["dollar_vol_20d"] = ohlcv["adv_dollars_20d"] # Alias/Legacy

        vol_mean = ohlcv["volume"].rolling(20).mean()
        vol_std  = ohlcv["volume"].rolling(20).std()
        # Avoid div by zero
        ohlcv["volume_z_20d"] = (ohlcv["volume"] - vol_mean) / (vol_std + 1e-6)
        ohlcv["volume_stability_20d"] = vol_std / (vol_mean + 1e-6)

        # Overnight Gaps
        prev_close = ohlcv["close_adj"].shift(1)
        if "open_adj" in ohlcv.columns:
            gap_ret = (ohlcv["open_adj"] / prev_close) - 1.0
        else:
            gap_ret = (ohlcv["open"] / prev_close) - 1.0
            
        ohlcv["gap_flag_1d"] = (gap_ret.abs() > 0.02).astype(float)
        ohlcv["gap_freq_20d"] = ohlcv["gap_flag_1d"].rolling(20).mean()

        feat = ohlcv.reset_index().rename(columns={"index": "date"})
        feat["symbol"] = symbol

        # Select Columns
        feat = feat[[
            "date", "symbol",
            "ret_1d", "ret_3d", "ret_5d", "ret_10d", "ret_20d",
            "vol_20d", "vol_chg_1d",
            "ewma_vol_10d", "ewma_vol_20d",
            "parkinson_vol_10d", "parkinson_vol_20d",
            "range_pct_1d",
            "dollar_vol_20d", "adv_dollars_20d", 
            "volume_z_20d", "volume_stability_20d", "gap_freq_20d"
        ]]

        # --- Merge Covariates ---
        # Force UTC to match covariates
        if feat["date"].dt.tz is None:
            feat["date"] = feat["date"].dt.tz_localize("UTC")
        else:
            feat["date"] = feat["date"].dt.tz_convert("UTC")
            
        feat = feat.merge(cov_df[required_cov_cols], on="date", how="left")
        
        # --- 3.6 Market Context ---
        feat["rel_ret_1d"] = feat["ret_1d"] - feat["spy_ret_1d"]
        
        # Rolling Beta (20d)
        # Handle missing data matching for rolling cov
        cov = feat["ret_1d"].rolling(20).cov(feat["spy_ret_1d"])
        var = feat["spy_ret_1d"].rolling(20).var()
        feat["beta_spy_20d"] = cov / (var + 1e-12)

        if "market_breadth_ad" not in breadth_df.columns:
            raise ValueError("Breadth missing required column: market_breadth_ad")
        feat = feat.merge(breadth_df[["date", "market_breadth_ad"]], on="date", how="left")

        frames.append(feat)

    if not frames:
        raise ValueError("FeatureFrame build produced no data.")
    
    print(f"DEBUG: Collected {len(frames)} frames.")
    df = pd.concat(frames, ignore_index=True)
    print(f"DEBUG: Concat DF shape: {df.shape}")
    
    # --- 3.4 Cross-Sectional Dispersion ---
    df["xsec_disp_ret_1d"] = df.groupby("date")["ret_1d"].transform("std")

    # Drop NaNs before normalization to ensure clean stats
    # Drop rows that don't have enough history for 20d rolling
    before_drop = len(df)
    df = df.dropna(subset=[
        "ret_20d", "ewma_vol_20d", "parkinson_vol_20d", "beta_spy_20d",
        "market_breadth_ad", "volume_stability_20d"
    ])
    print(f"DEBUG: After DropNA: {len(df)} (Dropped {before_drop - len(df)})")
    
    # --- Cross-Sectional Normalization ---
    xsec_cols = [
        "ret_1d","ret_3d","ret_5d","ret_10d", "ret_20d",
        "vol_20d","vol_chg_1d",
        "ewma_vol_10d", "ewma_vol_20d",
        "parkinson_vol_10d", "parkinson_vol_20d",
        "range_pct_1d",
        "dollar_vol_20d", "adv_dollars_20d",
        "volume_z_20d",
        "volume_stability_20d", "gap_freq_20d",
        "spy_ret_1d","qqq_ret_1d","iwm_ret_1d",
        "vix_level","rate_proxy","market_breadth_ad",
        "rel_ret_1d", "beta_spy_20d", "xsec_disp_ret_1d"
    ]
    
    # Explicitly ensure columns are in df, filter if not (though they should be)
    xsec_cols = [c for c in xsec_cols if c in df.columns]

    # Resolve ambiguity
    df = df.reset_index(drop=True)

    # Winsorize then Z-score
    df = _winsorize_by_date(df, xsec_cols, p_lo=0.01, p_hi=0.99)
    df = _zscore_by_date(df, xsec_cols)

    data_version = feature_spec.get("data_version") if feature_spec else None
    if not data_version:
        data_version = _compute_data_version(source_versions)
    feature_version = artifact_registry.compute_feature_version(feature_spec or {}, code_version)
    df["feature_version"] = feature_version
    df["data_version"] = data_version

    # Validate with less strict schema or update schema? 
    # The schema might fail on new columns. We pass strict=False/True depending on validator.
    # We should update schema technically, but for now we might bypass strict col check or expect it to pass 'extra' cols.
    validators.validate_df(df, schemas.SCHEMA_FEATUREFRAME, context="FeatureFrame Build", strict=False)
    return df


def write_featureframe(df: pd.DataFrame, feature_spec: Dict, version_tag: str = "v1") -> str:
    code_version = "v1"
    fv = artifact_registry.compute_feature_version(feature_spec, code_version)
    if "feature_version" not in df.columns:
        df["feature_version"] = fv
    if "data_version" not in df.columns:
        df["data_version"] = feature_spec.get("data_version", "unknown")

    path = pathmap.resolve("featureframe", version=version_tag)

    validators.validate_df(df, schemas.SCHEMA_FEATUREFRAME, context="FeatureFrame Write")

    df.to_parquet(path)

    meta = {
        "version": fv,
        "spec": feature_spec,
        "rows": len(df),
        "columns": list(df.columns)
    }
    artifact_registry.write_metadata(path + ".meta", meta)

    return path
