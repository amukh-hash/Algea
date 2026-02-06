
import os
import pandas as pd
import numpy as np
import logging
from typing import List, Dict
from backend.app.ops import pathmap, config, artifact_registry
from backend.app.features import schemas, validators
from backend.app.data import ingest_daily, security_master

logger = logging.getLogger(__name__)

def build_universe_manifest(asof_date, base_symbols: List[str], rules: Dict) -> pd.DataFrame:
    """
    Applies filtering rules to base symbols for a given date.
    Returns Schema B5 DataFrame.
    """
    # Create both Naive and UTC versions of asof_ts for robust filtering
    asof_ts_utc = pd.Timestamp(asof_date).tz_localize("UTC")
    asof_ts_naive = pd.Timestamp(asof_date)
    
    security_df = security_master.load_security_master()
    security_df = security_df.set_index("symbol", drop=False)

    metrics_rows = []
    for symbol in base_symbols:
        # Load FULL history (no end_date args to avoid ingest_daily TZ issues)
        ohlcv = ingest_daily.load_ohlcv(symbol, end_date=None)
        if ohlcv.empty:
            continue
            
        # Detect TZ of loaded data
        is_tz_aware = ohlcv["date"].dt.tz is not None
        if is_tz_aware:
            cutoff = asof_ts_utc
        else:
            cutoff = asof_ts_naive
            
        ohlcv = ohlcv.sort_values("date")
        ohlcv = ohlcv[ohlcv["date"] <= cutoff]

        last_row = ohlcv.tail(1)
        close_adj = float(last_row["close_adj"].iloc[0]) if not last_row.empty else np.nan

        recent = ohlcv.tail(20)
        if len(recent) < 5:
            continue

        dollar_vol = recent["close_adj"] * recent["volume"]
        adv20 = float(dollar_vol.median())

        returns = recent["close_adj"].pct_change().dropna()
        vol20 = float(returns.std()) if not returns.empty else np.nan

        sm_row = security_df.loc[symbol] if symbol in security_df.index else None
        asset_type = sm_row["asset_type"] if sm_row is not None else "COMMON"
        ipo_date = sm_row["ipo_date"] if sm_row is not None else pd.Timestamp("1900-01-01")
        sector = sm_row["sector"] if sm_row is not None else None

        if pd.isna(ipo_date):
            ipo_age_td = 0
        else:
            ipo_age_td = len(pd.bdate_range(start=ipo_date, end=asof_ts_naive))

        metrics_rows.append(
            {
                "symbol": symbol,
                "asset_type": asset_type,
                "close_adj": close_adj,
                "adv20": adv20,
                "vol20": vol20,
                "ipo_date": ipo_date,
                "ipo_age_td": ipo_age_td,
                "sector": sector,
            }
        )

    metrics_df = pd.DataFrame(metrics_rows)
    if metrics_df.empty:
        return metrics_df

    return apply_universe_rules(metrics_df, rules)

def apply_universe_rules(metrics_df: pd.DataFrame, rules: Dict) -> pd.DataFrame:
    """
    metrics_df expected cols: symbol, asset_type, close_adj, adv20, vol20, ipo_date, sector
    """
    df = metrics_df.copy()
    
    # Defaults
    min_price = rules.get("min_price", 5.0)
    min_adv = rules.get("min_adv", 25e6)
    min_ipo_days = rules.get("min_ipo_days", 252)
    top_n = rules.get("top_n", config.UNIVERSE_BUFFER_MAX)
    
    # Reason Codes
    df["eligible"] = False
    df["reason_code"] = "OK"
    
    # 1. Asset Type
    mask_common = df["asset_type"].isin(["COMMON", "stock", "Common Stock"])
    df.loc[~mask_common, "reason_code"] = "NON_COMMON"
    
    # 2. Price
    mask_price = df["close_adj"] >= min_price
    # Only overwrite if currently OK
    mask_ok = df["reason_code"] == "OK"
    df.loc[mask_ok & ~mask_price, "reason_code"] = "PRICE_LT_5"
    
    # 3. IPO Age
    # df['ipo_age_td'] should be provided or calc from ipo_date
    # Stub logic for now
    if "ipo_age_td" not in df.columns:
        df["ipo_age_td"] = 9999
    
    mask_ok = df["reason_code"] == "OK"
    mask_ipo = df["ipo_age_td"] >= min_ipo_days
    df.loc[mask_ok & ~mask_ipo, "reason_code"] = "IPO_LT_252TD"
    
    # 4. Liquidity
    mask_ok = df["reason_code"] == "OK"
    mask_adv = df["adv20"] >= min_adv
    df.loc[mask_ok & ~mask_adv, "reason_code"] = "ADV20_LT_25M"
    
    # 5. Sector Cap? (Optional)
    
    # Final Eligibility
    df.loc[df["reason_code"] == "OK", "eligible"] = True
    
    # Top N Filtering (if too many eligible)
    eligible = df[df["eligible"]].copy()
    if len(eligible) > top_n:
        # Sort by ADV descending
        eligible = eligible.sort_values("adv20", ascending=False)
        cutoff = eligible.iloc[top_n]["adv20"]
        # Mark those below cutoff
        mask_cut = (df["eligible"]) & (df["adv20"] < cutoff)
        df.loc[mask_cut, "eligible"] = False
        df.loc[mask_cut, "reason_code"] = "ADV_RANK_CUTOFF"
        
    return df

def write_universe_manifest(asof_date, df: pd.DataFrame, rules: Dict) -> str:
    path = pathmap.resolve("manifest", date=asof_date)
    
    # Versioning
    code_version = "v1" # Git hash ideally
    uv = artifact_registry.compute_universe_version(rules, code_version)
    df["universe_version"] = uv
    df["asof_date"] = pd.to_datetime(asof_date)
    
    # Renaming to Schema B5
    # Input has 'adv20' -> Schema 'adv20_median'
    rename_map = {"adv20": "adv20_median", "vol20": "vol20_median"}
    df.rename(columns=rename_map, inplace=True)
    
    # Validate
    validators.validate_df(df, schemas.SCHEMA_UNIVERSE_MANIFEST, context="Universe Write")
    
    # Ensure dir
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path)
    return path

class UniverseSelector:
    """
    Wraps universe selection logic.
    """
    def __init__(self, rules: Dict = None):
        self.rules = rules or {
            "min_price": 5.0,
            "min_adv": 25e6,
            "min_ipo_days": 252,
            "top_n": 500
        }

    def select(self, raw_df: pd.DataFrame, meta_df: pd.DataFrame, asof_date: str) -> pd.DataFrame:
        """
        Runs the universe selection pipeline.
        
        Args:
            raw_df: DataFrame with 'symbol', 'date', 'close', 'volume' etc.
            meta_df: Metadata with 'symbol', 'ipo_date', 'sector' etc.
            asof_date: Date to select for.
            
        Returns:
            DataFrame with 'symbol', 'is_eligible', 'reason_code'.
        """
        base_symbols = raw_df["symbol"].unique().tolist()
        result = build_universe_manifest(asof_date, base_symbols, self.rules)
        if result.empty:
            return pd.DataFrame(columns=["symbol", "is_eligible", "reason_code"])

        return result[["symbol", "eligible", "reason_code"]].rename(columns={"eligible": "is_eligible"})
