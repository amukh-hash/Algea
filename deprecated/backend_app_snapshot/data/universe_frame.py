
import polars as pl
import numpy as np
import pandas as pd
from typing import Optional, List, Dict
import logging
from backend.app.ops import artifact_registry, pathmap
from backend.app.data.universe_config import UniverseRules
from backend.app.data import schema_contracts as sc

logger = logging.getLogger(__name__)

class UniverseBuilder:
    """
    Builds the Rolling UniverseFrame artifacts.
    Computes eligibility masks (observable, tradable) using strictly trailing data.
    """
    def __init__(self, rules: UniverseRules = None):
        self.rules = rules or UniverseRules()

    def build(self, ohlcv_df: pl.DataFrame, metadata_df: Optional[pl.DataFrame] = None) -> pl.DataFrame:
        """
        Build UniverseFrame from raw OHLCV (Polars).
        OHLCV expected columns: date, ticker, close, volume, close_prev (or computed).
        Metadata expected columns: ticker, name, asset_type (optional).
        """
        # 1. Prepare Data
        # Ensure sorted by date per ticker
        df = ohlcv_df.sort(["ticker", "date"])

        # 2. Compute Rolling Metrics
        # dvol
        df = df.with_columns((pl.col("close") * pl.col("volume")).alias("dvol"))
        
        # Returns (if not present)
        # We can compute ret1 = close / close.shift(1) - 1 per group
        if "ret1" not in df.columns:
            df = df.with_columns(
                (pl.col("close") / pl.col("close").shift(1).over("ticker") - 1).alias("ret1")
            )

        # Rolling Ops (Window functions over ticker)
        # Operations:
        # - adv20: mean(dvol, 20)
        # - adv60: mean(dvol, 60)
        # - vol20: std(ret1, 20)
        # - missing_60: count(close is null, 60) -> tricky in polars if rows missing. 
        #   Assumes daily rows exist. If rows are missing from DF, rolling won't see them as nulls.
        #   Ideally data is reindexed. For now, we assume nulls are explicit or we count zeroes?
        #   Let's assume "missing" means specific nulls or gaps. 
        #   If rows are just not there, we rely on "age" and "valid count".
        #   Let's check "zero_volume_60": count(volume == 0, 60)
        
        # Define windows
        w20 = self.rules.adv20_window
        w60 = self.rules.adv60_window

        df = df.with_columns([
            pl.col("dvol").rolling_mean(w20).over("ticker").alias("adv20"),
            pl.col("dvol").rolling_mean(w60).over("ticker").alias("adv60"),
            pl.col("ret1").rolling_std(w20).over("ticker").alias("vol20"),
            (pl.col("volume") == 0).cast(pl.Int32).rolling_sum(w60).over("ticker").alias("zero_volume_60"),
             # Max share: max(dvol / sum(dvol, 20)) over 20d
             # First compute daily share of 20d volume? No, "share of TODAY's volume in last 20d"?
             # Usually "max_share20" means: max( dvol[t] / sum(dvol[t-19...t]) ) over window?
             # Or "volume spike": day_vol / adv20?
             # Let's interpret "max_share20" as per user spec: "max(dvol_day / sum(dvol over 20d)) over last 20d"
             # actually "sum(dvol over 20d)" is adv20 * 20.
             # So daily_share = dvol / (adv20 * 20). 
             # Then rolling_max(daily_share, 20).
        ])
        
        # Post-compute Max Share
        w_share_denom = self.rules.adv20_window
        w_spike = self.rules.spike_window
        
        df = df.with_columns(
            (pl.col("dvol") / (pl.col("adv20") * w_share_denom + 1.0)).alias("daily_share")
        ).with_columns(
            pl.col("daily_share").rolling_max(w_spike).over("ticker").alias("max_share20")
        )

        # Age (Trading Days)
        # cumul_count per ticker
        df = df.with_columns(
            pl.col("date").cum_count().over("ticker").alias("age_days")
        )

        # 3. Apply Metadata Filters
        # (Exclude ETFs etc if metadata provided)
        # We'll create a "metadata_ok" mask
        if metadata_df is not None:
             # Join metadata
             df = df.join(metadata_df.select(["ticker", "name", "asset_type"]), on="ticker", how="left")
             
             # Case-insensitive exclude patterns
             # Default True
             # Check asset_type == 'US_EQUITY' logic? User said "status active, asset_class us_equity".
             # Assuming input DF already filtered for active/us_equity or we check here?
             # Let's assume metadata_df passed in IS the valid universe snapshot (active/us_equity).
             # If ticker not in metadata, it's NOT OK? Or warning?
             # User: "Use Alpaca metadata snapshot if available... If metadata missing, do not hard-fail"
             
             # Name filter
             # Construct regex from patterns
             patterns = self.rules.exclude_name_patterns
             if patterns:
                 # Check if name contains any pattern
                 # output: boolean col "name_ok"
                 # coalesce name to "" to avoid null issues
                 name_col = pl.col("name").fill_null("").str.to_uppercase()
                 
                 # Create expression: ~ (name.contains(P1) | name.contains(P2) ...)
                 # Polars doesn't support list-based contains efficiently in one go without loop?
                 # using regex union: (P1|P2|P3)
                 regex = "|".join([p.upper() for p in patterns])
                 df = df.with_columns(
                     (~name_col.str.contains(regex)).alias("metadata_ok")
                 )
             else:
                 df = df.with_columns(pl.lit(True).alias("metadata_ok"))
                 
             # Fallback: if ticker not in metadata (metadata_ok is null?), default behavior?
             # Left join makes metadata_ok null if missing. Compute fallback.
             df = df.with_columns(pl.col("metadata_ok").fill_null(True)) # User said don't hard fail
             
        else:
            df = df.with_columns(pl.lit(True).alias("metadata_ok"))


        # 4. Apply Masks
        # Observable
        r = self.rules
        
        # Fill null metrics with bad values for comparison
        # (e.g. adv20 null -> 0)
        df = df.with_columns([
            pl.col("adv20").fill_null(0),
            pl.col("adv60").fill_null(0),
            pl.col("vol20").fill_null(float('inf')), # Infinite vol is bad
            pl.col("zero_volume_60").fill_null(999),
            pl.col("max_share20").fill_null(1.0),
            pl.col("close").fill_null(0.0)
        ])

        mask_obs = (
            (pl.col("age_days") >= r.min_age_observable) &
            (pl.col("close") >= r.min_price_observable) &
            (pl.col("adv20") >= r.min_adv20_observable) &
            (pl.col("adv60") >= r.min_adv60_observable) &
            (pl.col("zero_volume_60") <= r.max_zero_volume_60_observable) &
            (pl.col("vol20").is_finite()) & 
            (pl.col("metadata_ok"))
        )
        
        mask_trad = (
            (pl.col("age_days") >= r.min_age_tradable) &
            (pl.col("close") >= r.min_price_tradable) &
            (pl.col("adv20") >= r.min_adv20_tradable) &
            (pl.col("adv60") >= r.min_adv60_tradable) &
            (pl.col("zero_volume_60") <= r.max_zero_volume_60_tradable) &
            (pl.col("max_share20") <= r.max_share20_tradable) &
            (pl.col("vol20").is_finite()) &
            (pl.col("metadata_ok"))
        )

        df = df.with_columns([
            mask_obs.alias("is_observable"),
            mask_trad.alias("is_tradable")
        ])
        
        # Ensure tradable implies observable (logic says strict > relaxed, but verify)
        # Should be true by definition of thresholds, but let's enforce?
        # mask_trad = mask_trad & mask_obs? 
        # Yes.
        df = df.with_columns(
            (pl.col("is_tradable") & pl.col("is_observable")).alias("is_tradable")
        )

        # 5. Tiers and Weights (Tradable Only)
        # Tier
        # A if >= tier_a
        # B if < tier_a and >= tier_b
        # C else
        # NULL if not tradable
        
        tier_expr = (
            pl.when(pl.col("is_tradable").not_())
            .then(None)
            .when(pl.col("adv20") >= r.tier_a_min_adv20)
            .then(pl.lit("A"))
            .when(pl.col("adv20") >= r.tier_b_min_adv20)
            .then(pl.lit("B"))
            .otherwise(pl.lit("C"))
        ).alias("tier")
        
        # Weight
        # w = clip( log(adv20 / min) / log(max / min), 0, 1 )
        # if not tradable, 0
        
        # avoid log(0)
        # min_adv20_tradable is base.
        # Actually rules.weight_min_adv20 vs min_adv20_tradable?
        # User specified weight_min same as tradable min usually.
        
        w_min = r.weight_min_adv20
        w_max = r.weight_max_adv20
        # Denom
        denom = np.log(w_max / w_min)
        
        # Numerator: log(adv20 / w_min)
        # Clip at 0 and 1
        
        # Calculation in polars expressions
        # log(adv20) - log(w_min)
        # We need numpy log or polars log? Polars has .log()
        
        weight_expr = (
            pl.when(pl.col("is_tradable").not_())
            .then(0.0)
            .otherwise(
                ((pl.col("adv20") / w_min).log() / denom).clip(0.0, 1.0)
            )
        ).alias("weight")
        
        df = df.with_columns([tier_expr, weight_expr])
        
        # 6. Cleanup Columns
        # Normalize Keys (ticker -> symbol, cast conversions)
        df = sc.normalize_keys(df)
        
        # Keep diagnostic cols
        # normalize_keys ensures 'symbol' is present if ticker was present
        keep_cols = [
            "date", "symbol", 
            "is_observable", "is_tradable", "tier", "weight",
            "close", "dvol", "adv20", "adv60", "vol20", "zero_volume_60", "max_share20", "age_days"
        ]
        
        # validate strict schema for required cols? 
        # save() does strict check. build() returns superset.
        
        return df.select([c for c in keep_cols if c in df.columns])

    def save(self, df: pl.DataFrame, out_dir: str, version: str = "v2"):
        """
        Save UniverseFrame partitioned by year.
        Enforces constraints:
        - date is Date
        - symbol exists
        - (date, symbol) is unique
        """
        # Constraints Check (Strict)
        sc.assert_schema(
            df, 
            required_cols=sc.UNIVERSEFRAME_V2_REQUIRED_COLS,
            dtype_map={sc.DATE_COL: pl.Date, sc.SYMBOL_COL: pl.Utf8}
        )
        
        # Uniqueness Check (fast)
        # count rows vs count unique keys
        n_rows = len(df)
        n_keys = df.select(["date", "symbol"]).n_unique()
        if n_rows != n_keys:
             raise ValueError(f"UniverseFrame has duplicate (date, symbol) keys! Rows={n_rows}, Unique={n_keys}")

        # Add year column for partitioning
        df_save = df.with_columns(pl.col("date").dt.year().alias("year"))
        
        # Write Parquet with partitioning
        # backend/data_canonical/universe/universe_frame_v2/
        out_path = f"{out_dir}/universe_frame_{version}"
        
        import os
        os.makedirs(out_path, exist_ok=True)
        
        unique_years = df_save["year"].unique().to_list()
        for y in unique_years:
            partition = df_save.filter(pl.col("year") == y)
            path = f"{out_path}/year={y}/data.parquet"
            # Directory
            os.makedirs(os.path.dirname(path), exist_ok=True)
            partition.drop("year").write_parquet(path)
            
        logger.info(f"Saved UniverseFrame to {out_path} (Partitioned by Year)")
