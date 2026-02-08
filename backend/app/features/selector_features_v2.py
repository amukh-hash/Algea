"""
SelectorFeatureFrame V2 Builder
Deterministic, point-in-time safe feature engineering for Rank Selector model.
Inputs: Adjusted OHLCV, UniverseFrame V2
Outputs: Normalized FeatureFrame (Parquet)
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict
import numpy as np
import polars as pl
import json
import os
from backend.app.ops import run_recorder, pathmap
from backend.app.data import schema_contracts as sc

logger = logging.getLogger(__name__)

@dataclass
class SelectorFeatureConfig:
    """Configuration for Selector Feature Generation"""
    min_breadth_train: int = 200
    horizon_days: int = 5
    lookback_window: int = 20
    volatility_window: int = 20
    relative_vol_window: int = 20
    tie_break_policy: str = "sort_value_then_symbol"

class SelectorFeatureBuilder:
    def __init__(self, config: SelectorFeatureConfig):
        self.config = config

    def build_feature_frame(self, 
                          ohlcv_path: Path, 
                          universe_path: Path, 
                          output_dir: Path,
                          run_id: Optional[str] = None) -> None:
        """
        Main pipeline execution:
        1. Load & Join Data
        2. Compute Raw Features
        3. Filter by Universe (Tradable)
        4. Cross-Sectional Normalization
        5. Generate Targets
        6. Persist Artifacts
        """
        logger.info(f"Building SelectorFeatureFrame V2 with config: {self.config}")

        # 1. Load Data
        ohlcv = pl.scan_parquet(str(ohlcv_path))
        
        # Enforce symbol key in OHLCV
        if "ticker" in ohlcv.collect_schema().names():
            ohlcv = ohlcv.rename({"ticker": "symbol"})

        # Load Universe (Recursive)
        universe = pl.scan_parquet(str(universe_path) + "/**/*.parquet", hive_partitioning=True)

        # Enforce symbol key in Universe
        if "ticker" in universe.collect_schema().names():
             universe = universe.rename({"ticker": "symbol"})

        # Left join universe onto OHLCV (or inner join?)
        # We only care about universe days, but we need history for features.
        # Strategy: Compute features on FULL OHLCV first, THEN join & filter.
        
        # 2. Compute Raw Features (Lazy)
        logger.info("Computing raw features...")
        features_lf = self._compute_raw_features(ohlcv)
        
        # Cast features date to Date to match Universe (if universe is Date)
        features_lf = features_lf.with_columns(pl.col("date").cast(pl.Date))
        
        # 3. Join & Filter
        # 3. Join & Filter
        logger.info("Joining with UniverseFrame...")
        combined_lf = features_lf.join(
            universe, on=["date", "symbol"], how="inner"
        ).filter(
            pl.col("is_tradable") == True
        )
        
        # Collect to Eager for Grouped Normalization (Polars window functions are good, but rank mapping needs care)
        # We need to drop days with insufficient breadth.
        df = combined_lf.collect()
        
        # 4. Small-N Handling
        logger.info("Handling small-N days...")
        breadth = df.group_by("date").count()
        valid_days = breadth.filter(pl.col("count") >= self.config.min_breadth_train).select("date")
        
        dropped_days_count = len(breadth) - len(valid_days)
        logger.info(f"Dropping {dropped_days_count} days with breadth < {self.config.min_breadth_train}")
        
        df = df.join(valid_days, on="date", how="inner")
        
        # 5. Cross-Sectional Normalization
        logger.info("Applying deterministic rank normalization...")
        df = self._apply_rank_normalization(df)
        
        # 6. Generate Targets
        logger.info(f"Generating targets (Horizon={self.config.horizon_days}d)...")
        # Targets need FUTURE data. 
        # We need to rejoin with original OHLCV to get future returns?
        # OR, we computed fwd returns in step 2 (shift -H)?
        # Let's compute fwd returns in step 2.
        
        df = self._compute_targets(df)
        
        # 7. Validation & Persist
        # Validate Bounds
        feature_cols = ["x_lr1", "x_lr5", "x_lr20", "x_vol", "x_relvol"]
        for col in feature_cols:
            min_val = df[col].min()
            max_val = df[col].max()
            if min_val < -1.001 or max_val > 1.001:
                logger.error(f"Feature {col} out of bounds: [{min_val}, {max_val}]")
                # We can raise or warn. Hard fail as per user request.
                raise ValueError(f"Feature {col} out of bounds: [{min_val}, {max_val}]")
        
        
        # Normalize Keys (Defensive)
        df = sc.normalize_keys(df)
        
        # Enforce Constraints
        # 1. Check Schema
        sc.assert_schema(
            df,
            required_cols=sc.SELECTOR_FEATURES_V2_REQUIRED_COLS,
            dtype_map={sc.DATE_COL: pl.Date, sc.SYMBOL_COL: pl.Utf8}
        )
        
        # 2. Check Bounds (Strict)
        # Check if x_ cols are within [-1.001, 1.001]
        x_cols = [c for c in df.columns if c.startswith("x_")]
        for col in x_cols:
             min_val = df[col].min()
             max_val = df[col].max()
             if min_val < -1.001 or max_val > 1.001:
                  logger.error(f"Feature {col} out of bounds: [{min_val}, {max_val}]")
                  raise ValueError(f"Feature {col} out of bounds: [{min_val}, {max_val}]")
        
        # Add year for partitioning
        df = df.with_columns(pl.col("date").dt.year().alias("year"))
        
        output_dir.mkdir(parents=True, exist_ok=True)
        # Partitioned Write
        # Using partition_by is cleaner but loop is safer for memory if large
        unique_years = df["year"].unique().to_list()
        for y in unique_years:
            partition = df.filter(pl.col("year") == y)
            path = output_dir / f"year={y}" / "part-0.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            partition.drop("year").write_parquet(path)
            
        logger.info(f"Saved FeatureFrame to {output_dir} (Partitioned by Year)")
        
        # Summary And Metadata
        summary = {
            "schema_signature": sc.schema_signature(df),
            "total_rows": len(df),
            "unique_dates": df["date"].n_unique(),
            "n_symbols": df.select("symbol").n_unique(),
            "min_date": str(df["date"].min()),
            "max_date": str(df["date"].max()),
            "x_cols": x_cols,
            "bounds_check": "passed",
            "median_breadth": float(breadth["count"].median()),
            "dropped_days": dropped_days_count,
            "config": self.config.__dict__
        }
        
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

        if run_id:
            run_recorder.register_artifact(
                run_id,
                name="selector_feature_frame_v2",
                type="dataset",
                path=str(output_dir)
            )

    def _compute_raw_features(self, ohlcv: pl.LazyFrame) -> pl.LazyFrame:
        """
        Compute causal raw features and future returns for targets.
        Using window functions over ticker group.
        """
        # Sort by symbol, date to ensure rolling windows work correctly (if not using 'by')
        # However, .over("symbol") handles grouping.
        # But rolling_sum doesn't inherently sort by date within group unless data is sorted.
        
        return ohlcv.sort(["symbol", "date"]).with_columns([
             # 1. Log Returns: ln(close / close_lag1)
             (pl.col("close") / pl.col("close").shift(1).over("symbol")).log().alias("log_return_1d")
        ]).with_columns([
            # 2. Rolling Returns
            pl.col("log_return_1d").rolling_sum(window_size=5).over("symbol").alias("log_return_5d"),
            pl.col("log_return_1d").rolling_sum(window_size=20).over("symbol").alias("log_return_20d"),
            
            # 3. Volatility (20d std)
            pl.col("log_return_1d").rolling_std(window_size=self.config.volatility_window).over("symbol").alias("volatility_20d"),
            
            # 4. Relative Volume
            # median of shifted volume
            (pl.col("volume") / (
                pl.col("volume").shift(1).rolling_median(window_size=self.config.relative_vol_window).over("symbol") + 1.0
            )).alias("relative_volume_20d"),
            
            # 5. Targets (Future)
            (pl.col("close").shift(-self.config.horizon_days).over("symbol") / pl.col("close")).log().alias("target_return_raw"),
            
            # Future Volatility for Risk Adj
            pl.col("log_return_1d").rolling_std(window_size=self.config.horizon_days).shift(-self.config.horizon_days).over("symbol").alias("fwd_volatility")
        ])

    def _apply_rank_normalization(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Apply Cross-Sectional Rank Normalization maps to [-1, 1].
        Deterministic tie-breaking using (value, symbol).
        """
        # Features to normalize
        features = ["log_return_1d", "log_return_5d", "log_return_20d", "volatility_20d", "relative_volume_20d"]
        
        # Pre-process: Invert Volatility
        df = df.with_columns(
            (-pl.col("volatility_20d")).alias("vol_signal")
        )
        
        norm_features = ["log_return_1d", "log_return_5d", "log_return_20d", "vol_signal", "relative_volume_20d"]
        out_names = ["x_lr1", "x_lr5", "x_lr20", "x_vol", "x_relvol"]
        
        # Function to apply per-group (date)
        # Using map_groups might be cleaner than over() expression loops if complex
        # But looping over features via expression is efficient.
        
        exprs = []
        for feat, out_name in zip(norm_features, out_names):
            # Construct rank expression per date
            # struct(feat, symbol).rank("ordinal") breaks ties by symbol!
            # rank 1..N
            # r = rank - 1 => 0..N-1
            # N_t = count over date
            
            r = pl.struct([feat, "symbol"]).rank("ordinal").over("date") - 1
            N_t = pl.col("symbol").count().over("date")
            
            # Map to [-1, 1]
            norm = (2.0 * (r / (N_t - 1)) - 1.0).alias(out_name)
            exprs.append(norm)
            
        return df.with_columns(exprs)

    def _compute_targets(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Compute training targets: y_rank, y_trade.
        """
        # y_rank = target_return_raw (computed in raw features)
        df = df.rename({"target_return_raw": "y_rank"})
        
        # y_trade logic:
        # Risk Adj Return = y_rank / (fwd_vol + 1e-8)
        # Threshold: >= 70th percentile OF THE DAY
        
        # Fallback for fwd_vol: if null, use volatility_20d (current)
        fwd_vol = pl.coalesce([pl.col("fwd_volatility"), pl.col("volatility_20d")])
        
        # Calculate Risk Adjusted Return
        risk_adj = (pl.col("y_rank") / (fwd_vol + 1e-8)).alias("risk_adj_return")
        
        df = df.with_columns(risk_adj)
        
        # Determine 70th Percentile Cutoff per Date
        # Rank the risk_adj return within date
        # rank 1..N
        # pct = (rank - 1) / (N - 1)
        # if pct >= 0.70 => Top 30%
        
        r = pl.struct(["risk_adj_return", "symbol"]).rank("ordinal").over("date") - 1
        N_t = pl.col("symbol").count().over("date")
        rank_pct = r / (N_t - 1)
        
        y_trade = (rank_pct >= 0.70).cast(pl.Int32).alias("y_trade")
        
        # Handle Nans: If y_rank is null (end of history), y_trade should be null?
        # Or 0? Usually training drops last H rows anyway.
        # Let's keep Nans as is, downstream loader will filter or dropna.
        
        return df.with_columns([
            y_trade
        ])

if __name__ == "__main__":
    import argparse
    import os
    import sys
    from backend.app.ops import pathmap
    
    logging.basicConfig(level=logging.INFO)
    
    paths = pathmap.get_paths()
    DEFAULT_OHLCV = pathmap.resolve("ohlcv_adj", ticker="*").replace("ticker=*", "") # Hacky? No, pathmap doesn't expose raw_ohlcv directly. 
    # Use pathmap.get_paths().data_canonical / "ohlcv_adj" ?? 
    # Actually pathmap doesn't expose raw_ohlcv global file. 
    # Let's use the explicit raw_ohlcv path if it exists, as before, or derive from pathmap.
    # But for universe and output we have helpers.
    
    ROOT = Path(os.getcwd())
    # Try to find raw_ohlcv in artifacts or canonical
    # We'll default to the one in artifacts/universe as per previous script
    DEFAULT_OHLCV = ROOT / "backend/data/artifacts/universe/raw_ohlcv.parquet"
    
    DEFAULT_UNIVERSE = pathmap.get_universe_frame_root(paths, version="v2")
    DEFAULT_OUTPUT = pathmap.get_selector_features_root(paths, version="v2")
    
    parser = argparse.ArgumentParser(description="Build SelectorFeatureFrame V2")
    parser.add_argument("--start", type=str, default="2006-01-01", help="Start Date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2025-12-31", help="End Date (YYYY-MM-DD)")
    parser.add_argument("--horizon", type=int, default=5, help="Forecast Horizon (days)")
    parser.add_argument("--ohlcv", type=Path, default=DEFAULT_OHLCV, help="Path to raw_ohlcv.parquet")
    parser.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE, help="Path to UniverseFrame root")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output directory")
    
    args = parser.parse_args()
    
    cfg = SelectorFeatureConfig(
        horizon_days=args.horizon
    )
    
    builder = SelectorFeatureBuilder(cfg)
    
    run_id = None
    try:
        # Convert args to dict with string paths for JSON safety
        safe_config = {k: str(v) if isinstance(v, Path) else v for k, v in args.__dict__.items()}
        
        run_id = run_recorder.init_run(
            pipeline_type="selector_features_v2",
            trigger="manual",
            config=safe_config,
            data_versions={"universe": "v2", "ohlcv": "raw"},
            tags=["feature_engineering"]
        )
        run_recorder.set_status(run_id, "RUNNING", stage="build")
    except Exception as e:
        logger.warning(f"Run recorder init failed: {e}")

    try:
        builder.build_feature_frame(
            ohlcv_path=args.ohlcv,
            universe_path=args.universe,
            output_dir=args.output,
            run_id=run_id
        )
        if run_id:
            run_recorder.finalize_run(run_id, "PASSED")
            
    except Exception as e:
        logger.error(f"Build failed: {e}")
        if run_id:
            run_recorder.set_status(run_id, "FAILED", error={"message": str(e)})
            run_recorder.finalize_run(run_id, "FAILED")
        sys.exit(1)
