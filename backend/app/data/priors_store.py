
import os
import json
import hashlib
import polars as pl
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional

PRIORS_SCHEMA = {
    "date": pl.Date,
    "ticker": pl.Utf8,
    "p_mu5": pl.Float32,
    "p_mu10": pl.Float32,
    "p_sig5": pl.Float32,
    "p_sig10": pl.Float32,
    "p_pdown5": pl.Float32,
    "p_pdown10": pl.Float32,
}

def stable_hash_priors_metadata(metadata: Dict[str, Any]) -> str:
    """
    Produce a stable hash of the metadata dictionary.
    Excludes keys starting with '_' from hash.
    """
    clean_meta = {k: v for k, v in metadata.items() if not k.startswith("_")}
    # Sort keys for stability
    serialized = json.dumps(clean_meta, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]

def write_priors_frame(df: pl.DataFrame, output_dir: str, metadata: Dict[str, Any], version: str = "v1"):
    """
    Writes PriorsFrame to parquet (partitioned by date) and saves metadata.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Validate Schema
    # Cast/Check columns
    for col, dtype in PRIORS_SCHEMA.items():
        if col not in df.columns:
            raise ValueError(f"PriorsFrame missing column: {col}")
        # cast
        df = df.with_columns(pl.col(col).cast(dtype))
        
    # 2. Add Hash/Version to Metadata
    meta_hash = stable_hash_priors_metadata(metadata)
    metadata["_hash"] = meta_hash
    metadata["_created_at"] = str(pd.Timestamp.now())
    metadata["_version"] = version
    
    # 3. Save Metadata
    meta_path = os.path.join(output_dir, "priors_meta.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
        
    # 4. Save Data (Partitioned by Date)
    # We use hive partitioning: output_dir/date=YYYY-MM-DD/data.parquet
    # Ensure date is string for partitioning? Polars writes dates as YYYY-MM-DD string in partitioning?
    # Actually Polars write_parquet with partition_by is easier.
    
    # We want to append? Or overwrite?
    # Usually we build for a range and save.
    
    # To ensure atomicity/cleanliness, we might want to write to a staging area or 
    # just overwrite partitions.
    
    # Let's write dataset.
    # Note: partition_by might create many small files if dates are many. 
    # But usually we run this daily or batch.
    
    # Convert date to string for directory structure control?
    # partition_by supports Date type.
    
    # output_dir/data.parquet?
    # User requirement: "backend/data/priors/priors_frame.parquet (partition by date or ticker)"
    # Let's partition by date.
    
    # We use a dataset writer approach
    # use_pyarrow=True for dataset writing
    
    # Check if empty
    if df.height == 0:
        return
        
    try:
        # We manually partition to control filename?
        # Or simple:
        # df.write_parquet(output_dir, use_pyarrow=True, pyarrow_options={"partition_cols": ["date"]})
        # But this makes output_dir a directory of partitions.
        pass
        
    except Exception:
        pass

    # Polars native partition write
    # df.write_parquet(output_dir, partition_by="date")
    # This creates output_dir/date=.../xxx.parquet
    
    df.write_parquet(output_dir, partition_by="date")
    
def read_priors_frame(source: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pl.DataFrame:
    """
    Reads PriorsFrame. Can accept a directory (partitioned) or file.
    Optionally filters by date range.
    """
    # Scan
    try:
        if os.path.isdir(source):
            lf = pl.scan_parquet(os.path.join(source, "**", "*.parquet"))
        else:
            lf = pl.scan_parquet(source)
            
        if start_date:
            lf = lf.filter(pl.col("date") >= pl.lit(start_date).cast(pl.Date))
        if end_date:
            lf = lf.filter(pl.col("date") <= pl.lit(end_date).cast(pl.Date))
            
        return lf.collect()
        
    except Exception as e:
        # If no files found
        print(f"Warning: Could not read priors from {source}: {e}")
        return pl.DataFrame(schema=PRIORS_SCHEMA)
