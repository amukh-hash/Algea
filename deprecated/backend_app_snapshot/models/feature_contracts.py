import hashlib
from typing import List
import polars as pl
import pandas as pd

def compute_contract_hash(columns: List[str]) -> str:
    """
    Computes a stable hash of the expected column names.
    Columns are sorted to ensure stability.
    """
    sorted_cols = sorted(columns)
    joined = ",".join(sorted_cols)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]

def validate_contract(data: pl.DataFrame | pd.DataFrame, contract_hash: str) -> bool:
    """
    Validates that the dataframe columns match the expected contract hash.
    Does NOT check data types (simplification), just column presence.
    """
    if isinstance(data, pd.DataFrame):
        cols = data.columns.tolist()
    else:
        cols = data.columns
        
    current_hash = compute_contract_hash(cols)
    
    # In a strict system, this returns False if mismatch.
    # For Algaie, we might allow supersets? 
    # Current impl: Exact match (after sorting).
    
    return current_hash == contract_hash
