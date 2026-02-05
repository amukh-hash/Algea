
import pandas as pd
import numpy as np
from typing import Dict, List
from backend.app.ops import config

def validate_df(df: pd.DataFrame, schema: Dict, context: str = "", strict: bool = True) -> None:
    """
    Validates DataFrame columns and dtypes.
    """
    if df.empty:
        # Warn but allow?
        # Logic might rely on empty DF in some edge cases.
        return 

    req_cols = schema["columns"]
    nullable = set(schema.get("nullable", []))
    
    # 1. Column Presence
    missing = [c for c in req_cols if c not in df.columns]
    if missing:
        msg = f"[{context}] Missing required columns: {missing}"
        if strict or config.STRICT_SCHEMA:
            raise ValueError(msg)
        else:
            print(f"WARN: {msg}")

    # 2. Dtype Check (Loose)
    # Pandas dtypes are tricky.
    for col, expected in req_cols.items():
        if col not in df.columns: continue
        
        dtype = str(df[col].dtype)
        # Simplify checks
        is_float = "float" in dtype
        is_int = "int" in dtype
        is_obj = "object" in dtype or "string" in dtype
        is_dt = "datetime" in dtype
        
        # Validations
        # Validations
        if "float" in expected and not is_float:
             if strict or config.STRICT_SCHEMA:
                 raise ValueError(f"[{context}] Column '{col}' expected float, got {dtype}")
        if "int" in expected and not is_int:
             if strict or config.STRICT_SCHEMA:
                 raise ValueError(f"[{context}] Column '{col}' expected int, got {dtype}")
        if "string" in expected and not is_obj:
             if strict or config.STRICT_SCHEMA:
                 raise ValueError(f"[{context}] Column '{col}' expected string/object, got {dtype}")
        
    # 3. Null Checks
    for col in req_cols:
        if col not in df.columns: continue
        if col not in nullable:
            if df[col].isnull().any():
                msg = f"[{context}] Column '{col}' contains NaNs but is not nullable."
                if strict or config.STRICT_SCHEMA:
                    raise ValueError(msg)

def enforce_unique(df: pd.DataFrame, keys: List[str]) -> None:
    if df.duplicated(subset=keys).any():
        raise ValueError(f"Duplicate entries found for keys: {keys}")
