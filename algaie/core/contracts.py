"""
Schema and feature contracts for the algaie data pipeline.

Combines:
  - Feature contracts (hash-based column validation) from deprecated/models/feature_contracts.py
  - Schema contracts (column presence + dtype enforcement) from deprecated/data/schema_contracts.py
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, List

import pandas as pd

try:
    import polars as pl

    _HAS_POLARS = True
except ImportError:  # pragma: no cover
    _HAS_POLARS = False


# ═══════════════════════════════════════════════════════════════════════════
# Schema constants
# ═══════════════════════════════════════════════════════════════════════════

DATE_COL = "date"
SYMBOL_COL = "symbol"

UNIVERSEFRAME_V2_REQUIRED_COLS: List[str] = [
    "date", "symbol",
    "is_observable", "is_tradable",
    "tier", "weight",
]

SELECTOR_FEATURES_V2_REQUIRED_COLS: List[str] = [
    "date", "symbol",
    "x_lr1", "x_lr5", "x_lr20", "x_vol", "x_relvol",
    "y_rank", "tier", "weight",
]

PRIORS_REQUIRED_COLS: List[str] = [
    "date", "symbol",
    "prior_drift_20d", "prior_vol_20d",
    "prior_downside_q10_20d", "prior_trend_conf_20d",
    "chronos_model_id", "context_len", "horizon", "prior_version",
]


# ═══════════════════════════════════════════════════════════════════════════
# Feature contract helpers (hash-based)
# ═══════════════════════════════════════════════════════════════════════════

def compute_contract_hash(columns: List[str]) -> str:
    """Stable SHA-256 hash (first 16 hex chars) of sorted column names."""
    sorted_cols = sorted(columns)
    joined = ",".join(sorted_cols)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def validate_contract(
    data: pd.DataFrame,
    contract_hash: str,
) -> bool:
    """Validate that the DataFrame columns match the expected contract hash (exact match)."""
    cols = list(data.columns)
    current_hash = compute_contract_hash(cols)
    return current_hash == contract_hash


# ═══════════════════════════════════════════════════════════════════════════
# Schema assertion helpers (column-presence + dtype)
# ═══════════════════════════════════════════════════════════════════════════

def normalize_keys(df: Any) -> Any:
    """
    Normalise DataFrame keys to canonical schema:
      * Rename ``ticker`` → ``symbol``
      * Cast ``date`` → Date
      * Cast ``symbol`` → Utf8 / str

    Supports both Polars and Pandas DataFrames.
    """
    if _HAS_POLARS and isinstance(df, pl.DataFrame):
        if "ticker" in df.columns and "symbol" not in df.columns:
            df = df.rename({"ticker": "symbol"})
        if "date" in df.columns and df.schema["date"] != pl.Date:
            df = df.with_columns(pl.col("date").cast(pl.Date))
        if "symbol" in df.columns and df.schema["symbol"] != pl.Utf8:
            df = df.with_columns(pl.col("symbol").cast(pl.Utf8))
        return df

    # Pandas path
    if isinstance(df, pd.DataFrame):
        if "ticker" in df.columns and "symbol" not in df.columns:
            df = df.rename(columns={"ticker": "symbol"})
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"]).dt.date
        return df

    raise TypeError(f"normalize_keys: unsupported type {type(df)}")


def assert_schema(
    df: Any,
    required_cols: List[str],
    dtype_map: Dict[str, Any] | None = None,
) -> None:
    """
    Assert that *df* contains *required_cols* (and optionally matches dtypes).

    Raises ``ValueError`` on mismatch.  Works with both Polars and Pandas.
    """
    cols = set(df.columns)

    missing = [c for c in required_cols if c not in cols]
    if missing:
        raise ValueError(f"Schema assertion failed. Missing columns: {missing}")

    if dtype_map:
        for col, expected_dtype in dtype_map.items():
            if col not in cols:
                continue
            if _HAS_POLARS and isinstance(df, pl.DataFrame):
                actual = df.schema[col]
                if actual != expected_dtype:
                    raise ValueError(
                        f"Schema assertion failed. Column '{col}' expected {expected_dtype}, got {actual}"
                    )
            elif isinstance(df, pd.DataFrame):
                actual = df[col].dtype
                if actual != expected_dtype:
                    raise ValueError(
                        f"Schema assertion failed. Column '{col}' expected {expected_dtype}, got {actual}"
                    )


def schema_signature(df: Any) -> List[Dict[str, str]]:
    """Return a JSON-serialisable list of ``{name, dtype}`` dicts."""
    if _HAS_POLARS and isinstance(df, pl.DataFrame):
        return [{"name": name, "dtype": str(dtype)} for name, dtype in df.schema.items()]
    if isinstance(df, pd.DataFrame):
        return [{"name": name, "dtype": str(dtype)} for name, dtype in df.dtypes.items()]
    raise TypeError(f"schema_signature: unsupported type {type(df)}")
