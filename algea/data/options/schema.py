"""
Strict schema definition and validation for EOD option chain snapshots.

Fail-fast on bad data: null IVs, inverted bid/ask, missing DTE coverage,
extreme IV values, or sparse chains.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Required columns and dtype expectations
# ---------------------------------------------------------------------------

OPTION_CHAIN_REQUIRED_COLS: List[str] = [
    "date",
    "underlying",
    "underlying_price",
    "expiry",
    "dte",
    "option_type",
    "strike",
    "bid",
    "ask",
    "mid",
    "implied_vol",
    "open_interest",
    "volume",
    "multiplier",
    "risk_free_rate",
    "dividend_yield",
]

# Reasonable bounds
_IV_MIN = 0.01
_IV_MAX = 5.0
_DTE_MIN_REQUIRED = 30
_DTE_MAX_REQUIRED = 45


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class ChainValidationError(ValueError):
    """Raised when option chain data fails validation."""


def validate_chain(
    df: pd.DataFrame,
    *,
    require_dte_coverage: bool = True,
    min_dte: int = _DTE_MIN_REQUIRED,
    max_dte: int = _DTE_MAX_REQUIRED,
    iv_min: float = _IV_MIN,
    iv_max: float = _IV_MAX,
    min_strikes_per_expiry: int = 5,
) -> pd.DataFrame:
    """Validate an option chain DataFrame.  Returns the frame unchanged if
    valid; raises ``ChainValidationError`` on any violation.

    Checks performed:
    1. Required columns present
    2. No null implied_vol
    3. implied_vol within bounds
    4. bid <= ask
    5. mid between bid and ask (tolerance 1e-6)
    6. DTE coverage includes [min_dte, max_dte] range
    7. Per-expiry minimum strike count (sparseness guard)
    """
    # 1. Columns
    missing = [c for c in OPTION_CHAIN_REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ChainValidationError(f"Missing columns: {missing}")

    if df.empty:
        raise ChainValidationError("Chain DataFrame is empty")

    # 2. Null IV
    null_iv = df["implied_vol"].isna().sum()
    if null_iv > 0:
        raise ChainValidationError(f"{null_iv} rows have null implied_vol")

    # 3. IV bounds
    iv = df["implied_vol"]
    oob = ((iv < iv_min) | (iv > iv_max)).sum()
    if oob > 0:
        raise ChainValidationError(
            f"{oob} rows have implied_vol outside [{iv_min}, {iv_max}]"
        )

    # 4. bid <= ask
    inverted = (df["bid"] > df["ask"] + 1e-8).sum()
    if inverted > 0:
        raise ChainValidationError(f"{inverted} rows have bid > ask")

    # 5. mid between bid and ask
    tol = 1e-6
    bad_mid = (
        (df["mid"] < df["bid"] - tol) | (df["mid"] > df["ask"] + tol)
    ).sum()
    if bad_mid > 0:
        raise ChainValidationError(
            f"{bad_mid} rows have mid outside [bid, ask]"
        )

    # 6. DTE coverage
    if require_dte_coverage:
        dte_vals = df["dte"].unique()
        has_coverage = any(min_dte <= d <= max_dte for d in dte_vals)
        if not has_coverage:
            raise ChainValidationError(
                f"No expiries with DTE in [{min_dte}, {max_dte}]. "
                f"Available DTE: {sorted(dte_vals)}"
            )

    # 7. Sparse chains
    per_exp = df.groupby("expiry")["strike"].nunique()
    sparse = per_exp[per_exp < min_strikes_per_expiry]
    if not sparse.empty:
        raise ChainValidationError(
            f"Sparse expiries (< {min_strikes_per_expiry} strikes): "
            f"{dict(sparse)}"
        )

    return df
