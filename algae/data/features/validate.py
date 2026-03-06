"""
Feature validation utilities.

Includes:
  - validate_feature_frame: scan for non-finite values (existing stub)
  - validate_df: schema, dtype, and null checks (from deprecated validators.py)
  - enforce_unique: duplicate-key assertion (from deprecated validators.py)
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set

import pandas as pd

from algae.data.common import BaseValidationError, ValidationIssue, find_non_finite_rows


# Re-export for backward compatibility
FeatureIssue = ValidationIssue
FeatureValidationError = BaseValidationError


def validate_feature_frame(df: pd.DataFrame, strict: bool = False) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    if df.empty:
        issues.append(ValidationIssue("feature frame empty", []))

    # Single isfinite pass instead of computing twice
    non_finite = find_non_finite_rows(df)
    if non_finite.any():
        issues.append(ValidationIssue("non-finite values", df.index[non_finite].tolist()))

    if strict and issues:
        raise FeatureValidationError(issues)
    return issues


# ---------------------------------------------------------------------------
# Schema / dtype / null validation  (from deprecated/features/validators.py)
# ---------------------------------------------------------------------------

# Dtype matcher lookup replaces verbose if/elif chain
_DTYPE_MATCHERS = {
    "float": "float",
    "int": "int",
    "string": ("object", "string"),
    "datetime": "datetime",
}


def validate_df(
    df: pd.DataFrame,
    schema: Dict[str, str],
    context: str = "",
    strict: bool = True,
    nullable: Optional[Set[str]] = None,
) -> None:
    """
    Validate DataFrame columns, dtypes, and nulls.

    Parameters
    ----------
    df : DataFrame to check
    schema : ``{column_name: expected_dtype_keyword}`` — dtype keywords are
             loose matches (e.g. ``"float"``, ``"int"``, ``"string"``).
    context : label for error messages
    strict : raise ``ValueError`` on mismatch (vs. print warning)
    nullable : set of column names allowed to contain NaN
    """
    if df.empty:
        return

    nullable = nullable or set()

    # 1. Column presence
    missing = [c for c in schema if c not in df.columns]
    if missing:
        msg = f"[{context}] Missing required columns: {missing}"
        if strict:
            raise ValueError(msg)

    # 2. Dtype (loose) — uses lookup dict instead of chained ifs
    for col, expected in schema.items():
        if col not in df.columns or not expected:
            continue
        dtype = str(df[col].dtype)
        match_val = _DTYPE_MATCHERS.get(expected)
        if match_val is None:
            continue
        if isinstance(match_val, tuple):
            ok = any(m in dtype for m in match_val)
        else:
            ok = match_val in dtype
        if not ok:
            msg = f"[{context}] Column '{col}' expected {expected}, got {dtype}"
            if strict:
                raise ValueError(msg)

    # 3. Null checks
    for col in schema:
        if col not in df.columns or col in nullable:
            continue
        if df[col].isnull().any():
            msg = f"[{context}] Column '{col}' contains NaNs but is not nullable."
            if strict:
                raise ValueError(msg)


def enforce_unique(df: pd.DataFrame, keys: List[str]) -> None:
    """Raise ``ValueError`` if duplicate rows exist for the given key columns."""
    if df.duplicated(subset=keys).any():
        raise ValueError(f"Duplicate entries found for keys: {keys}")
