"""Shared utilities for data validation and I/O across data sub-modules."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Shared validation primitives (issues #1, #2)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ValidationIssue:
    """A single validation problem with affected row indices."""
    message: str
    rows: List[int]


class BaseValidationError(RuntimeError):
    """Base error for all data validation failures."""

    def __init__(self, issues: Iterable[ValidationIssue]) -> None:
        self.issues = list(issues)
        message = "; ".join(issue.message for issue in self.issues)
        super().__init__(message)


# ---------------------------------------------------------------------------
# Shared helpers (issues #6-8, #39, #42, #43)
# ---------------------------------------------------------------------------

def find_non_finite_rows(df: pd.DataFrame, columns: pd.DataFrame | None = None) -> np.ndarray:
    """Return boolean array of rows with non-finite values.

    Computes ``np.isfinite`` only once instead of twice (issue #6/#7/#8).
    """
    numeric = columns if columns is not None else df.select_dtypes(include=[np.number])
    if numeric.empty:
        return np.zeros(len(df), dtype=bool)
    arr = numeric.to_numpy()
    return ~np.isfinite(arr).all(axis=1)


def get_close_column(df: pd.DataFrame) -> str:
    """Return 'close_adj' if present, else 'close' (issue #42)."""
    if "close_adj" in df.columns:
        return "close_adj"
    if "close" in df.columns:
        return "close"
    raise ValueError("DataFrame must contain 'close_adj' or 'close' column")


def ensure_datetime(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    """Ensure *col* is datetime, converting in-place if needed (issue #43)."""
    if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
        df[col] = pd.to_datetime(df[col])
    return df


def write_dataframe(df: pd.DataFrame, destination: Path) -> None:
    """Write DataFrame to parquet with automatic directory creation (issue #41)."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(destination, index=False)
