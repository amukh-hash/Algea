from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd

from algae.core.config import PipelineConfig
from algae.data.common import BaseValidationError, ValidationIssue


# Re-export for backward compatibility
ValidationError = BaseValidationError


def validate_canonical_daily(df: pd.DataFrame) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    if df.empty:
        issues.append(ValidationIssue("canonical daily is empty", []))
        return issues
    if "date" not in df.columns:
        issues.append(ValidationIssue("missing date column", []))
        return issues

    if df["date"].duplicated().any():
        dup_rows = df.index[df["date"].duplicated()].tolist()
        issues.append(ValidationIssue("duplicate dates", dup_rows))

    sorted_dates = pd.to_datetime(df["date"])
    if not sorted_dates.is_monotonic_increasing:
        issues.append(ValidationIssue("non-monotonic dates", df.index.tolist()))

    if (df["close"] <= 0).any():
        bad_rows = df.index[df["close"] <= 0].tolist()
        issues.append(ValidationIssue("close <= 0", bad_rows))

    high_violations = df["high"] < df[["open", "close"]].max(axis=1)
    if high_violations.any():
        issues.append(
            ValidationIssue("high below open/close", df.index[high_violations].tolist())
        )

    low_violations = df["low"] > df[["open", "close"]].min(axis=1)
    if low_violations.any():
        issues.append(
            ValidationIssue("low above open/close", df.index[low_violations].tolist())
        )

    zero_fill = (df[["open", "high", "low", "close", "volume"]] == 0).all(axis=1)
    if zero_fill.any():
        issues.append(ValidationIssue("zero-filled row", df.index[zero_fill].tolist()))

    return issues


def quarantine_invalid_returns(
    df: pd.DataFrame,
    config: PipelineConfig,
    report_path: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ratios = df["close"] / df["close"].shift(1)
    invalid = (ratios <= 0) | (ratios - 1 <= -1)
    invalid_rows = df.loc[invalid].copy()
    if not invalid_rows.empty:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        invalid_rows.to_parquet(report_path, index=False)
    valid_df = df.loc[~invalid].copy()
    invalid_frac = len(invalid_rows) / max(len(df), 1)
    if invalid_frac > config.max_invalid_frac:
        raise ValidationError(
            [ValidationIssue("invalid return fraction exceeded", invalid_rows.index.tolist())]
        )
    return valid_df, invalid_rows
