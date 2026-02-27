from __future__ import annotations

from typing import List

import pandas as pd

from algea.data.common import BaseValidationError, ValidationIssue, find_non_finite_rows


# Re-export for backward compatibility
SignalIssue = ValidationIssue
SignalValidationError = BaseValidationError


def validate_signal_frame(df: pd.DataFrame) -> None:
    issues: List[ValidationIssue] = []

    # Single isfinite pass on specific columns instead of computing twice
    numeric = df[["score", "rank"]]
    non_finite = find_non_finite_rows(df, columns=numeric)
    if non_finite.any():
        issues.append(ValidationIssue("non-finite scores", df.index[non_finite].tolist()))

    if issues:
        raise SignalValidationError(issues)
