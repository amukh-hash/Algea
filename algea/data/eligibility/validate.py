from __future__ import annotations

from typing import List

import pandas as pd

from algea.data.common import BaseValidationError, ValidationIssue


# Re-export for backward compatibility
EligibilityIssue = ValidationIssue
EligibilityValidationError = BaseValidationError


def validate_eligibility_frame(df: pd.DataFrame) -> None:
    issues: List[ValidationIssue] = []
    if df.empty:
        issues.append(ValidationIssue("eligibility frame empty", []))
    if df["date"].isna().any():
        issues.append(ValidationIssue("missing dates", df.index[df["date"].isna()].tolist()))
    if df["ticker"].isna().any():
        issues.append(
            ValidationIssue("missing tickers", df.index[df["ticker"].isna()].tolist())
        )
    if "is_eligible" not in df.columns:
        issues.append(ValidationIssue("missing is_eligible", []))
    if issues:
        raise EligibilityValidationError(issues)
