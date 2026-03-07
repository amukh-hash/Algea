from __future__ import annotations

from typing import List

import pandas as pd

from algae.data.common import BaseValidationError, ValidationIssue, find_non_finite_rows
from algae.models.foundation.base import PRIORS_REQUIRED_COLUMNS


# Re-export for backward compatibility
PriorsIssue = ValidationIssue
PriorsValidationError = BaseValidationError


def validate_priors_frame(df: pd.DataFrame) -> None:
    issues: List[ValidationIssue] = []

    # Empty frame is valid — no data means no leakage
    if df.empty:
        return

    # Schema presence check
    missing = [c for c in PRIORS_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise PriorsValidationError(
            [ValidationIssue(f"missing required priors columns: {missing}", [])]
        )

    # Single isfinite pass instead of computing twice
    non_finite = find_non_finite_rows(df)
    if non_finite.any():
        issues.append(ValidationIssue("non-finite priors", df.index[non_finite].tolist()))

    # Validate sigma is positive
    neg_sigma = (df["p_sig5"] <= 0) | (df["p_sig10"] <= 0)
    if neg_sigma.any():
        issues.append(ValidationIssue("non-positive sigma", df.index[neg_sigma].tolist()))

    for col in ["p_pdown5", "p_pdown10"]:
        oob = (df[col] < 0) | (df[col] > 1)
        if oob.any():
            issues.append(ValidationIssue("p_down out of bounds", df.index[oob].tolist()))

    if issues:
        raise PriorsValidationError(issues)
