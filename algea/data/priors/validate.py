from __future__ import annotations

from typing import List

import pandas as pd

from algea.data.common import BaseValidationError, ValidationIssue, find_non_finite_rows


# Re-export for backward compatibility
PriorsIssue = ValidationIssue
PriorsValidationError = BaseValidationError


def validate_priors_frame(df: pd.DataFrame) -> None:
    issues: List[ValidationIssue] = []

    # Single isfinite pass instead of computing twice
    non_finite = find_non_finite_rows(df)
    if non_finite.any():
        issues.append(ValidationIssue("non-finite priors", df.index[non_finite].tolist()))

    # Compute conditions once, reuse the mask
    neg_sigma = (df["p_sig5"] <= 0) | (df["p_sig10"] <= 0)
    if neg_sigma.any():
        issues.append(ValidationIssue("non-positive sigma", df.index[neg_sigma].tolist()))

    for col in ["p_pdown5", "p_pdown10"]:
        oob = (df[col] < 0) | (df[col] > 1)
        if oob.any():
            issues.append(ValidationIssue("p_down out of bounds", df.index[oob].tolist()))

    if issues:
        raise PriorsValidationError(issues)
