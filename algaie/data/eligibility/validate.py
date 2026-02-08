from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd


@dataclass(frozen=True)
class EligibilityIssue:
    message: str
    rows: List[int]


class EligibilityValidationError(RuntimeError):
    def __init__(self, issues: List[EligibilityIssue]) -> None:
        self.issues = issues
        super().__init__("; ".join(issue.message for issue in issues))


def validate_eligibility_frame(df: pd.DataFrame) -> None:
    issues: List[EligibilityIssue] = []
    if df.empty:
        issues.append(EligibilityIssue("eligibility frame empty", []))
    if df["date"].isna().any():
        issues.append(EligibilityIssue("missing dates", df.index[df["date"].isna()].tolist()))
    if df["ticker"].isna().any():
        issues.append(
            EligibilityIssue("missing tickers", df.index[df["ticker"].isna()].tolist())
        )
    if "is_eligible" not in df.columns:
        issues.append(EligibilityIssue("missing is_eligible", []))
    if issues:
        raise EligibilityValidationError(issues)
