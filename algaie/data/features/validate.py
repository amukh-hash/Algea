from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureIssue:
    message: str
    rows: List[int]


class FeatureValidationError(RuntimeError):
    def __init__(self, issues: List[FeatureIssue]) -> None:
        self.issues = issues
        super().__init__("; ".join(issue.message for issue in issues))


def validate_feature_frame(df: pd.DataFrame, strict: bool = False) -> List[FeatureIssue]:
    issues: List[FeatureIssue] = []
    if df.empty:
        issues.append(FeatureIssue("feature frame empty", []))
    numeric = df.select_dtypes(include=[np.number])
    if not np.isfinite(numeric.to_numpy()).all():
        bad_rows = df.index[~np.isfinite(numeric.to_numpy()).all(axis=1)].tolist()
        issues.append(FeatureIssue("non-finite values", bad_rows))
    if strict and issues:
        raise FeatureValidationError(issues)
    return issues
