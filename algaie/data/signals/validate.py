from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SignalIssue:
    message: str
    rows: List[int]


class SignalValidationError(RuntimeError):
    def __init__(self, issues: List[SignalIssue]) -> None:
        self.issues = issues
        super().__init__("; ".join(issue.message for issue in issues))


def validate_signal_frame(df: pd.DataFrame) -> None:
    issues: List[SignalIssue] = []
    numeric = df[["score", "rank"]]
    if not np.isfinite(numeric.to_numpy()).all():
        bad_rows = df.index[~np.isfinite(numeric.to_numpy()).all(axis=1)].tolist()
        issues.append(SignalIssue("non-finite scores", bad_rows))
    if issues:
        raise SignalValidationError(issues)
