from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PriorsIssue:
    message: str
    rows: List[int]


class PriorsValidationError(RuntimeError):
    def __init__(self, issues: List[PriorsIssue]) -> None:
        self.issues = issues
        super().__init__("; ".join(issue.message for issue in issues))


def validate_priors_frame(df: pd.DataFrame) -> None:
    issues: List[PriorsIssue] = []
    numeric = df.select_dtypes(include=[np.number])
    if not np.isfinite(numeric.to_numpy()).all():
        bad_rows = df.index[~np.isfinite(numeric.to_numpy()).all(axis=1)].tolist()
        issues.append(PriorsIssue("non-finite priors", bad_rows))
    if (df[["p_sig5", "p_sig10"]] <= 0).any().any():
        bad_rows = df.index[(df["p_sig5"] <= 0) | (df["p_sig10"] <= 0)].tolist()
        issues.append(PriorsIssue("non-positive sigma", bad_rows))
    for col in ["p_pdown5", "p_pdown10"]:
        if ((df[col] < 0) | (df[col] > 1)).any():
            bad_rows = df.index[(df[col] < 0) | (df[col] > 1)].tolist()
            issues.append(PriorsIssue("p_down out of bounds", bad_rows))
    if issues:
        raise PriorsValidationError(issues)
