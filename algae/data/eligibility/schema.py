from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class EligibilitySchema:
    required_columns: List[str] = ("date", "ticker", "is_eligible")
    optional_columns: List[str] = ("reason_codes",)
