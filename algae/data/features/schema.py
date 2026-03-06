from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class FeatureSchema:
    required_columns: List[str] = ("date", "ticker")
