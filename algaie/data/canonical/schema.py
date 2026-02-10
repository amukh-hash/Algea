from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class CanonicalDailySchema:
    required_columns: List[str] = field(
        default_factory=lambda: ["date", "open", "high", "low", "close", "volume"]
    )
    optional_columns: List[str] = field(
        default_factory=lambda: ["vwap", "vix", "rate_proxy"]
    )

    @property
    def columns(self) -> List[str]:
        return self.required_columns + self.optional_columns
