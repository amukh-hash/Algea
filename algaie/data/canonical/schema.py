from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List


@dataclass(frozen=True)
class CanonicalDailySchema:
    required_columns: List[str] = (
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
    )
    optional_columns: List[str] = ("vwap", "vix", "rate_proxy")

    @property
    def columns(self) -> Iterable[str]:
        return list(self.required_columns) + list(self.optional_columns)
