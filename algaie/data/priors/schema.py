from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class PriorsSchema:
    required_columns: List[str] = (
        "date",
        "ticker",
        "p_mu5",
        "p_mu10",
        "p_sig5",
        "p_sig10",
        "p_pdown5",
        "p_pdown10",
    )
