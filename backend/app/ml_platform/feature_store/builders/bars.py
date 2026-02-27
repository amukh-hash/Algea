from __future__ import annotations

from typing import Iterable


def build_bars_dataset(rows: Iterable[dict]) -> list[dict]:
    return sorted(rows, key=lambda r: (r.get("timestamp"), r.get("symbol")))
