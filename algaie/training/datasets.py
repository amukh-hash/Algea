from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd


@dataclass(frozen=True)
class WindowConfig:
    context: int
    horizon: int


def build_valid_windows(canonical_daily: pd.DataFrame, config: WindowConfig) -> Dict[str, List[int]]:
    df = canonical_daily.sort_values(["ticker", "date"]).copy()
    df["ret_ratio"] = df.groupby("ticker")["close"].pct_change() + 1
    df["valid"] = df["ret_ratio"] > 0
    valid_windows: Dict[str, List[int]] = {}
    window_length = config.context + config.horizon
    for ticker, group in df.groupby("ticker"):
        valid_flags = group["valid"].fillna(False).to_numpy()
        indices: List[int] = []
        for idx in range(len(group) - window_length + 1):
            window = valid_flags[idx : idx + window_length]
            if window.all():
                indices.append(idx)
        valid_windows[ticker] = indices
    return valid_windows


def write_valid_windows(index: Dict[str, List[int]], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(index, indent=2), encoding="utf-8")
