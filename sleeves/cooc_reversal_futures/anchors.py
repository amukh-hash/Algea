from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd


@dataclass(frozen=True)
class AnchorQuote:
    price: float
    method_used: str
    timestamp: datetime


def anchor_price(quotes: pd.DataFrame, anchor_ts: datetime, method: str) -> AnchorQuote:
    """Build anchor prices with deterministic fallback MID -> VWAP_WINDOW -> BID_ASK_EXEC."""
    before = quotes[quotes["timestamp"] <= anchor_ts].copy()
    if before.empty:
        raise ValueError("No quote available at/before anchor")
    row = before.iloc[-1]
    bid = float(row["bid"]) if "bid" in row and pd.notna(row["bid"]) else 0.0
    ask = float(row["ask"]) if "ask" in row and pd.notna(row["ask"]) else 0.0
    trade = float(row.get("price", (bid + ask) / 2.0))

    if method == "MID" and bid > 0 and ask > 0:
        return AnchorQuote((bid + ask) / 2.0, "MID", row["timestamp"])

    if method in {"MID", "VWAP_WINDOW"}:
        window = before[before["timestamp"] >= anchor_ts - pd.Timedelta(minutes=5)]
        if "volume" in window and window["volume"].sum() > 0:
            vwap = float((window["price"] * window["volume"]).sum() / window["volume"].sum())
            return AnchorQuote(vwap, "VWAP_WINDOW", row["timestamp"])

    if bid > 0 and ask > 0:
        return AnchorQuote(ask if method == "BID_ASK_EXEC" else (bid + ask) / 2.0, "BID_ASK_EXEC", row["timestamp"])
    return AnchorQuote(trade, "LAST", row["timestamp"])
