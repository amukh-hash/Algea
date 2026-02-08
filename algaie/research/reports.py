from __future__ import annotations

from typing import Dict


def build_summary(metrics: Dict[str, Dict[str, float]]) -> str:
    lines = ["# Backtest Summary", ""]
    equity = metrics.get("equity", {})
    trades = metrics.get("trades", {})
    if equity:
        lines.append("## Equity Metrics")
        for key, value in equity.items():
            lines.append(f"- {key}: {value:.4f}")
        lines.append("")
    if trades:
        lines.append("## Trade Metrics")
        for key, value in trades.items():
            lines.append(f"- {key}: {value:.4f}")
        lines.append("")
    return "\n".join(lines)
