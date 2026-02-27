from __future__ import annotations


def build_rollout(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda r: r["timestamp"])
