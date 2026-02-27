from __future__ import annotations

import hashlib


def build_panel_universe(symbols: list[str], max_assets: int = 64) -> list[str]:
    stable = sorted({s.strip().upper() for s in symbols if s and s.strip()})
    return stable[:max_assets]


def panel_universe_hash(symbols: list[str]) -> str:
    stable = build_panel_universe(symbols, max_assets=10_000)
    return hashlib.sha256("|".join(stable).encode("utf-8")).hexdigest()
