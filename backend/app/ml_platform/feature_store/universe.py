from __future__ import annotations

import hashlib


def canonical_universe(universe_id: str, symbols: list[str]) -> list[str]:
    return sorted({s.strip().upper() for s in symbols if s and s.strip()})


def universe_hash(universe_id: str, symbols: list[str]) -> str:
    stable = canonical_universe(universe_id, symbols)
    payload = f"{universe_id}|{'|'.join(stable)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
