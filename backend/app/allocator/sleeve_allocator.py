from __future__ import annotations


def allocate_sleeve_gross(
    sleeve_metrics: dict[str, dict[str, float]],
    total_gross_cap: float = 1.0,
    sleeve_min: float = 0.0,
    sleeve_max: float = 0.7,
    max_turnover: float = 0.25,
    prev_allocations: dict[str, float] | None = None,
) -> dict[str, float]:
    prev_allocations = prev_allocations or {}
    raw = {}
    for sleeve, m in sorted(sleeve_metrics.items()):
        score = float(m.get("expected_return_proxy", 0.0)) - float(m.get("uncertainty", 0.0)) - float(m.get("drift", 0.0)) - float(m.get("drawdown", 0.0))
        raw[sleeve] = max(0.0, score)
    s = sum(raw.values()) or 1.0
    alloc = {k: min(sleeve_max, max(sleeve_min, total_gross_cap * v / s)) for k, v in raw.items()}
    for k, v in list(alloc.items()):
        prev = prev_allocations.get(k, v)
        delta = v - prev
        if abs(delta) > max_turnover:
            alloc[k] = prev + (max_turnover if delta > 0 else -max_turnover)
    return alloc
