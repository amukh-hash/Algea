from __future__ import annotations

from statistics import mean, pstdev


def simple_drift_score(train_mean: float, live_mean: float) -> float:
    return abs(train_mean - live_mean)


def zscore_ood_score(series: list[float], baseline: dict) -> float:
    if not series:
        return 1.0
    live_mean = mean(series)
    base_mean = float(baseline.get("mean", live_mean))
    base_std = float(baseline.get("std", pstdev(series) if len(series) > 1 else 1.0))
    base_std = base_std if base_std > 1e-8 else 1.0
    return abs(live_mean - base_mean) / base_std


def tenor_drift_score(history: dict[int, list[dict]], baseline: dict) -> float:
    vals = []
    for tenor, rows in history.items():
        base = baseline.get(str(tenor), baseline.get(int(tenor), {})) if isinstance(baseline, dict) else {}
        bmean = float(base.get("rv_hist_20_mean", 0.0)) if isinstance(base, dict) else 0.0
        bstd = float(base.get("rv_hist_20_std", 1.0)) if isinstance(base, dict) else 1.0
        bstd = bstd if bstd > 1e-8 else 1.0
        for r in rows:
            vals.append(abs(float(r.get("rv_hist_20", 0.0)) - bmean) / bstd)
    return sum(vals) / max(len(vals), 1)
