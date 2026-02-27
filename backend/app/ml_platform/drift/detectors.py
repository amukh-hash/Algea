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


def prediction_consistency_score(current: list[float], baseline: dict) -> float:
    if not current:
        return 0.0
    mu = mean(current)
    base_mu = float(baseline.get("pred_mean", 0.0))
    base_std = float(baseline.get("pred_std", 1.0))
    base_std = base_std if base_std > 1e-8 else 1.0
    return abs(mu - base_mu) / base_std


def confidence_entropy_correlation(confidence: list[float], outcome_proxy: list[float]) -> float:
    n = min(len(confidence), len(outcome_proxy))
    if n < 2:
        return 0.0
    x = confidence[:n]
    y = outcome_proxy[:n]
    mx = sum(x) / n
    my = sum(y) / n
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    denx = sum((a - mx) ** 2 for a in x) ** 0.5
    deny = sum((b - my) ** 2 for b in y) ** 0.5
    den = denx * deny
    if den <= 1e-12:
        return 0.0
    return num / den
