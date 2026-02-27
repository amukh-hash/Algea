from __future__ import annotations

from statistics import pstdev


def deterministic_quantile_forecast(
    series: list[float], prediction_length: int, quantiles: list[float]
) -> dict[str, list[float]]:
    if not series:
        raise ValueError("series must be non-empty")
    last = float(series[-1])
    drift = (float(series[-1]) - float(series[0])) / max(len(series) - 1, 1)
    vol = pstdev(series) if len(series) > 1 else 0.0
    out: dict[str, list[float]] = {}
    for q in quantiles:
        q_key = f"{q:.2f}"
        spread = (q - 0.5) * 2.0 * max(vol, 1e-6)
        out[q_key] = [last + drift * (i + 1) + spread for i in range(prediction_length)]
    return out


def summarize_uncertainty(forecast: dict[str, list[float]]) -> dict[str, float]:
    q10 = forecast.get("0.10") or next(iter(forecast.values()))
    q90 = forecast.get("0.90") or next(reversed(forecast.values()))
    iqr = [abs(b - a) for a, b in zip(q10, q90)]
    if not iqr:
        return {"iqr_mean": 0.0, "iqr_max": 0.0}
    return {"iqr_mean": sum(iqr) / len(iqr), "iqr_max": max(iqr)}
