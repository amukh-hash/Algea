from __future__ import annotations


def _ret(series: list[float], window: int) -> float:
    if len(series) < 2:
        return 0.0
    idx = max(0, len(series) - 1 - window)
    base = series[idx] if series[idx] else 1.0
    return (series[-1] / base) - 1.0


def compute_sector_flow_proxies(
    market_series: list[float],
    sector_series: dict[str, list[float]],
    sector_volume: dict[str, list[float]],
    breadth_proxy: float,
) -> dict[str, float]:
    market_ret = _ret(market_series, 5)
    rels: list[float] = []
    vw_rels: list[float] = []
    for sector in sorted(sector_series):
        s_ret = _ret(sector_series[sector], 5)
        rel = s_ret - market_ret
        rels.append(rel)
        vols = sector_volume.get(sector, [])
        weight = float(vols[-1]) if vols else 1.0
        vw_rels.append(rel * max(weight, 1.0))
    strength = sum(rels) / max(len(rels), 1)
    vw_strength = sum(vw_rels) / max(sum(max(float((sector_volume.get(s, [1])[-1])), 1.0) for s in sorted(sector_series)), 1.0)
    realized_vol = 0.0
    if len(market_series) > 2:
        rets = [(market_series[i] / market_series[i - 1]) - 1.0 for i in range(1, len(market_series))]
        realized_vol = (sum(r * r for r in rets[-20:]) / max(1, len(rets[-20:]))) ** 0.5
    return {
        "sector_rel_strength": float(strength),
        "volume_weighted_rel_return": float(vw_strength),
        "breadth_proxy": float(breadth_proxy),
        "vol_regime": float(realized_vol),
    }
