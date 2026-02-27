from __future__ import annotations

from .sector_flow import compute_sector_flow_proxies


def compute_market_context(
    asof: str,
    market_series: list[float],
    breadth_proxy: float,
    sector_mom: list[float] | None = None,
    sector_series: dict[str, list[float]] | None = None,
    sector_volume: dict[str, list[float]] | None = None,
) -> dict[str, float]:
    sector_mom = sector_mom or [0.0, 0.0]
    if len(market_series) < 2:
        base = {
            "mkt_ret_5d": 0.0,
            "mkt_vol_20d": 0.0,
            "vix_proxy": 0.0,
            "breadth_proxy": breadth_proxy,
            "sector_momentum_top1": sector_mom[0],
            "sector_momentum_top2": sector_mom[1] if len(sector_mom) > 1 else 0.0,
        }
        base.update(compute_sector_flow_proxies([], {}, {}, breadth_proxy))
        return base
    ret_5d = (market_series[-1] / market_series[max(0, len(market_series) - 6)]) - 1.0
    returns = [(market_series[i] / market_series[i - 1]) - 1.0 for i in range(1, len(market_series))]
    vol_20d = (sum(r * r for r in returns[-20:]) / max(1, min(20, len(returns)))) ** 0.5
    base = {
        "mkt_ret_5d": ret_5d,
        "mkt_vol_20d": vol_20d,
        "vix_proxy": vol_20d,
        "breadth_proxy": breadth_proxy,
        "sector_momentum_top1": sector_mom[0],
        "sector_momentum_top2": sector_mom[1] if len(sector_mom) > 1 else 0.0,
    }
    base.update(
        compute_sector_flow_proxies(
            market_series,
            sector_series or {},
            sector_volume or {},
            breadth_proxy,
        )
    )
    return base
