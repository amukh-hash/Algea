from __future__ import annotations


def encode_market_context(context: dict[str, float]) -> list[float]:
    keys = [
        "mkt_ret_5d",
        "mkt_vol_20d",
        "vix_proxy",
        "breadth_proxy",
        "sector_momentum_top1",
        "sector_momentum_top2",
    ]
    return [float(context.get(k, 0.0)) for k in keys]
