from __future__ import annotations


def encode_market_context(context: dict[str, float]) -> list[float]:
    keys = [
        "mkt_ret_5d",
        "mkt_vol_20d",
        "vix_proxy",
        "breadth_proxy",
        "sector_momentum_top1",
        "sector_momentum_top2",
        "sector_rel_strength",
        "volume_weighted_rel_return",
        "vol_regime",
    ]
    return [float(context.get(k, 0.0)) for k in keys]


def validate_market_context(context: dict[str, float]) -> None:
    required = {
        "mkt_ret_5d",
        "mkt_vol_20d",
        "breadth_proxy",
        "sector_rel_strength",
        "volume_weighted_rel_return",
        "vol_regime",
    }
    missing = sorted(k for k in required if k not in context)
    if missing:
        raise ValueError(f"market context required fields missing: {missing}")
