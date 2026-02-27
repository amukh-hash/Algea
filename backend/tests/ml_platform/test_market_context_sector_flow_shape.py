from backend.app.ml_platform.feature_store.market_context import compute_market_context


def test_market_context_sector_flow_shape():
    out = compute_market_context(
        asof="2026-01-01",
        market_series=[100, 101, 102, 101, 103, 104],
        breadth_proxy=0.55,
        sector_series={"XLK": [100, 101], "XLE": [100, 99]},
        sector_volume={"XLK": [10, 11], "XLE": [8, 9]},
    )
    for k in ["sector_rel_strength", "volume_weighted_rel_return", "breadth_proxy", "vol_regime"]:
        assert k in out
