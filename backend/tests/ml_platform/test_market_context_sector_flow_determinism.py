from backend.app.ml_platform.feature_store.market_context import compute_market_context


def test_market_context_sector_flow_determinism():
    kwargs = dict(
        asof="2026-01-01",
        market_series=[100, 101, 103, 102, 104, 106],
        breadth_proxy=0.61,
        sector_series={"XLK": [100, 102, 104, 105, 107, 109], "XLF": [100, 99, 100, 101, 102, 103]},
        sector_volume={"XLK": [10, 11, 12, 13, 14, 15], "XLF": [9, 9, 10, 10, 10, 11]},
    )
    a = compute_market_context(**kwargs)
    b = compute_market_context(**kwargs)
    assert a == b
