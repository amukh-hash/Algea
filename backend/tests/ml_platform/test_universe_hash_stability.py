from backend.app.ml_platform.feature_store.universe import universe_hash


def test_universe_hash_stable_sorting():
    h1 = universe_hash("u", ["MSFT", "AAPL"])
    h2 = universe_hash("u", ["AAPL", "MSFT", "AAPL"])
    assert h1 == h2
