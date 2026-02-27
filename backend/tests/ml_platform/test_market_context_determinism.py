from backend.app.ml_platform.feature_store.market_context import compute_market_context


def test_market_context_deterministic():
    s = [100 + i for i in range(30)]
    assert compute_market_context("2026-01-01", s, 0.5) == compute_market_context("2026-01-01", s, 0.5)
