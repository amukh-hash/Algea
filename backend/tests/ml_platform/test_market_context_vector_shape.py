from backend.app.ml_platform.feature_store.market_context import compute_market_context
from backend.app.ml_platform.models.selector_master.context_encoder import encode_market_context


def test_market_context_shape():
    ctx = compute_market_context("2026-01-01", [100 + i for i in range(30)], 0.6, [0.1, 0.2])
    vec = encode_market_context(ctx)
    assert len(vec) == 6
