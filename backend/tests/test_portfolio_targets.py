import pandas as pd

from algea.trading.risk import PortfolioTargetBuilder, PortfolioTargetConfig


def test_portfolio_target_builder_respects_constraints():
    signals = pd.DataFrame(
        {
            "date": ["2024-01-02"] * 4,
            "ticker": ["AAA", "BBB", "CCC", "DDD"],
            "score": [2.0, 1.0, 0.5, -1.0],
            "rank": [1, 2, 3, 4],
        }
    )
    eligibility = pd.DataFrame(
        {
            "date": ["2024-01-02"] * 4,
            "ticker": ["AAA", "BBB", "CCC", "DDD"],
            "is_eligible": [True, True, False, True],
        }
    )
    prices = pd.DataFrame(
        {
            "date": ["2024-01-02"] * 4,
            "ticker": ["AAA", "BBB", "CCC", "DDD"],
            "close": [10.0, 10.0, 10.0, 10.0],
        }
    )
    config = PortfolioTargetConfig(top_k=2, max_weight_per_name=0.6, cash_buffer_pct=0.1)
    builder = PortfolioTargetBuilder(config)
    targets = builder.build_targets(
        signals=signals,
        eligibility=eligibility,
        prices=prices,
        asof=pd.Timestamp("2024-01-02"),
        total_equity=1000.0,
    )
    assert set(targets["ticker"]) == {"AAA", "BBB"}
    assert targets["target_weight"].max() <= 0.6
    assert abs(targets["target_weight"].sum() - 0.9) < 1e-6
