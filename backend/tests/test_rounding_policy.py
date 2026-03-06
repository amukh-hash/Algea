import pandas as pd

from algae.trading.portfolio import Portfolio


def test_rounding_policy_rounds_shares():
    portfolio = Portfolio(cash=1000.0)
    targets = pd.DataFrame(
        {
            "ticker": ["AAA"],
            "target_weight": [0.5],
        }
    )
    prices = pd.DataFrame(
        {
            "date": ["2024-01-02"],
            "ticker": ["AAA"],
            "close": [30.0],
        }
    )
    intents = portfolio.build_order_intents(
        asof=pd.Timestamp("2024-01-02").date(),
        target_weights=targets,
        prices=prices,
        equity=1000.0,
        rounding_policy="round",
    )
    assert intents
    assert intents[0].quantity.is_integer()
