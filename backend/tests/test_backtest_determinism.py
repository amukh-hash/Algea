import pandas as pd

from algaie.research.backtest_engine import BacktestConfig, BacktestEngine
from algaie.trading.risk import PortfolioTargetConfig


def test_backtest_determinism():
    signals = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03"],
            "ticker": ["AAA", "AAA"],
            "score": [1.0, 1.0],
            "rank": [1, 1],
        }
    )
    eligibility = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03"],
            "ticker": ["AAA", "AAA"],
            "is_eligible": [True, True],
        }
    )
    prices = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "ticker": ["AAA", "AAA", "AAA"],
            "open": [10.0, 11.0, 12.0],
            "close": [10.0, 11.0, 12.0],
        }
    )
    engine = BacktestEngine(
        PortfolioTargetConfig(top_k=1, max_names=1, cash_buffer_pct=0.0),
        BacktestConfig(fill_price_mode="close", rounding_policy="fractional"),
    )
    result_one = engine.run(signals, eligibility, prices, pd.Timestamp("2024-01-02").date(), pd.Timestamp("2024-01-03").date())
    result_two = engine.run(signals, eligibility, prices, pd.Timestamp("2024-01-02").date(), pd.Timestamp("2024-01-03").date())
    pd.testing.assert_frame_equal(result_one.equity_curve, result_two.equity_curve)
