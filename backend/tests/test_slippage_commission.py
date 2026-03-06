import pandas as pd

from algae.research.backtest_engine import BacktestConfig, BacktestEngine
from algae.trading.costs import CommissionModel, SlippageModel
from algae.trading.risk import PortfolioTargetConfig


def test_slippage_and_commission_reduce_equity():
    signals = pd.DataFrame(
        {
            "date": ["2024-01-02"],
            "ticker": ["AAA"],
            "score": [1.0],
            "rank": [1],
        }
    )
    eligibility = pd.DataFrame(
        {
            "date": ["2024-01-02"],
            "ticker": ["AAA"],
            "is_eligible": [True],
        }
    )
    prices = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03"],
            "ticker": ["AAA", "AAA"],
            "open": [10.0, 10.0],
            "close": [10.0, 10.0],
            "volume": [1000.0, 1000.0],
        }
    )
    base_engine = BacktestEngine(
        PortfolioTargetConfig(top_k=1, max_names=1, cash_buffer_pct=0.0),
        BacktestConfig(fill_price_mode="close", rounding_policy="fractional"),
    )
    cost_engine = BacktestEngine(
        PortfolioTargetConfig(top_k=1, max_names=1, cash_buffer_pct=0.0),
        BacktestConfig(
            fill_price_mode="close",
            rounding_policy="fractional",
            slippage_model=SlippageModel(model="bps", bps=10.0),
            commission_model=CommissionModel(per_trade=1.0),
        ),
    )
    base_result = base_engine.run(signals, eligibility, prices, pd.Timestamp("2024-01-02").date(), pd.Timestamp("2024-01-02").date())
    cost_result = cost_engine.run(signals, eligibility, prices, pd.Timestamp("2024-01-02").date(), pd.Timestamp("2024-01-02").date())
    assert cost_result.equity_curve["equity"].iloc[-1] <= base_result.equity_curve["equity"].iloc[-1]
