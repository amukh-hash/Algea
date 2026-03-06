from datetime import date

import pandas as pd

from algae.data.eligibility.build import build_eligibility
from algae.research.backtest_engine import BacktestConfig, BacktestEngine
from algae.trading.portfolio import Portfolio, Position
from algae.trading.risk import PortfolioTargetBuilder, PortfolioTargetConfig


def test_multiday_targets_exist_on_many_dates():
    canonical = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"] * 2),
            "ticker": ["A", "A", "A", "B", "B", "B"],
            "close": [10, 11, 12, 20, 21, 22],
            "volume": [100] * 6,
        }
    )
    signals = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"] * 2),
            "ticker": ["A", "A", "A", "B", "B", "B"],
            "score": [1, 1, 1, 2, 2, 2],
            "rank": [2, 2, 2, 1, 1, 1],
        }
    )
    elig = build_eligibility(canonical, date(2024, 1, 3))
    builder = PortfolioTargetBuilder(PortfolioTargetConfig())
    counts = []
    for d in ["2024-01-01", "2024-01-02", "2024-01-03"]:
        t = builder.build_targets(signals, elig, canonical, pd.Timestamp(d), total_equity=1000)
        counts.append(len(t))
    assert counts[0] > 0 and counts[1] > 0 and counts[2] > 0


def test_empty_targets_triggers_liquidation():
    engine = BacktestEngine(PortfolioTargetConfig(), BacktestConfig(exit_policy="signal"))
    portfolio = Portfolio(cash=0.0, positions={"A": Position("A", 1, 100.0, date(2024, 1, 1))})
    out = engine._apply_exit_policy(pd.DataFrame(columns=["date", "ticker", "target_weight", "score", "rank"]), portfolio, date(2024, 1, 5))
    assert not out.empty
    assert out.iloc[0]["ticker"] == "A"
    assert float(out.iloc[0]["target_weight"]) == 0.0
