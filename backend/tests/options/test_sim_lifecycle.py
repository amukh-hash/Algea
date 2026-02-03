import pytest
from datetime import datetime, timedelta
from backend.app.options.sim.credit_spread_sim import PositionSimulator
from backend.app.options.sim.policy import SimPolicy
from backend.app.options.types import SpreadCandidate

def test_sim_expiry_closure():
    cand = SpreadCandidate(
        underlying_ticker="AAPL",
        expiry_date="2023-01-10",
        dte=10,
        short_strike=100,
        long_strike=95,
        net_credit=1.0,
        width=5.0
    )
    policy = SimPolicy(close_at_dte=0)
    sim = PositionSimulator(cand, policy, datetime(2023, 1, 1))

    # 9 days later -> DTE 1
    sim.update(datetime(2023, 1, 9), 105.0) # OTM
    assert sim.is_open

    # 10 days later -> DTE 0 (Expiry)
    sim.update(datetime(2023, 1, 10), 105.0) # OTM -> Max Profit
    assert not sim.is_open
    assert sim.exit_reason == "EXPIRY"
    assert sim.pnl == 1.0 # Kept full credit (Closing cost 0)

def test_sim_max_loss_at_expiry():
    cand = SpreadCandidate(
        underlying_ticker="AAPL",
        expiry_date="2023-01-10",
        dte=10,
        short_strike=100,
        long_strike=95,
        net_credit=1.0,
        width=5.0
    )
    policy = SimPolicy()
    sim = PositionSimulator(cand, policy, datetime(2023, 1, 1))

    # Expiry at 90 (Below long strike) -> Max Loss
    sim.update(datetime(2023, 1, 10), 90.0)
    assert not sim.is_open

    # Closing Cost = (100 - 90) - (95 - 90) = 10 - 5 = 5.0
    # PnL = 1.0 - 5.0 = -4.0
    assert sim.pnl == -4.0
