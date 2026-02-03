import os
import sys
from datetime import datetime
# Ensure backend in path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

from backend.app.options.sim.credit_spread_sim import PositionSimulator
from backend.app.options.sim.policy import SimPolicy
from backend.app.options.types import SpreadCandidate

def main():
    print("Running Options Backtest...")

    # Mock candidate
    cand = SpreadCandidate(
        underlying_ticker="AAPL",
        expiry_date="2023-02-17",
        dte=30,
        short_strike=140,
        long_strike=135,
        net_credit=1.0,
        width=5.0,
        max_loss=400.0,
        strategy_type="put_credit_spread"
    )

    policy = SimPolicy()

    # Simulate
    sim = PositionSimulator(cand, policy, datetime(2023, 1, 17))

    # Mock price path
    prices = [150, 148, 145, 142, 138, 145, 155] # Dip then rally

    for i, p in enumerate(prices):
        sim.update(datetime(2023, 1, 17 + i), p)
        if not sim.is_open:
            print(f"Position closed on day {i}: PnL {sim.pnl}")
            break

    if sim.is_open:
        print("Position still open at end of path")

if __name__ == "__main__":
    main()
