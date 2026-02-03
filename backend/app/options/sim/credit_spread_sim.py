from typing import Optional, Dict
from backend.app.options.types import SpreadCandidate, OptionsPosition
from backend.app.options.sim.policy import SimPolicy
from backend.app.options.sim.payoff import calculate_spread_payoff
from datetime import datetime

class PositionSimulator:
    def __init__(self, candidate: SpreadCandidate, policy: SimPolicy, entry_time: datetime):
        self.candidate = candidate
        self.policy = policy
        self.entry_time = entry_time

        # State
        self.is_open = True
        self.exit_time: Optional[datetime] = None
        self.exit_reason: str = ""
        self.pnl: float = 0.0
        self.max_unrealized_loss: float = 0.0
        self.entry_credit = candidate.net_credit

    def update(self, current_time: datetime, underlying_price: float, iv_proxy: float = None):
        if not self.is_open:
            return

        # 1. Mark to Market
        # Estimate current spread value.
        # Without full pricing engine (BS), we can approximate or use Payoff if at expiry.

        dte = (datetime.strptime(self.candidate.expiry_date, "%Y-%m-%d") - current_time).days

        if dte <= self.policy.close_at_dte:
            # Close at Expiry/DTE limit
            closing_cost = calculate_spread_payoff(
                underlying_price,
                self.candidate.short_strike,
                self.candidate.long_strike,
                self.candidate.strategy_type
            )
            self.close_position(current_time, closing_cost, "EXPIRY")
            return

        # Estimate spread price roughly
        # This is the hardest part without a chain snapshot.
        # Simple Model: Linear decay of time value + Intrinsic.
        # Or just assume price moves.

        # For Phase 2A mock/sim:
        # We can implement a very rough "delta-based" price change.
        # Price ~ OriginalCredit * (TimeFactor) + DeltaEffect

        # Or better: don't simulate MTM perfectly, just check Stop Loss based on Underlying?
        # A common heuristic: If underlying breaches Short Strike, we are in trouble.

        if underlying_price < self.candidate.short_strike:
            # ITM (for put spread)
            # Estimate Loss.
            # Max loss is width - credit.
            # Let's say we hit stop if price < short_strike - X%
            pass

        # TP Check
        # If we held for 50% of time and price is far OTM?

    def close_position(self, timestamp: datetime, closing_cost: float, reason: str):
        self.is_open = False
        self.exit_time = timestamp
        self.exit_reason = reason
        # PnL = Credit Received - Cost to Close
        self.pnl = self.entry_credit - closing_cost
