import polars as pl
from typing import List, Dict, Optional, Tuple, Any
from datetime import date, datetime

from backend.app.core.config import REBALANCE_CALENDAR, MIN_HOLD_DAYS, MAX_HOLD_DAYS
from backend.app.risk.types import RiskDecision, ActionType

class PortfolioBuilder:
    """
    Constructs target portfolio from Leaderboard signals.
    Implements:
    - Top-K Selection
    - Turnover Constraints (Min Hold Days)
    - Rebalance Calendar
    - Replacement Thresholds
    """
    def __init__(self, target_size: int = 25, max_weight: float = 0.05):
        self.target_size = target_size
        self.max_weight = max_weight
        self.calendar = [d.upper() for d in REBALANCE_CALENDAR] # ["MON", "WED", "FRI"]
        self.min_hold_days = MIN_HOLD_DAYS
        self.max_hold_days = MAX_HOLD_DAYS

    def is_rebalance_day(self, current_date: date) -> bool:
        """
        Check if today is a scheduled rebalance day.
        """
        day_str = current_date.strftime("%a").upper()
        return day_str in self.calendar

    def construct_portfolio(self,
                            current_date: date,
                            leaderboard: pl.DataFrame,
                            current_holdings: Dict[str, Dict] # {ticker: {entry_date: ..., shares: ...}}
                           ) -> List[RiskDecision]:
        """
        Returns a list of BUY/SELL/REBALANCE decisions.
        """
        if not self.is_rebalance_day(current_date):
            # Only process mandatory exits (stop loss, max hold)?
            # Here we focus on rebalancing.
            # RiskManager handles stops separately in real-time loop.
            # But we might want to check max hold expiry.
            decisions = []
            for ticker, info in current_holdings.items():
                entry_date = info.get("entry_date")
                if entry_date:
                    # Parse date if string
                    if isinstance(entry_date, str):
                        entry_date = datetime.strptime(entry_date, "%Y-%m-%d").date()

                    held_days = (current_date - entry_date).days
                    if held_days >= self.max_hold_days:
                        # Force Liquidate
                        decisions.append(RiskDecision(
                            ticker=ticker,
                            action=ActionType.LIQUIDATE,
                            quantity=0, # All
                            reason="Max Hold Expiry",
                            max_position_size=0.0
                        ))
            return decisions

        # 1. Identify Candidates
        # Filter: exclude tickers with negative EV? Or just rank?
        # Leaderboard sorted by score desc.

        # Take Top K
        candidates = leaderboard.head(self.target_size)
        candidate_tickers = set(candidates["ticker"].to_list())

        # 2. Manage Existing Holdings
        decisions = []
        current_tickers = set(current_holdings.keys())

        # A. Sells (Held but not in Top K anymore)
        to_sell = []
        for ticker in current_tickers:
            if ticker not in candidate_tickers:
                # Check Min Hold
                entry_date = current_holdings[ticker].get("entry_date")
                if entry_date:
                    if isinstance(entry_date, str):
                        entry_date = datetime.strptime(entry_date, "%Y-%m-%d").date()
                    held_days = (current_date - entry_date).days
                    if held_days < self.min_hold_days:
                        # Keep it (Locked)
                        continue

                # Sell
                decisions.append(RiskDecision(
                    ticker=ticker,
                    action=ActionType.LIQUIDATE,
                    quantity=0,
                    reason=f"Dropped from Top {self.target_size}",
                    max_position_size=0.0
                ))
                to_sell.append(ticker)

        # Update current set after planned sells
        kept_tickers = current_tickers - set(to_sell)

        # B. Buys (In Top K but not held)
        # Capacity check
        slots_available = self.target_size - len(kept_tickers)

        if slots_available > 0:
            # Pick best available candidates not already held
            new_buys = []
            for row in candidates.iter_rows(named=True):
                t = row["ticker"]
                if t not in current_tickers:
                    new_buys.append(row)

            # Take top N
            for row in new_buys[:slots_available]:
                # Calculate size?
                # Simple equal weight 1/target_size
                # Or use EV?
                # For now, RiskDecision usually handles sizing via RiskManager?
                # Or we output Target Weight.
                # RiskDecision has quantity. We might need price to calc quantity.
                # Let's assume we send BUY signal with confidence/score, and EquityPod calculates quantity.

                decisions.append(RiskDecision(
                    ticker=row["ticker"],
                    action=ActionType.BUY,
                    quantity=1, # Placeholder, sizing downstream
                    reason=f"Top {self.target_size} Entry (Rank {row['rank']})",
                    max_position_size=self.max_weight # Example cap
                ))

        # C. Rebalance Weights of Kept?
        # Maybe later.

        return decisions
