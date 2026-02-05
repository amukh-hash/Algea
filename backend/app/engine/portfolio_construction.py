import polars as pl
from typing import List, Dict, Optional, Tuple, Any
from datetime import date, datetime

from backend.app.core.config import REBALANCE_CALENDAR, MIN_HOLD_DAYS, MAX_HOLD_DAYS
from backend.app.risk.types import RiskDecision, ActionType
from backend.app.portfolio.hrp import compute_hrp_weights
import pandas as pd

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
                            current_holdings: Dict[str, Dict], # {ticker: {entry_date: ..., shares: ...}}
                            recent_returns: Optional[pl.DataFrame] = None
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

        # Identify New Buys
        slots_available = self.target_size - len(kept_tickers)
        new_buys = []
        if slots_available > 0:
            for row in candidates.iter_rows(named=True):
                t = row["ticker"]
                if t not in current_tickers:
                    new_buys.append(row)
            new_buys = new_buys[:slots_available]

        # 3. Calculate Weights for Active Tickers
        active_tickers = list(kept_tickers.union(set(row["ticker"] for row in new_buys)))
        weights = {}
        
        if active_tickers:
            # Try HRP
            if recent_returns is not None and not recent_returns.is_empty():
                 # Filter returns for active tickers
                 # Assuming recent_returns is Polars DataFrame: date, tickers...
                 # Or Ticker, Date, Return?
                 # HRP expects Pandas DataFrame (T x N).
                 # Let's assume recent_returns is "wide" (date index, ticker columns) or we pivot.
                 # Implementation Detail: If Polars, convert to pandas.
                 try:
                     # Pivot if long format
                     if "ticker" in recent_returns.columns and "return" in recent_returns.columns:
                         # Pivot: index=date, columns=ticker, values=return
                         pdf = recent_returns.to_pandas().pivot(index="date", columns="ticker", values="return")
                     else:
                         # Assume wide
                         pdf = recent_returns.to_pandas()
                     
                     # Filter columns
                     cols = [t for t in active_tickers if t in pdf.columns]
                     if len(cols) > 1:
                         pdf_slice = pdf[cols]
                         weights = compute_hrp_weights(pdf_slice)
                     else:
                         # Fallback for 1 asset or no data
                         weights = {t: 1.0/len(active_tickers) for t in active_tickers}
                 except Exception as e:
                     print(f"HRP Failed: {e}. Fallback to Equal Weight.")
                     weights = {t: 1.0/len(active_tickers) for t in active_tickers}
            else:
                # Equal Weight Fallback
                val = 1.0 / len(active_tickers)
                weights = {t: val for t in active_tickers}

        # 4. Emit Decisions for New Buys
        slots_available = self.target_size - len(kept_tickers)
        if slots_available > 0:
            for row in new_buys[:slots_available]:
                t = row["ticker"]
                w = weights.get(t, 0.0)
                decisions.append(RiskDecision(
                    ticker=t,
                    action=ActionType.BUY,
                    quantity=1, # Sizing downstream uses target_weight
                    reason=f"Top {self.target_size} Entry (Rank {row['rank']})",
                    max_position_size=self.max_weight,
                    target_weight=w
                ))
        
        # 5. Emit HOLD/REBALANCE for Kept Tickers
        for t in kept_tickers:
            w = weights.get(t, 0.0)
            decisions.append(RiskDecision(
                ticker=t,
                action=ActionType.HOLD,
                quantity=0,
                reason="Maintained in Portfolio",
                target_weight=w
            ))

        return decisions
