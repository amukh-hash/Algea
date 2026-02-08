import polars as pl
from typing import Dict, Optional, List
from datetime import date
from backend.app.core import artifacts, config
from backend.app.risk.types import RiskDecision
from backend.app.portfolio.state import PortfolioState
from backend.app.engine.portfolio_construction import PortfolioBuilder

class PortfolioManager:
    """
    Central orchestration for portfolio ranking and decision making.
    Loads daily leaderboards and generates target instructions using PortfolioBuilder.
    """
    def __init__(self):
        self.builder = PortfolioBuilder()

    def generate_instructions(self, 
                            current_date: date, 
                            state: PortfolioState,
                            recent_returns: Optional[pl.DataFrame] = None) -> Dict[str, RiskDecision]:
        """
        Generates trading instructions for any ticker requiring action.
        Returns map {ticker: RiskDecision}.
        """
        # 1. Load Leaderboard
        lb_path = artifacts.resolve_leaderboard_path(str(current_date), "v1")
        if not lb_path:
            # Fallback: Check previous day? 
            # For strict implementation, we expect signal for the trading date.
            return {} 

        try:
            leaderboard = pl.read_parquet(lb_path)
        except Exception:
            return {}

        # 2. Extract Holdings for Builder
        # Builder expects Dict[str, Dict] with entry_date
        current_holdings = {}
        for ticker, pos in state.positions.items():
            current_holdings[ticker] = {
                "entry_date": pos.entry_date,
                "shares": pos.quantity,
                "avg_price": pos.avg_price
            }

        # 3. Generate Decisions via Builder
        # This handles Top-K, Min Hold Days, Rebalance Calendar
        decisions_list = self.builder.construct_portfolio(
            current_date, 
            leaderboard, 
            current_holdings,
            recent_returns=recent_returns
        )
        
        # 4. Convert to Instruction Map
        instructions = {d.ticker: d for d in decisions_list}
        return instructions
