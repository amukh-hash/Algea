import polars as pl
import pandas as pd
from typing import List, Dict, Optional, Any, Union
from datetime import date, datetime
import os

from backend.app.models.student_runner import StudentRunner
from backend.app.risk.risk_manager import RiskManager
from backend.app.risk.types import RiskDecision, ActionType
from backend.app.portfolio.state import PortfolioState
from backend.app.engine.swing_scheduler import SwingScheduler, TradingWindow
from backend.app.core.config import EXECUTION_MODE
from backend.app.core import artifacts
from backend.app.engine.portfolio_construction import PortfolioBuilder

class EquityPod:
    def __init__(self, ticker: str, model_path: str, preproc_path: str, device: str = "cpu"):
        # Legacy
        self.ticker = ticker # Still needed for single ticker context
        self.runner = StudentRunner(model_path, preproc_path, device=device)
        self.risk_manager = RiskManager()
        self.portfolio = PortfolioState()
        self.scheduler = SwingScheduler()
        
        # Buffer for lookback
        self.buffer = pl.DataFrame() 
        self.lookback = 128

        # New Execution Mode
        self.execution_mode = EXECUTION_MODE
        self.portfolio_builder = PortfolioBuilder()

    def on_tick(self, tick: Dict[str, Any], breadth: Dict[str, float]) -> Optional[RiskDecision]:
        """
        tick: {'timestamp': ..., 'open': ..., 'close': ..., 'volume': ...}
        """
        # 1. Update Portfolio & Buffer (Common)
        ts = tick['timestamp']
        price = tick['close']
        self.portfolio.update_price(self.ticker, price)
        
        row_dict = tick.copy()
        row_dict['ad_line'] = breadth.get('ad_line', 0.0)
        row_dict['bpi'] = breadth.get('bpi', 50.0)
        
        row = pl.DataFrame([row_dict]).select([
            pl.col("timestamp"),
            pl.col("open").cast(pl.Float64),
            pl.col("high").cast(pl.Float64),
            pl.col("low").cast(pl.Float64),
            pl.col("close").cast(pl.Float64),
            pl.col("volume").cast(pl.Float64),
            pl.col("ad_line").cast(pl.Float64),
            pl.col("bpi").cast(pl.Float64)
        ])
        
        if self.buffer.height == 0:
            self.buffer = row
        else:
            self.buffer = pl.concat([self.buffer, row])
            
        if self.buffer.height > self.lookback + 10:
            self.buffer = self.buffer.tail(self.lookback + 10)
            
        # 3. Check Schedule
        if isinstance(ts, str):
             ts = pd.Timestamp(ts)
        window = self.scheduler.get_window(ts)
        allowed_actions = self.scheduler.get_allowed_actions(window)
        
        if not allowed_actions:
            return None

        # --- EXECUTION LOGIC ---

        decision = None

        if self.execution_mode == "LEGACY":
            decision = self._run_legacy_inference(allowed_actions, breadth)

        elif self.execution_mode == "SHADOW":
            # Run Legacy for actual execution
            decision = self._run_legacy_inference(allowed_actions, breadth)

            # Run Ranking for logging (Shadow)
            # We need to know if we are in rebalance window?
            # Ranking signals are daily.
            # We assume ranking runs once per day/session.
            self._log_shadow_ranking(ts.date(), self.portfolio)
            
        elif self.execution_mode == "RANKING":
            # Run Ranking based execution
            decision = self._run_ranking_execution(ts.date(), self.portfolio, allowed_actions)

        return decision

    def _run_legacy_inference(self, allowed_actions: List[int], breadth: Dict[str, float]) -> Optional[RiskDecision]:
        if self.buffer.height < self.lookback:
            return None
            
        window_df = self.buffer.tail(self.lookback)
        try:
            signal = self.runner.infer(window_df)
        except Exception as e:
            print(f"Inference failed: {e}")
            return None
            
        decision = self.risk_manager.evaluate(self.ticker, signal, self.portfolio, breadth)
        
        if decision.action.value not in allowed_actions:
            if decision.action == ActionType.BUY:
                decision.action = ActionType.NO_NEW_RISK
                decision.reason += " | Scheduled Lockout"
            elif decision.action == ActionType.SELL:
                 decision.action = ActionType.HOLD
                 
        return decision

    def _run_ranking_execution(self, current_date: date, portfolio: PortfolioState, allowed_actions: List[int]) -> Optional[RiskDecision]:
        """
        Uses Daily Leaderboard to make decisions.
        This function is called per ticker, but the logic depends on portfolio context.
        Ideally EquityPod should just lookup its specific instruction from a portfolio-level plan?

        BUT: Current architecture is EquityPod per ticker.
        So we load the leaderboard, filter for THIS ticker, and check if it's a buy/sell candidate?

        Better: We compute the portfolio plan ONCE (centralized) and distribute instructions?
        Or: We compute it here, but efficiently.

        Let's load the leaderboard for 'current_date' (or last close).
        """
        # Resolve Leaderboard
        # Check cache?
        # Path: backend/data/signals/selector/v1/{date}.parquet
        # We need the leaderboard generated for decision making TODAY.
        # This usually means produced YESTERDAY close (or today morning).
        # As_of_date = current_date (if morning) or current_date - 1 (if trading day not closed).

        # Assumption: Leaderboard is keyed by 'as_of_date'.
        # If we trade on T, we use signals from T-1? Or T (pre-market)?
        # Let's assume we look for TODAY's signal file (generated pre-open).

        lb_path = artifacts.resolve_leaderboard_path(str(current_date), "v1")
        if not lb_path:
             # Try yesterday?
             # For now, strict.
             return None

        try:
            leaderboard = pl.read_parquet(lb_path)
        except Exception:
            return None

        # Get decisions from PortfolioBuilder
        # We need current holdings for ALL tickers?
        # EquityPod only knows its own ticker portfolio state?
        # NO: PortfolioState might be singleton or we need to pass full state.
        # Actually PortfolioState is usually per-pod in this design?
        # If per-pod, we can't do portfolio construction properly distributed.

        # CRITICAL ADAPTATION:
        # In this refactor, EquityPod acts as the executor for ONE ticker.
        # But it needs to know if it should be in the portfolio.
        # We can implement the logic:
        # "Am I in the top K?"

        # Current holdings map?
        # We assume we only know if WE hold it.
        # But to know if we should sell to make room, we need to know other candidates?
        # The PortfolioBuilder needs GLOBAL state.

        # Compromise for "Per-Ticker Agent":
        # We run the builder logic assuming we know our own status.
        # But we can't know "slots available" without global state.

        # FIX: We should have a central "PortfolioManager" that runs PortfolioBuilder.
        # But since we modify EquityPod:
        # We can just check: "Is this ticker in Top K?"
        # If Yes: Buy/Hold.
        # If No: Sell/Liquidate.

        target_size = self.portfolio_builder.target_size

        # My rank
        my_row = leaderboard.filter(pl.col("ticker") == self.ticker)
        if len(my_row) == 0:
            # Not in universe or no score
            # Liquidate if held
            if self.portfolio.current_position > 0:
                 return RiskDecision(ActionType.LIQUIDATE, 0, "Not in Universe", 1.0)
            return None

        rank = my_row["rank"][0]
        rank_pct = my_row["rank_pct"][0]
        score = my_row["score"][0]

        # Decision Logic
        is_held = self.portfolio.current_position > 0
        in_top_k = rank <= target_size

        if is_held:
            if not in_top_k:
                # Sell?
                # Check Min Hold
                # We need entry date from portfolio
                # Assuming portfolio tracks entry date per position
                # If not, assume liquidatable.
                 return RiskDecision(ActionType.LIQUIDATE, 0, f"Rank {rank} > {target_size}", 1.0)
            else:
                return RiskDecision(ActionType.HOLD, 0, f"Rank {rank} (Top {target_size})", rank_pct)
        else:
            if in_top_k:
                # Buy
                # Only if allowed (rebalance day?)
                if self.portfolio_builder.is_rebalance_day(current_date):
                     return RiskDecision(ActionType.BUY, 1, f"Rank {rank} Entry", rank_pct)

        return None

    def _log_shadow_ranking(self, date: date, portfolio: PortfolioState):
        # Log what we WOULD have done
        # Just print for now
        # logger.info(...)
        pass
        
    def execute_decision(self, decision: RiskDecision):
        # Update portfolio state (Paper execution)
        if decision.action == ActionType.BUY and decision.quantity > 0:
            price = self.buffer["close"][-1]
            self.portfolio.add_fill(self.ticker, decision.quantity, price)
            print(f"EXECUTED BUY {self.ticker} {decision.quantity} @ {price}")
            
        elif decision.action in (ActionType.SELL, ActionType.LIQUIDATE, ActionType.REDUCE) and decision.quantity <= 0:
            price = self.buffer["close"][-1]
            self.portfolio.add_fill(self.ticker, decision.quantity, price) # quantity is negative/zero logic handled
            print(f"EXECUTED SELL {self.ticker} {decision.quantity} @ {price}")
