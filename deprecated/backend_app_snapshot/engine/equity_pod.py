import logging
import polars as pl
import pandas as pd
from typing import List, Dict, Optional, Any, Union
from datetime import date, datetime
import os

from backend.app.models.student_inference import StudentRunner
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
        if self.execution_mode == "LEGACY":
            logging.warning(f"EquityPod [{ticker}] running in LEGACY mode (DEPRECATED).")

    def on_tick(self, tick: Dict[str, Any], breadth: Dict[str, float], instruction: Optional[RiskDecision] = None) -> Optional[RiskDecision]:
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
            # Log placeholder
            if instruction:
                self._log_shadow_ranking(ts.date(), instruction)
            
        elif self.execution_mode == "RANKING":
            # Apply Portfolio Instruction
            if instruction:
                # Check Schedule
                if instruction.action.value in allowed_actions:
                    decision = instruction
                else:
                    # Gated by schedule
                    # e.g. Trying to BUY in LATE_ADJUST?
                    pass
        
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

    def _log_shadow_ranking(self, date: date, instruction: RiskDecision):
        print(f"[SHADOW] {date} Instruction for {self.ticker}: {instruction}")

        
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
