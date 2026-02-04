import polars as pl
import pandas as pd
from typing import List, Dict, Optional, Any
from backend.app.models.student_runner import StudentRunner
from backend.app.risk.risk_manager import RiskManager
from backend.app.risk.types import RiskDecision, ActionType
from backend.app.portfolio.state import PortfolioState
from backend.app.engine.swing_scheduler import SwingScheduler, TradingWindow

class EquityPod:
    def __init__(self, ticker: str, model_path: str, preproc_path: str, device: str = "cpu"):
        self.ticker = ticker
        self.runner = StudentRunner(model_path, preproc_path, device=device)
        self.risk_manager = RiskManager()
        self.portfolio = PortfolioState()
        self.scheduler = SwingScheduler()
        
        # Buffer for lookback
        # Schema must match MarketFrame
        self.buffer = pl.DataFrame() 
        self.lookback = 128 # Should get from model metadata

    def on_tick(self, tick: Dict[str, Any], breadth: Dict[str, float]) -> Optional[RiskDecision]:
        """
        tick: {'timestamp': ..., 'open': ..., 'close': ..., 'volume': ...}
        """
        # 1. Update Portfolio
        ts = tick['timestamp']
        price = tick['close']
        self.portfolio.update_price(self.ticker, price)
        
        # 2. Update Buffer
        # Convert tick to DF row
        # We need all columns required by preproc
        # timestamp, open, high, low, close, volume, ad_line, bpi (joined from breadth)
        
        row_dict = tick.copy()
        row_dict['ad_line'] = breadth.get('ad_line', 0.0)
        row_dict['bpi'] = breadth.get('bpi', 50.0)
        
        # Ensure types for Polars
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
            
        # Maintain buffer size (lookback + margin)
        if self.buffer.height > self.lookback + 10:
            self.buffer = self.buffer.tail(self.lookback + 10)
            
        # 3. Check Schedule
        window = self.scheduler.get_window(pd.Timestamp(ts))
        allowed_actions = self.scheduler.get_allowed_actions(window)
        
        if not allowed_actions:
            return None
            
        # 4. Inference
        # Need enough data
        if self.buffer.height < self.lookback:
            return None
            
        # Get last 'lookback' rows
        window_df = self.buffer.tail(self.lookback)
        
        # Run model
        # Try/Except for runtime safety
        try:
            signal = self.runner.infer(window_df)
        except Exception as e:
            print(f"Inference failed: {e}")
            return None
            
        # 5. Risk Management
        decision = self.risk_manager.evaluate(self.ticker, signal, self.portfolio, breadth)
        
        # Filter action by allowed_actions
        if decision.action.value not in allowed_actions:
            # Downgrade to HOLD or NO_NEW_RISK
            if decision.action == ActionType.BUY:
                decision.action = ActionType.NO_NEW_RISK
                decision.reason += " | Scheduled Lockout"
            elif decision.action == ActionType.SELL:
                 # If Sell not allowed? (e.g. Pre-open).
                 decision.action = ActionType.HOLD
                 
        return decision
        
    def execute_decision(self, decision: RiskDecision):
        # Update portfolio state (Paper execution)
        if decision.action == ActionType.BUY and decision.quantity > 0:
            price = self.portfolio.positions.get(self.ticker, self.portfolio.positions.get("dummy", None)) # Need price
            # We don't have current price in decision.
            # Use buffer last close.
            price = self.buffer["close"][-1]
            self.portfolio.add_fill(self.ticker, decision.quantity, price)
            print(f"EXECUTED BUY {self.ticker} {decision.quantity} @ {price}")
            
        elif decision.action in (ActionType.SELL, ActionType.LIQUIDATE, ActionType.REDUCE) and decision.quantity < 0:
            price = self.buffer["close"][-1]
            # quantity is negative
            self.portfolio.add_fill(self.ticker, decision.quantity, price)
            print(f"EXECUTED SELL {self.ticker} {decision.quantity} @ {price}")
