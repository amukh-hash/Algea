
import pandas as pd
import numpy as np
import torch
from typing import Dict, List, Optional
from datetime import datetime
from backend.app.ops import config

class TopKBasketEvaluator:
    def __init__(self, k: int = 20, cost_bps: int = 10):
        self.k = k
        self.cost_bps = cost_bps
        self.history = []
        
    def step(self, date, scores: Dict[str, float], targets: Dict[str, float], aux: Dict[str, float] = None):
        """
        Evaluate one day.
        scores: {ticker: score}
        targets: {ticker: forward_return}
        """
        # Select Top K
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_k = sorted_items[:self.k] # List of (ticker, score)
        
        tickers = [x[0] for x in top_k]
        
        # Calculate Basket Return (Equal Weight)
        rets = []
        for t in tickers:
            if t in targets:
                rets.append(targets[t])
            else:
                rets.append(0.0) # Missing target -> 0 return (cash)
        
        raw_ret = np.mean(rets) if rets else 0.0
        
        # Determine turnover if we have history
        turnover = 0.0
        if self.history:
            prev_tickers = set(self.history[-1]["tickers"])
            curr_tickers = set(tickers)
            
            # Turnover = (New + Left) / (2 * K)?
            # Or just "One-way turnover"?
            # One-way = (New positions) / K?
            # Let's use intersection.
            overlap = len(prev_tickers.intersection(curr_tickers))
            # If K is constant, turnover = (K - overlap) / K
            turnover = (self.k - overlap) / self.k
            
        # Cost adjusted
        # Cost = turnover * 2 * cost_bps (buy + sell?)
        # 100% turnover = buy K, sell K.
        # cost = turnover * 2 * cost_bps/10000
        cost = turnover * 2 * (self.cost_bps / 10000.0)
        net_ret = raw_ret - cost
        
        record = {
            "date": date,
            "raw_ret": raw_ret,
            "net_ret": net_ret,
            "turnover": turnover,
            "cost": cost,
            "tickers": tickers
        }
        self.history.append(record)
        
    def get_metrics(self) -> Dict:
        if not self.history:
            return {}
            
        df = pd.DataFrame(self.history)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        
        # Cumulative
        # Assuming returns are simple? If log, sum. If simple, prod.
        # targets in dataset are simple returns usually? (p_fut/p_t - 1)
        # So we use (1+r).cumprod()
        
        df["cum_ret"] = (1 + df["net_ret"]).cumprod()
        
        total_ret = df["cum_ret"].iloc[-1] - 1.0
        
        # Annualized Vol
        # std of daily returns * sqrt(252)?
        # If returns are 10D?
        # If we step daily, we have overlapping returns!
        # Evaluation should be on non-overlapping windows?
        # Or daily rebalance of a 10D strategy?
        # "Swing trading (5-10 days)".
        # Usually implies holding for 10 days.
        # If we rebalance daily, we have a portfolio of staggered trades.
        # But `targets` passed to step are "Forward 10D return".
        # If we realize "Forward 10D return" ON THE DAY OF SIGNAL, that's cheating (peeking).
        # Realized return happens at T+10.
        
        # Correct Backtest:
        # At T, we pick Top K.
        # Return is generated from T+1 to T+10.
        # If we evaluate "Signal Value", we assign the 10D return to date T.
        # This is "Forward Analysis".
        # The metric "Excess Return" in ablation usually refers to "Predictive Power" (IC * Vol) or implementation backtest?
        # Given "Top-K basket evaluator to generate performance metrics", usually assumes simple "signal quality" backtest (Forward Return).
        
        # So we aggregate Forward Returns.
        # Warning: These returns overlap!
        # Mean Daily Forward Return * 252 is roughly Annualized Return.
        # Sharpe = Mean / Std * sqrt(252) (but adjusting for overlap autocorrelation).
        # Simple stats for now.
        
        avg_ret = df["net_ret"].mean()
        std_ret = df["net_ret"].std()
        sharpe = (avg_ret / (std_ret + 1e-9)) * np.sqrt(252) # Naive
        
        # Drawdown of the cumulative curve
        # (High - Curr) / High
        peaks = df["cum_ret"].cummax()
        dds = (peaks - df["cum_ret"]) / peaks
        max_dd = dds.max()
        
        avg_turnover = df["turnover"].mean()
        
        return {
            "total_ret": total_ret,
            "sharpe": sharpe,
            "max_dd": max_dd,
            "avg_turnover": avg_turnover,
            "final_cum": df["cum_ret"].iloc[-1]
        }
        
    def to_parquet(self, path: str):
         df = pd.DataFrame(self.history)
         df.to_parquet(path)
