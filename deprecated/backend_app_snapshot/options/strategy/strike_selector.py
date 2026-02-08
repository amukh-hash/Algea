from typing import Optional, List
from backend.app.options.types import SpreadCandidate
from backend.app.options.data.types import OptionChainSnapshot
from backend.app.models.types import DistributionForecast
from backend.app.options.strategy.spread_rules import validate_spread
from backend.app.options.strategy.dte_policy import get_target_dte_range

class StrikeSelector:
    def __init__(self):
        pass
        
    def select_best_spread(
        self, 
        underlying_price: float, 
        chain: OptionChainSnapshot, 
        forecast: DistributionForecast,
        min_credit: float = 0.10
    ) -> Optional[SpreadCandidate]:
        
        # 1. Filter Expiry (Chain should be for target expiry already)
        
        # 2. Filter Strikes (OTM Puts)
        # Bull Put Spread: Both strikes below current price (usually).
        # Short Strike < Price.
        
        puts = [r for r in chain.rows if r.option_type == "put"]
        puts.sort(key=lambda x: x.strike)
        
        candidates = []
        
        # Simple scan: Try all pairs
        for i in range(len(puts)):
            long_leg = puts[i]
            for j in range(i + 1, len(puts)):
                short_leg = puts[j]
                
                # Check strikes are OTM
                # if short_leg.strike >= underlying_price: continue 
                # (Can sell ITM but let's stick to OTM for safety/probability)
                if short_leg.strike >= underlying_price * 0.99: # 1% OTM buffer
                     continue
                     
                width = short_leg.strike - long_leg.strike
                if width < 0.5: continue
                
                # Credit = Short Bid - Long Ask (conservative)
                credit = short_leg.bid - long_leg.ask
                if credit < min_credit: continue
                
                max_loss = width - credit
                if max_loss <= 0: continue # Arb?
                
                # Create Candidate
                cand = SpreadCandidate(
                    underlying_ticker=chain.ticker,
                    expiry_date=chain.expiry,
                    dte=chain.dte,
                    strategy_type="put_credit_spread",
                    short_strike=short_leg.strike,
                    long_strike=long_leg.strike,
                    short_price=short_leg.bid,
                    long_price=long_leg.ask,
                    net_credit=round(credit, 2),
                    width=width,
                    max_loss=round(max_loss * 100, 2), # Per contract
                    max_profit=round(credit * 100, 2),
                    risk_reward_ratio=max_loss/credit if credit > 0 else 999
                )
                
                if validate_spread(cand):
                    candidates.append(cand)
        
        # 3. Rank Candidates
        # Prefer higher credit / width? Or Probability?
        # For now: Maximize credit while keeping risk controlled.
        candidates.sort(key=lambda x: x.net_credit, reverse=True)
        
        if not candidates:
            return None
            
        return candidates[0]
