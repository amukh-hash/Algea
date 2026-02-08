import pytest
from backend.app.options.types import SpreadCandidate

def test_spread_candidate_contract():
    cand = SpreadCandidate(
        underlying_ticker="AAPL",
        expiry_date="2023-01-01",
        dte=30,
        short_strike=95.0,
        long_strike=90.0,
        width=5.0,
        net_credit=1.0,
        max_loss=400.0,
        risk_reward_ratio=400.0/1.0
    )
    
    assert cand.short_strike > cand.long_strike # Put Credit
    assert cand.width == (cand.short_strike - cand.long_strike)
    assert cand.risk_reward_ratio > 0
    assert cand.strategy_type == "put_credit_spread"
