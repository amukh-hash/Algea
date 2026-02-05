import pytest
from unittest.mock import patch, MagicMock
from datetime import date, datetime
import polars as pl
from backend.app.engine.portfolio_manager import PortfolioManager
from backend.app.portfolio.state import PortfolioState, Position
from backend.app.risk.types import ActionType

@pytest.fixture
def mock_leaderboard():
    return pl.DataFrame({
        "ticker": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META"],
        "rank": [1, 2, 3, 4, 5, 6],
        "score": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
        "rank_pct": [0.99, 0.95, 0.9, 0.85, 0.8, 0.75]
    })

def test_generate_instructions_buy_top_k(mock_leaderboard):
    # Setup
    pm = PortfolioManager()
    state = PortfolioState(cash=100000) # Empty portfolio
    current_date = date(2024, 1, 3) # Wednesday (Rebalance likely allowed)
    
    with patch("backend.app.core.artifacts.resolve_leaderboard_path", return_value="dummy_path"), \
         patch("polars.read_parquet", return_value=mock_leaderboard), \
         patch.object(pm.builder, 'is_rebalance_day', return_value=True):
        
        instructions = pm.generate_instructions(current_date, state)
        
        # Should execute buys for Top 5 (default target size 25, but inputs are few)
        # Check AAPL
        assert "AAPL" in instructions
        assert instructions["AAPL"].action == ActionType.BUY
        assert "Top 25 Entry" in instructions["AAPL"].reason
        assert "Rank 1" in instructions["AAPL"].reason
        assert instructions["AAPL"].target_weight is not None

def test_generate_instructions_sell_dropout(mock_leaderboard):
    # Setup: Holding a stock (XYZ) that is NOT in leaderboard
    pm = PortfolioManager()
    state = PortfolioState(cash=100000)
    # Add held position
    state.positions["XYZ"] = Position("XYZ", 100, 10.0, 10.0, entry_date=date(2023, 12, 1))
    
    current_date = date(2024, 1, 3) # Wed
    
    with patch("backend.app.core.artifacts.resolve_leaderboard_path", return_value="dummy_path"), \
         patch("polars.read_parquet", return_value=mock_leaderboard), \
         patch.object(pm.builder, 'is_rebalance_day', return_value=True):
         
        instructions = pm.generate_instructions(current_date, state)
        
        # XYZ should be unsold
        assert "XYZ" in instructions
        assert instructions["XYZ"].action == ActionType.LIQUIDATE
        assert "Dropped from Top" in instructions["XYZ"].reason

def test_generate_instructions_hold_top_k(mock_leaderboard):
    # Setup: Holding AAPL (Rank 1)
    pm = PortfolioManager()
    state = PortfolioState(cash=100000)
    state.positions["AAPL"] = Position("AAPL", 50, 150.0, 150.0, entry_date=date(2023, 12, 1))
    
    current_date = date(2024, 1, 3)
    
    with patch("backend.app.core.artifacts.resolve_leaderboard_path", return_value="dummy_path"), \
         patch("polars.read_parquet", return_value=mock_leaderboard), \
         patch.object(pm.builder, 'is_rebalance_day', return_value=True):
         
        instructions = pm.generate_instructions(current_date, state)
        
        # Now PortfolioBuilder explicitly emits HOLDs for kept positions with weights
        assert "AAPL" in instructions
        assert instructions["AAPL"].action == ActionType.HOLD
        assert instructions["AAPL"].target_weight is not None
