import pytest
import pandas as pd
import numpy as np
from backend.app.portfolio import hrp
from backend.app.risk import risk_manager, types, posture
from backend.app.models.signal_types import ModelSignal
from backend.app.portfolio.state import PortfolioState

def test_hrp_weights():
    # correlated assets
    # A, B highly correlated. C uncorrelated.
    # HRP should allocate less to A/B cluster, more to C?
    # Actually HRP allocates based on inverse variance of clusters.
    
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=100)
    
    # C is stable
    c = np.random.normal(0.001, 0.01, 100)
    
    # A is volatile
    a = np.random.normal(0.001, 0.02, 100)
    
    # B is A + noise (highly correlated)
    b = a + np.random.normal(0, 0.001, 100)
    
    returns = pd.DataFrame({"A": a, "B": b, "C": c}, index=dates)
    
    weights = hrp.compute_hrp_weights(returns)
    
    assert sum(weights.values()) == pytest.approx(1.0)
    assert "A" in weights
    assert "B" in weights
    assert "C" in weights
    
    # Check that C gets significant weight (it's low vol and uncorrelated to A/B cluster)
    # A and B share a cluster, so they split that cluster's weight.
    # C is its own cluster.
    
    # Just ensure it runs and produces valid weights.
    assert all(w >= 0 for w in weights.values())

def test_risk_manager_defensive():
    rm = risk_manager.RiskManager()
    
    # Breadth: Defensive
    breadth = {"bpi": 10.0, "ad_slope": -1.0}
    
    # Signal: Bullish
    signal = ModelSignal(horizons=["3D"], direction_probs={"3D": 0.8})
    
    # Portfolio: Cash
    pf = PortfolioState()
    
    # Expect NO_NEW_RISK
    decision = rm.evaluate("AAPL", signal, pf, breadth)
    assert decision.action == types.ActionType.NO_NEW_RISK

def test_risk_manager_normal_buy():
    rm = risk_manager.RiskManager()
    
    # Breadth: Normal
    breadth = {"bpi": 60.0, "ad_slope": 1.0}
    
    # Signal: Bullish
    signal = ModelSignal(horizons=["3D"], direction_probs={"3D": 0.7})
    
    pf = PortfolioState()
    
    decision = rm.evaluate("AAPL", signal, pf, breadth)
    assert decision.action == types.ActionType.BUY
    assert decision.quantity > 0
