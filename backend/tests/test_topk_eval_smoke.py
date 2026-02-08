
import pytest
import numpy as np
from datetime import datetime, timedelta
from backend.app.eval.topk_basket_eval import TopKBasketEvaluator

def test_topk_eval_metrics():
    """
    Test metric calculation for Top-K evaluation.
    """
    evaluator = TopKBasketEvaluator(k=2, cost_bps=10) # Top 2
    
    # Day 1: Tickers A, B, C. Scores A=10, B=5, C=1. Top 2: A, B.
    # Targets: A=0.01, B=-0.01, C=0.
    date1 = datetime(2023, 1, 1).date()
    scores1 = {"A": 10.0, "B": 5.0, "C": 1.0}
    targets1 = {"A": 0.01, "B": -0.01, "C": 0.0}
    
    evaluator.step(date1, scores1, targets1)
    
    # Assert Day 1
    rec1 = evaluator.history[0]
    assert set(rec1["tickers"]) == {"A", "B"}
    assert rec1["raw_ret"] == 0.0 # (0.01 - 0.01)/2
    assert rec1["turnover"] == 0.0 # First day turnover is 0? Or 1?
    # Logic: if history empty, turnover=0.
    
    # Day 2: Tickers A, C, B. Scores A=10, C=8, B=1. Top 2: A, C.
    # Targets: A=0.02, C=0.02.
    date2 = datetime(2023, 1, 2).date()
    scores2 = {"A": 10.0, "B": 1.0, "C": 8.0}
    targets2 = {"A": 0.02, "B": 0.0, "C": 0.02}
    
    evaluator.step(date2, scores2, targets2)
    
    # Assert Day 2
    rec2 = evaluator.history[1]
    assert set(rec2["tickers"]) == {"A", "C"}
    assert rec2["raw_ret"] == 0.02 # (0.02 + 0.02)/2
    
    # Turnover:
    # Prev: {A, B}. Curr: {A, C}.
    # Overlap: {A} (size 1). K=2.
    # Turnover = (2 - 1) / 2 = 0.5.
    assert rec2["turnover"] == 0.5
    
    # Cost: turnover * 2 * 10bps = 0.5 * 2 * 0.0010 = 0.0010
    assert np.isclose(rec2["cost"], 0.0010)
    
    metrics = evaluator.get_metrics()
    assert metrics["avg_turnover"] == 0.25 # (0 + 0.5)/2
    assert metrics["total_ret"] > 0
    
if __name__ == "__main__":
    test_topk_eval_metrics()
    print("ALL TESTS PASSED")
