import numpy as np
import pytest
from backend.app.eval import metrics, promotion_gate

def test_metrics_logic():
    y_true = np.array([1.0, -1.0, 1.0, -1.0])
    # Perfect predictor
    y_pred_med = np.array([0.5, -0.5, 0.5, -0.5])

    acc = metrics.directional_accuracy(y_true, y_pred_med)
    assert acc == 1.0

    # Coverage
    # True: 0, Bounds: -1, 1 -> In
    # True: 2, Bounds: -1, 1 -> Out
    y_t = np.array([0.0, 2.0])
    q_l = np.array([-1.0, -1.0])
    q_u = np.array([1.0, 1.0])

    cov = metrics.coverage_probability(y_t, q_l, q_u)
    assert cov == 0.5

def test_promotion_gate():
    gate = promotion_gate.PromotionGate(accuracy_tol=0.05)

    # Baseline: 0.60
    base = {"accuracy": 0.60, "width_90": 1.0}

    # Case 1: Better
    cand1 = {"accuracy": 0.65, "width_90": 1.0, "coverage_90": 0.90}
    passed, _ = gate.check(cand1, base)
    assert passed

    # Case 2: Worse but within tol
    cand2 = {"accuracy": 0.58, "width_90": 1.0, "coverage_90": 0.90} # Drop 0.02
    passed, _ = gate.check(cand2, base)
    assert passed

    # Case 3: Worse outside tol
    cand3 = {"accuracy": 0.54, "width_90": 1.0, "coverage_90": 0.90} # Drop 0.06
    passed, _ = gate.check(cand3, base)
    assert not passed

    # Case 4: Width explosion
    cand4 = {"accuracy": 0.65, "width_90": 2.0, "coverage_90": 0.90} # Double width
    passed, _ = gate.check(cand4, base)
    assert not passed
