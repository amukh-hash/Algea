import pytest
import numpy as np
import pandas as pd
import torch
from unittest.mock import MagicMock
from scripts.run_live_bolt import (
    volatility_target_scalar, 
    kelly_sizing, 
    RegimeHmm, 
    get_meta_confidence,
    perform_daytime_update, 
    compute_hrp_weights
)

def test_volatility_target_scalar():
    # If current vol < target (0.15), scalar should be > 1
    scalar = volatility_target_scalar(0.01, target_vol=0.15)
    assert scalar > 1.0
    assert abs(scalar - 15.0) < 0.1
    
    # If high vol, scalar < 1
    scalar_high = volatility_target_scalar(0.30, target_vol=0.15)
    assert scalar_high < 1.0
    assert abs(scalar_high - 0.5) < 0.1

def test_kelly_sizing():
    # p=0.6, b=2.0 (Odds)
    # Kelly = (0.6*2 - 0.4)/2 = (1.2 - 0.4)/2 = 0.8 / 2 = 0.4
    # Half-Kelly = 0.2
    size = kelly_sizing(0.6, barrier_ratio=2.0)
    assert abs(size - 0.2) < 0.01
    
    # p=0.3 -> Negative -> Size 0
    size_neg = kelly_sizing(0.3, barrier_ratio=2.0)
    assert size_neg == 0.0

def test_regime_hmm_logic():
    hmm = RegimeHmm()
    # Mock model
    hmm.model = MagicMock()
    hmm.model.n_components = 2
    hmm.model.covars_ = np.array([[[0.0001]], [[0.01]]]) # State 0 Low, State 1 High
    hmm.is_fitted = True
    
    # Case 1: Predicts State 0 (Low Vol/Calm) -> Multiplier 1.0
    hmm.model.predict.return_value = np.array([0])
    mult_calm = hmm.get_risk_multiplier(np.random.randn(20))
    assert mult_calm == 1.0
    
    # Case 2: Predicts State 1 (High Vol/Turbulent) -> Multiplier 0.5
    hmm.model.predict.return_value = np.array([1])
    mult_turb = hmm.get_risk_multiplier(np.random.randn(20))
    assert mult_turb == 0.5

def test_judge_meta_confidence():
    # Mock judge model
    judge = MagicMock()
    # predict_proba returns [prob_0, prob_1]
    judge.predict_proba.return_value = np.array([[0.2, 0.8]])
    
    # Quantiles: q10=-1, q50=0, q90=1
    # Spread = (1 - -1)/0 -> Error handling? 
    # Logic uses abs(q50)+1e-8.
    
    q_tensor = torch.tensor([[[ -1.0], [0.0], [1.0] ]]) # (1, 3, 1)
    
    conf = get_meta_confidence(judge, q_tensor.numpy(), [0.1])
    assert conf == 0.8
    
    # Check if correct features passed
    # Spread = 2.0 / 1e-8 -> Big number
    features = judge.predict_proba.call_args[0][0] # First arg
    assert features.shape == (1, 2) # [0.1, spread]

def test_hrp_weights():
    # Mock HRP
    # Just check it returns series of 1.0 for single asset fallback
    # Creating a full riskfolio test setup is heavy, checking fallback logic
    df = pd.DataFrame(np.random.randn(10, 1), columns=['A'])
    # Length < 50 -> Fallback
    w = compute_hrp_weights(df)
    assert w['A'] == 1.0

def test_perform_daytime_update_runs():
    # Smoke test for noise injection
    model = MagicMock()
    model.device = 'cpu'
    # Mock encoder
    model.encoder = MagicMock()
    # Ensure requires_grad=True so backward() doesn't fail
    mock_output = torch.randn(1, 10, 32, requires_grad=True)
    model.encoder.return_value.last_hidden_state = mock_output
    
    tokenizer = MagicMock()
    tokenizer.context_input_transform.return_value = (torch.zeros(1, 10), None, None)
    
    optimizer = MagicMock()
    
    window = np.random.randn(32)
    
    # We need to ensure we don't catch the error in test if we want to debug
    # But code catches it.
    
    perform_daytime_update(model, tokenizer, window, optimizer)
    
    # Assert optimizer step called
    assert optimizer.step.called
