import torch
import pytest
import numpy as np
from app.models.loss import TeacherLoss, StudentMultiObjectiveLoss, teacher_logits_to_quantiles

def test_teacher_loss():
    loss_fn = TeacherLoss()
    # Batch=2, Time=5, Vocab=10
    logits = torch.randn(2, 5, 10)
    targets = torch.randint(0, 10, (2, 5))
    
    loss = loss_fn(logits, targets)
    assert loss.item() > 0
    assert not torch.isnan(loss)

def test_teacher_logits_to_quantiles():
    # Vocab size 5, centers 0-4
    vocab_size = 5
    centers = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
    
    # Create logits that strongly favor specific bins
    # Batch=1, Time=1
    # Example: P = [0.1, 0.1, 0.6, 0.1, 0.1]
    # CDF = [0.1, 0.2, 0.8, 0.9, 1.0]
    # Quantiles: 
    # 0.1 -> index 0 (val 0.0) or index 1 depending on >= logic?
    # CDF[0]=0.1 >= 0.1? Yes -> index 0 -> val 0.0
    # 0.5 -> index 2 (val 2.0)
    # 0.9 -> index 3 (val 3.0)
    
    logits = torch.tensor([[[0.0, 0.0, 5.0, 0.0, 0.0]]]) # High prob at index 2
    
    quantiles_vals = teacher_logits_to_quantiles(logits, centers, quantiles=[0.5])
    
    # Expectation: 2.0
    assert torch.abs(quantiles_vals[0, 0, 0] - 2.0) < 0.1

def test_student_multi_objective_loss():
    loss_fn = StudentMultiObjectiveLoss(risk_free_rate=0.0, quantiles=[0.1, 0.5, 0.9], 
                                        student_dim=10, teacher_dim=10)
    
    # Batch=2, Quantiles=3, Horizon=5
    student_q = torch.randn(2, 3, 5, requires_grad=True)
    teacher_q = torch.randn(2, 3, 5) # Target
    tbm_labels = torch.randint(0, 2, (2,)) # Binary labels
    
    # Forward without hidden
    loss, components = loss_fn(student_q, teacher_q, tbm_labels)
    
    assert loss.item() != 0
    assert not torch.isnan(loss)
    # Check components
    distill, sortino, direction, feature = components
    assert feature == 0.0
    
    # Forward WITH hidden
    student_h = torch.randn(2, 5, 10)
    teacher_h = torch.randn(2, 5, 10)
    
    loss_h, components_h = loss_fn(student_q, teacher_q, tbm_labels, student_h, teacher_h)
    
    assert components_h[3] > 0.0 # Feature loss should be active
    
    # Backprop check
    loss_h.backward()
    assert student_q.grad is not None
