import pytest
import numpy as np
from backend.app.models.teacher_o_runner import TeacherORunner
from backend.app.models.tiny_o_runner import TinyORunner
from backend.app.inference.spec_decode_options import SpecDecodeEngine

def test_spec_decode_run():
    tiny = TinyORunner(seed=42)
    teacher = TeacherORunner(seed=42)
    engine = SpecDecodeEngine(tiny, teacher)
    
    dummy_input = np.random.rand(1, 10, 4)
    
    result = engine.generate(dummy_input, steps=10)
    
    assert "tokens" in result
    assert "accept_rate" in result
    assert isinstance(result["accept_rate"], float)
    assert 0.0 <= result["accept_rate"] <= 1.0
    
    # Check that we got tokens (mock usually accepts some)
    # With seed 42, we expect some result
    assert len(result["tokens"]) <= len(result["proposed"])
    
    # If fallback triggered, tokens count < proposed count
    if result["fallback_triggered"]:
        assert len(result["tokens"]) < len(result["proposed"])
    else:
        assert len(result["tokens"]) == len(result["proposed"])
