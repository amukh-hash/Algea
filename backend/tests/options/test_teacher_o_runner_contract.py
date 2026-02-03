import pytest
import numpy as np
from backend.app.models.teacher_o_runner import TeacherORunner
from backend.app.models.types import DistributionForecast

def test_teacher_o_runner_contract():
    runner = TeacherORunner(seed=42)
    dummy_input = np.random.rand(1, 10, 4)

    forecast = runner.predict_distribution(dummy_input)

    assert isinstance(forecast, DistributionForecast)
    assert "1D" in forecast.horizons
    assert "0.50" in forecast.quantiles["1D"]
    assert forecast.metadata["mock"] is True

def test_teacher_o_verify_tokens():
    runner = TeacherORunner(seed=42)
    dummy_input = np.random.rand(1, 10, 4)
    tokens = [10, 20, 30]

    valid = runner.verify_tokens(dummy_input, tokens)
    assert len(valid) == len(tokens)
    assert all(isinstance(v, bool) for v in valid)
