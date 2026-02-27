import pytest

from backend.app.ml_platform.training.datasets.tsfm_windows import build_tsfm_windows
from backend.app.ml_platform.config import MLPlatformConfig


def test_chronos_invalid_duration_throws_builder():
    with pytest.raises(ValueError, match="invalid duration"):
        build_tsfm_windows([1, 2, 3], context_length=1, prediction_length=1, timestamps=["2026-01-01T00:00:00", "2026-01-01T00:01:00", "2026-01-01T00:02:00"], downsample_freq="10bad")


def test_chronos_invalid_duration_throws_config(monkeypatch):
    monkeypatch.setenv("TSFM_DOWNSAMPLE_FREQ_DURATION", "10bad")
    with pytest.raises(ValueError, match="invalid duration"):
        MLPlatformConfig()
