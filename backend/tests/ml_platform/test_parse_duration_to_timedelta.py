from datetime import timedelta

import pytest

from backend.app.ml_platform.utils.downsample import parse_duration_to_timedelta


def test_parse_duration_valid():
    assert parse_duration_to_timedelta("1min") == timedelta(minutes=1)
    assert parse_duration_to_timedelta("5min") == timedelta(minutes=5)
    assert parse_duration_to_timedelta("1h") == timedelta(hours=1)
    assert parse_duration_to_timedelta("30s") == timedelta(seconds=30)


def test_parse_duration_invalid():
    with pytest.raises(ValueError, match="invalid duration"):
        parse_duration_to_timedelta("abc")
    with pytest.raises(ValueError, match="invalid duration"):
        parse_duration_to_timedelta("10x")
