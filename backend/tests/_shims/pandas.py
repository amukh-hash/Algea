from __future__ import annotations

from datetime import timedelta


class DataFrame:
    pass


def to_timedelta(value):
    if isinstance(value, timedelta):
        return value
    if isinstance(value, str) and value.endswith('min'):
        return timedelta(minutes=int(value[:-3]))
    if isinstance(value, str) and value.endswith('h'):
        return timedelta(hours=int(value[:-1]))
    if isinstance(value, (int, float)):
        return timedelta(seconds=float(value))
    raise ValueError(f"unsupported timedelta value: {value}")
