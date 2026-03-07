from __future__ import annotations


def zeros(n):
    return [0.0] * int(n)


class _Array:
    def __init__(self, value):
        self._value = value

    def item(self):
        return self._value


def asarray(v):
    if isinstance(v, _Array):
        return v
    return _Array(v)
