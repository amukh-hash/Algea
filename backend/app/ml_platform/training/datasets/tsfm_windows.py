from __future__ import annotations


def build_tsfm_windows(series: list[float], context_length: int, prediction_length: int) -> list[tuple[list[float], list[float]]]:
    windows: list[tuple[list[float], list[float]]] = []
    span = context_length + prediction_length
    if len(series) < span:
        return windows
    for i in range(0, len(series) - span + 1):
        chunk = series[i : i + span]
        windows.append((chunk[:context_length], chunk[context_length:]))
    return windows
