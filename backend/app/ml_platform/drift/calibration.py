from __future__ import annotations


def expected_calibration_error(confidence: float, accuracy: float) -> float:
    return abs(confidence - accuracy)


def interval_coverage_error(intervals: list[tuple[float, float]], realized: list[float], nominal_coverage: float) -> float:
    if not intervals or not realized:
        return 0.0
    hits = 0
    for (lo, hi), y in zip(intervals, realized):
        if lo <= y <= hi:
            hits += 1
    empirical = hits / max(len(realized), 1)
    return abs(empirical - nominal_coverage)
