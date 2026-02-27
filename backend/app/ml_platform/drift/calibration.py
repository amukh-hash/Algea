def expected_calibration_error(confidence: float, accuracy: float) -> float:
    return abs(confidence - accuracy)
