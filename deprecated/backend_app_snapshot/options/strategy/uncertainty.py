import numpy as np

def compute_uncertainty_ratio(predicted_uncertainty: float, iv: float) -> float:
    """
    Ratio > 1.0 means model is more uncertain than market (IV).
    Ratio < 1.0 means model is more confident.
    """
    if iv <= 0:
        return 999.0
    return predicted_uncertainty / iv

def normalize_uncertainty(raw_uncertainty: float, history_mean: float = 0.5, history_std: float = 0.1) -> float:
    """
    Z-score normalization or similar.
    """
    if history_std <= 0:
        return 0.0
    return (raw_uncertainty - history_mean) / history_std
