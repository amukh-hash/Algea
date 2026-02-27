from backend.app.ml_platform.drift.detectors import zscore_ood_score


def test_drift_scoring_uses_baseline() -> None:
    score = zscore_ood_score([100.0, 101.0, 102.0], {"mean": 100.0, "std": 1.0})
    assert score > 0
