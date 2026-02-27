import pandas as pd

from algea.research.walk_forward import build_walk_forward_splits


def test_walk_forward_splits_holdout_and_no_overlap():
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    splits = build_walk_forward_splits(
        dates,
        train_window_days=40,
        test_window_days=10,
        step_days=10,
        expanding=True,
        holdout_pct=0.2,
    )
    assert splits
    holdout_start = dates[int(len(dates) * (1 - 0.2))]
    for split in splits:
        assert split.train_end < split.test_start
        assert split.test_end < holdout_start
