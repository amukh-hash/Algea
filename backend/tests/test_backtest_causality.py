import pandas as pd

from algaie.data.features.build import build_features
from algaie.data.signals.build import build_signals
from algaie.models.foundation.chronos2 import FoundationModelConfig, SimpleChronos2


def test_backtest_causality_signals_truncated_match():
    canonical = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "ticker": ["AAA", "AAA", "AAA"],
            "open": [10.0, 11.0, 12.0],
            "high": [10.5, 11.5, 12.5],
            "low": [9.5, 10.5, 11.5],
            "close": [10.0, 11.0, 12.0],
            "volume": [100, 110, 120],
        }
    )
    full_features = build_features(canonical)
    truncated_features = build_features(canonical[canonical["date"] <= "2024-01-02"])
    model = SimpleChronos2(FoundationModelConfig())
    full_priors = model.infer(canonical)
    truncated_priors = model.infer(canonical[canonical["date"] <= "2024-01-02"])
    full_signals = build_signals(full_features, full_priors)
    truncated_signals = build_signals(truncated_features, truncated_priors)
    full_day = full_signals[full_signals["date"] == pd.Timestamp("2024-01-02")]
    trunc_day = truncated_signals[truncated_signals["date"] == pd.Timestamp("2024-01-02")]
    pd.testing.assert_frame_equal(full_day.reset_index(drop=True), trunc_day.reset_index(drop=True))
