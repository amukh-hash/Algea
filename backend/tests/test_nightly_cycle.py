"""Nightly cycle smoke tests — verify full pipeline with stub provider.

Tests the end-to-end nightly cycle including features, priors, signals,
and strategy execution without requiring HuggingFace or network access.
"""
import pytest
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from algae.models.foundation.base import StatisticalFallbackProvider
from backend.scripts.run.run_nightly_cycle import run


def _make_canonical(n_days: int = 30) -> pd.DataFrame:
    """Generate deterministic canonical daily data with enough rows."""
    np.random.seed(999)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    close = 100 + np.cumsum(np.random.randn(n_days) * 0.5)
    return pd.DataFrame({
        "date": dates,
        "ticker": "AAA",
        "open": close - 0.5,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": 1000 + np.random.randint(0, 200, n_days),
    })


def test_nightly_cycle_smoke(tmp_path: Path) -> None:
    """Full nightly cycle with stub provider — no HuggingFace needed."""
    config_path = tmp_path / "config.json"
    artifact_root = tmp_path / "artifacts"
    config_path.write_text(json.dumps({"artifact_root": str(artifact_root)}), encoding="utf-8")

    df = _make_canonical(30)
    input_path = tmp_path / "canonical.parquet"
    df.to_parquet(input_path, index=False)

    asof = date.fromisoformat("2024-02-09")  # After all data
    provider = StatisticalFallbackProvider()
    run(str(config_path), [input_path], asof, model_provider=provider)

    summary_path = artifact_root / "reports" / "nightly" / str(asof) / "summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["asof"] == str(asof)
    assert summary["signals"] >= 0


def test_nightly_cycle_malformed_config(tmp_path: Path) -> None:
    """Nightly cycle with invalid JSON config should fail informatively."""
    config_path = tmp_path / "config.json"
    config_path.write_text("{not valid json!!!", encoding="utf-8")

    df = _make_canonical(5)
    input_path = tmp_path / "canonical.parquet"
    df.to_parquet(input_path, index=False)

    with pytest.raises(Exception):
        provider = StatisticalFallbackProvider()
        run(str(config_path), [input_path], date.fromisoformat("2024-01-10"), model_provider=provider)


def test_nightly_cycle_empty_data(tmp_path: Path) -> None:
    """Nightly cycle with no eligible tickers should produce empty signals."""
    config_path = tmp_path / "config.json"
    artifact_root = tmp_path / "artifacts"
    config_path.write_text(json.dumps({"artifact_root": str(artifact_root)}), encoding="utf-8")

    # Only 1 row — not enough for any features or priors
    df = pd.DataFrame({
        "date": ["2024-01-01"],
        "ticker": ["AAA"],
        "open": [10.0], "high": [12.0], "low": [9.0], "close": [11.0],
        "volume": [100.0],
    })
    input_path = tmp_path / "canonical.parquet"
    df.to_parquet(input_path, index=False)

    provider = StatisticalFallbackProvider()
    run(str(config_path), [input_path], date.fromisoformat("2024-01-02"), model_provider=provider)

    summary_path = artifact_root / "reports" / "nightly" / "2024-01-02" / "summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["signals"] == 0
