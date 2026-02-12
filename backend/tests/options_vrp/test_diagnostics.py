"""Tests for rolling performance diagnostics (Phase 8)."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backend.app.evaluation.diagnostics import (
    DiagnosticsInput,
    compute_diagnostics,
    rolling_es,
    rolling_forecast_calibration_error,
    rolling_sharpe,
    rolling_sortino,
    save_diagnostics,
)


def _make_pnl(n: int = 200) -> pd.Series:
    np.random.seed(42)
    return pd.Series(
        np.random.normal(0.001, 0.01, n),
        index=pd.bdate_range("2024-01-01", periods=n),
    )


class TestRollingSharpe:
    def test_shape(self):
        pnl = _make_pnl()
        sharpe = rolling_sharpe(pnl, window=63)
        assert len(sharpe) == len(pnl)

    def test_nan_at_start(self):
        pnl = _make_pnl()
        sharpe = rolling_sharpe(pnl, window=63)
        assert pd.isna(sharpe.iloc[0])

    def test_positive_for_positive_drift(self):
        pnl = pd.Series(np.random.normal(0.005, 0.01, 200),
                         index=pd.bdate_range("2024-01-01", periods=200))
        sharpe = rolling_sharpe(pnl, window=63)
        # Tail should be positive
        assert sharpe.dropna().iloc[-1] > 0


class TestRollingSortino:
    def test_shape(self):
        pnl = _make_pnl()
        sortino = rolling_sortino(pnl, window=63)
        assert len(sortino) == len(pnl)


class TestRollingES:
    def test_negative_for_mixed_pnl(self):
        pnl = _make_pnl()
        es = rolling_es(pnl, window=63, level=0.95)
        valid = es.dropna()
        assert valid.iloc[-1] < 0  # ES should be negative (worst tail)


class TestForecastCalibration:
    def test_zero_error_for_perfect_forecast(self):
        n = 200
        idx = pd.bdate_range("2024-01-01", periods=n)
        rv = pd.Series(np.random.uniform(0.10, 0.20, n), index=idx)
        mae = rolling_forecast_calibration_error(rv, rv, window=63)
        valid = mae.dropna()
        assert valid.iloc[-1] < 1e-10


class TestDiagnosticsRunner:
    def test_produces_all_columns(self):
        n = 200
        idx = pd.bdate_range("2024-01-01", periods=n)
        inp = DiagnosticsInput(
            daily_pnl=pd.Series(np.random.normal(0, 0.01, n), index=idx),
            scenario_losses=pd.Series(np.random.uniform(-0.05, 0, n), index=idx),
            forecast_rv=pd.Series(np.random.uniform(0.10, 0.25, n), index=idx),
            realised_rv=pd.Series(np.random.uniform(0.10, 0.25, n), index=idx),
            convexity_scores=pd.Series(np.random.uniform(10, 50, n), index=idx),
        )
        df = compute_diagnostics(inp)
        expected = {
            "rolling_sharpe", "rolling_sortino", "rolling_es95",
            "rolling_worst_scenario", "rolling_forecast_mae", "rolling_convexity",
        }
        assert expected.issubset(set(df.columns))

    def test_save_creates_csv(self):
        n = 100
        idx = pd.bdate_range("2024-01-01", periods=n)
        inp = DiagnosticsInput(
            daily_pnl=pd.Series(np.random.normal(0, 0.01, n), index=idx),
            scenario_losses=pd.Series(np.random.uniform(-0.05, 0, n), index=idx),
            forecast_rv=pd.Series(np.random.uniform(0.10, 0.25, n), index=idx),
            realised_rv=pd.Series(np.random.uniform(0.10, 0.25, n), index=idx),
            convexity_scores=pd.Series(np.random.uniform(10, 50, n), index=idx),
        )
        df = compute_diagnostics(inp)
        with tempfile.TemporaryDirectory() as tmp:
            path = save_diagnostics(df, Path(tmp))
            assert path.exists()
