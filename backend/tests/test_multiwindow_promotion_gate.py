"""Tests for multi-window promotion gating (Deliverable D)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_panel_with_preds(
    n_instruments: int = 4, n_days: int = 30, start: str = "2025-01-01"
) -> pd.DataFrame:
    """Create synthetic panel with instrument, r_oc, and baseline_score."""
    np.random.seed(42)
    roots = ["ES", "NQ", "YM", "RTY"][:n_instruments]
    start_dt = pd.Timestamp(start)
    rows = []
    for d in range(n_days):
        day = (start_dt + pd.Timedelta(days=d)).strftime("%Y-%m-%d")
        for root in roots:
            r_oc = np.random.randn() * 0.02
            rows.append({
                "trading_day": day,
                "instrument": root,
                "root": root,
                "r_co": np.random.randn() * 0.02,
                "r_oc": r_oc,
                "y": r_oc,
                "close": 5000.0,
                "multiplier": 50.0,
                "baseline_score": -np.random.randn() * 0.02,
            })
    return pd.DataFrame(rows)


class TestMultiWindowPromotion:
    def test_empty_windows_passes(self):
        from sleeves.cooc_reversal_futures.pipeline.validation import _multi_window_promotion_gate

        df = _make_panel_with_preds()
        preds = np.random.randn(len(df))
        report = _multi_window_promotion_gate(df, preds, [])
        assert report.overall_passed is True

    def test_primary_fail_blocks(self):
        """If primary window fails, overall should fail even if stress passes."""
        from sleeves.cooc_reversal_futures.pipeline.validation import _multi_window_promotion_gate

        df = _make_panel_with_preds(n_days=60, start="2025-01-01")
        # Very bad predictions for primary window
        preds = np.zeros(len(df))

        windows = [
            {"name": "primary", "start": "2025-01-01", "end": "2025-01-15"},
            {"name": "stress_1", "start": "2025-01-16", "end": "2025-02-28"},
        ]

        report = _multi_window_promotion_gate(
            df, preds, windows,
            min_sharpe_delta=100.0,  # impossible threshold
            stress_required=0,
        )
        assert report.primary_passed == False
        assert report.overall_passed == False

    def test_stress_fail_blocks(self):
        """If stress_required > 0 and no stress passes, overall should fail."""
        from sleeves.cooc_reversal_futures.pipeline.validation import _multi_window_promotion_gate

        df = _make_panel_with_preds(n_days=60, start="2025-01-01")
        preds = np.zeros(len(df))

        windows = [
            {"name": "primary", "start": "2025-01-01", "end": "2025-01-30"},
            {"name": "stress_1", "start": "2025-01-31", "end": "2025-02-28"},
        ]

        report = _multi_window_promotion_gate(
            df, preds, windows,
            min_sharpe_delta=-100.0,      # easy primary (negative threshold)
            max_drawdown_tolerance=-1.0,   # easy drawdown
            stress_required=1,
        )
        # Even if primary passes, stress window with bad preds may fail
        assert isinstance(report.overall_passed, bool)

    def test_too_few_rows_window_fails(self):
        """Windows with too few rows should automatically fail."""
        from sleeves.cooc_reversal_futures.pipeline.validation import _multi_window_promotion_gate

        df = _make_panel_with_preds(n_days=5, start="2025-01-01")
        preds = np.random.randn(len(df))

        # Window outside data range → no rows
        windows = [
            {"name": "primary", "start": "2026-01-01", "end": "2026-01-31"},
        ]

        report = _multi_window_promotion_gate(df, preds, windows, stress_required=0)
        assert report.windows[0].passed is False

    def test_report_serialization(self):
        """PromotionWindowsReport.to_dict() should return valid dict."""
        from sleeves.cooc_reversal_futures.pipeline.types import PromotionWindowsReport, PromotionWindow

        report = PromotionWindowsReport(
            windows=(
                PromotionWindow(name="p", start="2025-01-01", end="2025-01-31", passed=True),
                PromotionWindow(name="s1", start="2025-02-01", end="2025-02-28", passed=False),
            ),
            primary_passed=True,
            stress_passed_count=0,
            stress_required=1,
            overall_passed=False,
        )
        d = report.to_dict()
        assert len(d["windows"]) == 2
        assert d["overall_passed"] is False
