"""H2 tests — per-feature timestamp guarding.

Verifies that per-feature cutoff columns (feature_cutoff_ts__<feat>)
are checked individually, and that global-pass + per-feature-fail works.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sleeves.cooc_reversal_futures.pipeline.dataset import (
    run_feature_timestamp_guard,
)


def _make_guarded_df(n: int = 10) -> pd.DataFrame:
    """Dataset with session_open_ts, global cutoff (passing), and per-feature cutoffs.

    Uses 'shock_flag' (optional in FeatureSpec) as the feature with a
    per-feature cutoff that VIOLATES, and 'sigma_co' as one that passes.
    """
    rng = np.random.default_rng(42)
    base = pd.Timestamp("2025-06-01 09:30:00", tz="America/New_York")
    rows = []
    for i in range(n):
        session_open = base + pd.Timedelta(days=i)
        # Global cutoff: safely before open (passes)
        global_cutoff = session_open - pd.Timedelta(hours=1)
        # Per-feature cutoff for 'shock_flag' (OPTIONAL): AFTER open (violates)
        shock_flag_cutoff = session_open + pd.Timedelta(minutes=5)
        # Per-feature cutoff for 'sigma_co': safely before (passes)
        sigma_co_cutoff = session_open - pd.Timedelta(hours=2)

        rows.append({
            "trading_day": (base + pd.Timedelta(days=i)).date(),
            "instrument": "ES",
            "r_co": rng.normal(0, 0.01),
            "sigma_co": rng.uniform(0.005, 0.02),
            "sigma_oc_hist": rng.uniform(0.005, 0.02),
            "sigma_oc": rng.uniform(0.005, 0.02),
            "volume_z": rng.normal(0, 1),
            "shock_flag": float(rng.random() > 0.9),
            "y": rng.normal(0, 0.01),
            "session_open_ts": session_open,
            "feature_cutoff_ts": global_cutoff,
            "feature_cutoff_ts__shock_flag": shock_flag_cutoff,
            "feature_cutoff_ts__sigma_co": sigma_co_cutoff,
        })
    return pd.DataFrame(rows)


class TestPerFeatureCutoff:
    """Per-feature timestamp guard tests."""

    def test_global_passes_per_feature_fails_strict_optional(self) -> None:
        """Global cutoff passes, but per-feature cutoff for optional shock_flag fails.
        strict=True → shock_flag is optional, so it should be dropped (not raised)."""
        df = _make_guarded_df()
        features = ["r_co", "sigma_co", "shock_flag"]
        out, report = run_feature_timestamp_guard(
            df, feature_columns=features,
            session_open_ts_col="session_open_ts",
            strict=True,
        )
        # shock_flag should be dropped (optional)
        assert "shock_flag" in report.dropped_features
        assert "shock_flag" not in out.columns
        # sigma_co and r_co should survive
        assert "r_co" in report.kept_features
        assert "sigma_co" in report.kept_features

    def test_global_passes_per_feature_fails_nonstrict(self) -> None:
        """strict=False: per-feature violation drops optional feature, no error."""
        df = _make_guarded_df()
        features = ["r_co", "sigma_co", "shock_flag"]
        out, report = run_feature_timestamp_guard(
            df, feature_columns=features,
            session_open_ts_col="session_open_ts",
            strict=False,
        )
        assert "shock_flag" in report.dropped_features
        assert "r_co" in report.kept_features

    def test_per_feature_passing_not_dropped(self) -> None:
        """sigma_co has per-feature cutoff that passes → should survive."""
        df = _make_guarded_df()
        features = ["r_co", "sigma_co"]
        out, report = run_feature_timestamp_guard(
            df, feature_columns=features,
            session_open_ts_col="session_open_ts",
            strict=False,
        )
        assert "sigma_co" in report.kept_features
        assert "sigma_co" not in report.dropped_features

    def test_guarded_by_lists_populated(self) -> None:
        """Report must track guarded_by_global and guarded_by_per_feature."""
        df = _make_guarded_df()
        features = ["r_co", "sigma_co", "shock_flag"]
        _out, report = run_feature_timestamp_guard(
            df, feature_columns=features,
            session_open_ts_col="session_open_ts",
            strict=False,
        )
        # sigma_co has per-feature col → guarded_by_per_feature
        assert "sigma_co" in report.guarded_by_per_feature
        # shock_flag also has per-feature col
        assert "shock_flag" in report.guarded_by_per_feature
        # r_co has no per-feature col → guarded_by_global
        assert "r_co" in report.guarded_by_global

    def test_leakage_report_json_contains_guarded_lists(self) -> None:
        """leakage_report.json should include the new guarded lists."""
        df = _make_guarded_df()
        features = ["r_co", "sigma_co", "shock_flag"]
        with tempfile.TemporaryDirectory() as tmpdir:
            _out, report = run_feature_timestamp_guard(
                df, feature_columns=features,
                session_open_ts_col="session_open_ts",
                strict=False,
                output_dir=tmpdir,
            )
            report_path = Path(tmpdir) / "leakage_report.json"
            assert report_path.exists()
            data = json.loads(report_path.read_text())
            assert "guarded_by_global" in data
            assert "guarded_by_per_feature" in data
            assert "unguarded_features" in data

    def test_unguarded_when_no_session_ts(self) -> None:
        """Without session_open_ts, all non-risky features are unguarded."""
        df = _make_guarded_df()
        features = ["r_co", "sigma_co", "shock_flag"]
        _out, report = run_feature_timestamp_guard(
            df, feature_columns=features,
            session_open_ts_col="nonexistent_col",
            strict=False,
        )
        assert set(report.unguarded_features) == {"r_co", "sigma_co", "shock_flag"}
        assert len(report.guarded_by_global) == 0
        assert len(report.guarded_by_per_feature) == 0
