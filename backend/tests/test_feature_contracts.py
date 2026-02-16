"""Tests for the feature contract system (G0/H1/H3).

Verifies:
  - FeatureSpec structure and FEATURE_SPECS registry completeness
  - sigma_oc_hist classified as preopen (causal rolling std)
  - sigma_oc alias is backward-compatible
  - Required feature timestamp violation → hard fail
  - Optional feature drop → proceed without KeyError
  - Feature guard returns structured result with kept/dropped lists
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.app.features.feature_spec import (
    FEATURE_SPECS,
    PREOPEN_FEATURES,
    REQUIRED_PREOPEN_FEATURES,
    OPTIONAL_PREOPEN_FEATURES,
    FeatureSpec,
    risky_features_for_preopen,
)
from sleeves.cooc_reversal_futures.features_core import (
    FEATURE_SCHEMA_V1,
    FEATURE_SCHEMA_V2,
)
from sleeves.cooc_reversal_futures.pipeline.dataset import (
    run_feature_timestamp_guard,
)


# ---------------------------------------------------------------------------
# FeatureSpec structure tests
# ---------------------------------------------------------------------------

class TestFeatureSpecStructure:
    """Test the FeatureSpec dataclass and registry."""

    def test_all_v2_features_have_specs(self) -> None:
        """Every V2 schema feature must have a corresponding FeatureSpec."""
        for feat in FEATURE_SCHEMA_V2:
            assert feat in FEATURE_SPECS, f"V2 feature '{feat}' missing from FEATURE_SPECS"

    def test_all_v1_features_required(self) -> None:
        """V1 core features should all be non-optional."""
        for feat in FEATURE_SCHEMA_V1:
            spec = FEATURE_SPECS[feat]
            assert not spec.optional, f"V1 feature '{feat}' should be required, not optional"

    def test_feature_spec_frozen(self) -> None:
        """FeatureSpec should be immutable."""
        spec = FEATURE_SPECS["r_co"]
        with pytest.raises(AttributeError):
            spec.name = "hacked"  # type: ignore[misc]

    def test_preopen_features_derived(self) -> None:
        """PREOPEN_FEATURES should match availability='preopen' specs."""
        for name in PREOPEN_FEATURES:
            assert FEATURE_SPECS[name].availability == "preopen"

    def test_required_optional_partition(self) -> None:
        """Required + optional should partition all preopen features."""
        assert set(REQUIRED_PREOPEN_FEATURES) | set(OPTIONAL_PREOPEN_FEATURES) == set(PREOPEN_FEATURES)
        assert set(REQUIRED_PREOPEN_FEATURES) & set(OPTIONAL_PREOPEN_FEATURES) == set()


# ---------------------------------------------------------------------------
# sigma_oc_hist / sigma_oc alias tests (H3)
# ---------------------------------------------------------------------------

class TestSigmaOCDecision:
    """sigma_oc_hist is causal → must NOT be in risky features."""

    def test_sigma_oc_hist_not_risky(self) -> None:
        risky = risky_features_for_preopen()
        assert "sigma_oc_hist" not in risky

    def test_sigma_oc_hist_is_preopen(self) -> None:
        assert FEATURE_SPECS["sigma_oc_hist"].availability == "preopen"

    def test_sigma_oc_hist_is_required(self) -> None:
        assert not FEATURE_SPECS["sigma_oc_hist"].optional

    def test_sigma_oc_hist_timestamp_rule(self) -> None:
        assert FEATURE_SPECS["sigma_oc_hist"].timestamp_rule == "asof_prev_close"

    def test_sigma_oc_alias_exists(self) -> None:
        """Backward-compat alias sigma_oc must exist in specs."""
        assert "sigma_oc" in FEATURE_SPECS
        assert FEATURE_SPECS["sigma_oc"].availability == "preopen"

    def test_sigma_oc_alias_is_optional(self) -> None:
        """Deprecated alias should be optional (safe to remove later)."""
        assert FEATURE_SPECS["sigma_oc"].optional

    def test_r_oc_still_risky(self) -> None:
        """Raw r_oc (same-day) must remain risky."""
        risky = risky_features_for_preopen()
        assert "r_oc" in risky
        assert "ret_oc" in risky


# ---------------------------------------------------------------------------
# Feature guard integration tests
# ---------------------------------------------------------------------------

def _make_dataset(n: int = 50) -> pd.DataFrame:
    """Minimal dataset for guard tests."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "trading_day": pd.date_range("2025-01-01", periods=n, freq="D"),
        "instrument": "ES",
        "r_co": rng.normal(0, 0.01, n),
        "r_oc": rng.normal(0, 0.01, n),
        "sigma_co": rng.uniform(0.005, 0.02, n),
        "sigma_oc_hist": rng.uniform(0.005, 0.02, n),
        "sigma_oc": rng.uniform(0.005, 0.02, n),
        "volume_z": rng.normal(0, 1, n),
        "y": rng.normal(0, 0.01, n),
    })


class TestFeatureGuard:
    """Test the feature timestamp guard with the new contract system."""

    def test_sigma_oc_hist_survives_guard(self) -> None:
        """sigma_oc_hist should NOT be dropped by the feature guard."""
        df = _make_dataset()
        features = ["r_co", "sigma_co", "sigma_oc_hist", "volume_z"]
        out, report = run_feature_timestamp_guard(
            df, feature_columns=features, strict=False,
        )
        assert "sigma_oc_hist" in out.columns
        assert "sigma_oc_hist" not in report.dropped_features
        assert "sigma_oc_hist" in report.kept_features

    def test_sigma_oc_alias_survives_guard(self) -> None:
        """sigma_oc (alias) should also NOT be dropped."""
        df = _make_dataset()
        features = ["r_co", "sigma_co", "sigma_oc", "volume_z"]
        out, report = run_feature_timestamp_guard(
            df, feature_columns=features, strict=False,
        )
        assert "sigma_oc" in out.columns
        assert "sigma_oc" not in report.dropped_features
        assert "sigma_oc" in report.kept_features

    def test_r_oc_dropped_when_used_as_feature(self) -> None:
        """r_oc (same-day) should be dropped when used as a feature."""
        df = _make_dataset()
        features = ["r_co", "r_oc", "sigma_co"]
        out, report = run_feature_timestamp_guard(
            df, feature_columns=features, strict=False,
        )
        assert "r_oc" in report.dropped_features
        assert "r_co" in report.kept_features
        assert "sigma_co" in report.kept_features

    def test_r_oc_strict_raises(self) -> None:
        """In strict mode, using r_oc as a feature should raise."""
        df = _make_dataset()
        features = ["r_co", "r_oc"]
        with pytest.raises(ValueError, match="risky features"):
            run_feature_timestamp_guard(
                df, feature_columns=features, strict=True,
            )

    def test_missing_feature_tracked(self) -> None:
        """Features not in the DataFrame should appear in missing_features."""
        df = _make_dataset()
        features = ["r_co", "nonexistent_feat"]
        _out, report = run_feature_timestamp_guard(
            df, feature_columns=features, strict=False,
        )
        assert "nonexistent_feat" in report.missing_features
        assert "r_co" in report.kept_features

    def test_optional_drop_no_keyerror(self) -> None:
        """Dropping an optional feature should not raise."""
        df = _make_dataset()
        features = ["r_co", "sigma_co"]
        out, report = run_feature_timestamp_guard(
            df, feature_columns=features, strict=False,
        )
        assert report.required_dropped is False
        assert len(report.kept_features) == 2

    def test_kept_features_complete(self) -> None:
        """kept_features + dropped + missing should cover all requested."""
        df = _make_dataset()
        features = ["r_co", "r_oc", "sigma_co", "sigma_oc_hist", "missing_one"]
        _out, report = run_feature_timestamp_guard(
            df, feature_columns=features, strict=False,
        )
        reconstructed = set(report.kept_features) | set(report.dropped_features) | set(report.missing_features)
        assert reconstructed == set(features)
