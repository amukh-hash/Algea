"""H1 tests — verify risky feature set is derived from FeatureSpec."""
from __future__ import annotations

import pytest

from backend.app.features.feature_spec import (
    FEATURE_SPECS,
    PREOPEN_FEATURES,
    risky_features_for_preopen,
    preopen_features,
)


class TestRiskyFeatureSet:
    """The risky set must include label/future columns and exclude preopen inputs."""

    def test_r_oc_is_risky(self) -> None:
        risky = risky_features_for_preopen()
        assert "r_oc" in risky

    def test_ret_oc_is_risky(self) -> None:
        risky = risky_features_for_preopen()
        assert "ret_oc" in risky

    def test_sigma_oc_hist_not_risky(self) -> None:
        """sigma_oc_hist (rolling std of prior r_oc) is causal → not risky."""
        risky = risky_features_for_preopen()
        assert "sigma_oc_hist" not in risky

    def test_preopen_features_not_risky(self) -> None:
        """No preopen feature should be in the risky set."""
        risky = risky_features_for_preopen()
        for feat in preopen_features():
            assert feat not in risky, f"Preopen feature '{feat}' should not be risky"

    def test_risky_includes_all_label_or_future(self) -> None:
        """All features with is_label_or_future=True must be in risky set."""
        risky = risky_features_for_preopen()
        for name, spec in FEATURE_SPECS.items():
            if spec.is_label_or_future:
                assert name in risky, f"'{name}' has is_label_or_future=True but not in risky"

    def test_risky_includes_non_preopen(self) -> None:
        """Features with availability != preopen must be in risky set."""
        risky = risky_features_for_preopen()
        for name, spec in FEATURE_SPECS.items():
            if spec.availability != "preopen":
                assert name in risky, f"'{name}' availability={spec.availability} but not in risky"

    def test_risky_returns_frozenset(self) -> None:
        risky = risky_features_for_preopen()
        assert isinstance(risky, frozenset)

    def test_no_duplicate_risky_definitions(self) -> None:
        """KNOWN_RISKY_FEATURES must not exist in dataset.py anymore."""
        import sleeves.cooc_reversal_futures.pipeline.dataset as ds
        assert not hasattr(ds, "KNOWN_RISKY_FEATURES"), (
            "KNOWN_RISKY_FEATURES still exists in dataset.py — should be removed"
        )
