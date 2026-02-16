"""Tests for risk target transform in PanelDataset (Deliverable A)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="torch required")


def _make_dataset(n_instruments: int = 4, n_days: int = 5) -> pd.DataFrame:
    """Create synthetic dataset with r_oc column."""
    np.random.seed(42)
    rows = []
    for d in range(n_days):
        day = f"2025-01-{10 + d:02d}"
        for i in range(n_instruments):
            rows.append({
                "trading_day": day,
                "instrument": f"INST_{i}",
                "r_co": np.random.randn() * 0.02,
                "r_oc": np.random.randn() * 0.03,
                "y": np.random.randn() * 0.01,
                "feat_a": np.random.randn(),
                "feat_b": np.random.randn(),
            })
    return pd.DataFrame(rows)


class TestRiskTargetTransform:
    def test_y_risk_equals_log_abs(self):
        """y_risk should equal log(eps + |r_oc|) for default transform."""
        from sleeves.cooc_reversal_futures.pipeline.panel_dataset import PanelDataset

        df = _make_dataset()
        eps = 1e-6
        ds = PanelDataset(df, ["feat_a", "feat_b"], "y",
                          extra_cols=["r_oc"], risk_target_eps=eps)

        item = ds[0]
        assert "y_risk" in item, "y_risk key must be present"

        y_risk = item["y_risk"].numpy()
        n = item["n_instruments"]
        r_oc = item["r_oc"].numpy()[:n]
        expected = np.log(eps + np.abs(r_oc))
        np.testing.assert_allclose(y_risk[:n], expected, rtol=1e-5)

    def test_y_risk_finite(self):
        """y_risk values must all be finite."""
        from sleeves.cooc_reversal_futures.pipeline.panel_dataset import PanelDataset

        df = _make_dataset()
        ds = PanelDataset(df, ["feat_a", "feat_b"], "y", extra_cols=["r_oc"])
        for i in range(len(ds)):
            item = ds[i]
            assert torch.isfinite(item["y_risk"]).all(), f"Non-finite y_risk on day {i}"

    def test_y_risk_nonnegative_for_log_abs(self):
        """For log_abs transform with eps>=1e-6, y_risk is always > -14."""
        from sleeves.cooc_reversal_futures.pipeline.panel_dataset import PanelDataset

        df = _make_dataset()
        ds = PanelDataset(df, ["feat_a", "feat_b"], "y", extra_cols=["r_oc"])
        for i in range(len(ds)):
            item = ds[i]
            n = item["n_instruments"]
            # log(1e-6) ≈ -13.8, so y_risk should be > -14
            assert (item["y_risk"][:n] > -14.0).all()

    def test_raw_abs_transform(self):
        """raw_abs transform should give |r_oc| directly."""
        from sleeves.cooc_reversal_futures.pipeline.panel_dataset import PanelDataset

        df = _make_dataset()
        ds = PanelDataset(df, ["feat_a", "feat_b"], "y",
                          extra_cols=["r_oc"], risk_target_transform="raw_abs")
        item = ds[0]
        n = item["n_instruments"]
        r_oc = item["r_oc"].numpy()[:n]
        np.testing.assert_allclose(item["y_risk"].numpy()[:n], np.abs(r_oc), rtol=1e-5)

    def test_no_y_risk_without_r_oc(self):
        """If no r_oc in extra_cols, y_risk should not appear."""
        from sleeves.cooc_reversal_futures.pipeline.panel_dataset import PanelDataset

        df = _make_dataset()
        ds = PanelDataset(df, ["feat_a", "feat_b"], "y", extra_cols=[])
        item = ds[0]
        assert "y_risk" not in item
