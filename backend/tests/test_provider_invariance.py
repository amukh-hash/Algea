"""Tests for R1: Provider invariance report."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sleeves.cooc_reversal_futures.pipeline.session_semantics import (
    build_provider_invariance_report,
    _compute_baseline_proxy_correlation,
    _compute_r_co_quantiles,
    _compute_missing_open_close,
)
from sleeves.cooc_reversal_futures.pipeline.types import ProviderInvarianceReport


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bars(
    roots: list[str],
    n_days: int = 60,
    seed: int = 0,
    noise_bps: float = 0.0,
) -> pd.DataFrame:
    """Create synthetic daily bars for testing.

    If noise_bps > 0, open/close are perturbed, simulating provider drift.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for root in roots:
        base_price = 100.0 + rng.uniform(-20, 20)
        for d in range(n_days):
            day = pd.Timestamp("2025-01-02") + pd.Timedelta(days=d)
            o = base_price + rng.normal(0, 0.5)
            c = o + rng.normal(0, 0.3)
            # add perturb
            noise_mult = 1 + rng.normal(0, noise_bps * 1e-4)
            o *= noise_mult
            c *= noise_mult
            r_co = (o - c) / c if c != 0 else 0.0  # close-to-open proxy
            r_oc = (c - o) / o if o != 0 else 0.0  # open-to-close
            rows.append({
                "root": root,
                "trading_day": day.date(),
                "open": o,
                "close": c,
                "ret_co": r_co,
                "ret_oc": r_oc,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBaselineProxyCorrelation:
    def test_identical_bars_perfect_corr(self):
        df = _make_bars(["ES", "NQ"], n_days=30)
        corr = _compute_baseline_proxy_correlation(df, df.copy(), ["ES", "NQ"])
        assert set(corr.keys()) == {"ES", "NQ"}
        for root, c in corr.items():
            assert c == pytest.approx(1.0, abs=1e-10), f"{root} corr={c}"

    def test_noisy_bars_still_high_corr(self):
        df_a = _make_bars(["ES"], n_days=60, seed=0, noise_bps=0.0)
        # Add small noise to provider B
        df_b = df_a.copy()
        rng = np.random.default_rng(99)
        df_b["ret_co"] = df_b["ret_co"] + rng.normal(0, 1e-5, len(df_b))
        corr = _compute_baseline_proxy_correlation(df_a, df_b, ["ES"])
        assert corr["ES"] > 0.99

    def test_insufficient_data_returns_nan(self):
        df_a = _make_bars(["ES"], n_days=3)  # too few days
        corr = _compute_baseline_proxy_correlation(df_a, df_a.copy(), ["ES"])
        assert np.isnan(corr["ES"])


class TestQuantileComparison:
    def test_identical_bars_quantiles_match(self):
        df = _make_bars(["CL"], n_days=40)
        q = _compute_r_co_quantiles(df, df.copy(), ["CL"])
        assert "CL" in q
        for qk in ("q10", "q25", "q50", "q75", "q90"):
            assert q["CL"][f"{qk}_a"] == pytest.approx(q["CL"][f"{qk}_b"], abs=1e-10)


class TestMissingOpenClose:
    def test_no_missing(self):
        df = _make_bars(["GC"], n_days=20)
        m = _compute_missing_open_close(df, df.copy(), ["GC"])
        assert m["GC"]["provider_a_missing"] == 0
        assert m["GC"]["provider_b_missing"] == 0

    def test_with_missing(self):
        df = _make_bars(["GC"], n_days=20)
        df_b = df.copy()
        df_b.loc[0:2, "open"] = np.nan
        m = _compute_missing_open_close(df, df_b, ["GC"])
        assert m["GC"]["provider_a_missing"] == 0
        assert m["GC"]["provider_b_missing"] == 3  # rows 0, 1, 2


class TestBuildProviderInvarianceReport:
    def test_consistent_providers_pass(self):
        df = _make_bars(["ES", "NQ"], n_days=30)
        report = build_provider_invariance_report(df, df.copy(), ["ES", "NQ"])
        assert isinstance(report, ProviderInvarianceReport)
        assert report.overall_consistent is True
        assert len(report.flags) == 0

    def test_low_corr_flagged(self):
        df_a = _make_bars(["ES"], n_days=60, seed=0)
        df_b = df_a.copy()
        # Scramble ret_co to kill correlation
        rng = np.random.default_rng(42)
        df_b["ret_co"] = rng.normal(0, 0.01, len(df_b))
        report = build_provider_invariance_report(
            df_a, df_b, ["ES"],
            correlation_threshold=0.90,
        )
        assert report.overall_consistent is False
        assert any("baseline proxy corr" in f for f in report.flags)

    def test_report_persists_json(self):
        df = _make_bars(["ES"], n_days=20)
        with tempfile.TemporaryDirectory() as tmpdir:
            report = build_provider_invariance_report(
                df, df.copy(), ["ES"],
                output_dir=tmpdir,
            )
            json_path = Path(tmpdir) / "provider_invariance_report.json"
            assert json_path.exists()
            data = json.loads(json_path.read_text(encoding="utf-8"))
            assert "baseline_proxy_correlation" in data
            assert data["overall_consistent"] is True

    def test_to_dict_round_trip(self):
        df = _make_bars(["CL"], n_days=20)
        report = build_provider_invariance_report(df, df.copy(), ["CL"])
        d = report.to_dict()
        assert "session_semantics" in d
        assert "baseline_proxy_correlation" in d
        assert "flags" in d
        assert isinstance(d["flags"], list)
