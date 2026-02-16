"""Tests for Phase 1.5 alignment & operational readiness modules.

All tests use deterministic synthetic data — no IBKR gateway required.
"""
from __future__ import annotations

import json
import tempfile
from datetime import date
from pathlib import Path
from typing import Dict
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Fixtures: synthetic bars
# ---------------------------------------------------------------------------

def _make_bars(roots: list[str], n_days: int = 60, seed: int = 42,
               perturb_bps: float = 0.0) -> pd.DataFrame:
    """Generate synthetic daily bars with optional perturbation."""
    rng = np.random.default_rng(seed)
    records = []
    base_date = date(2025, 6, 1)
    for root in roots:
        price = 5000.0 if root == "ES" else 18000.0
        for i in range(n_days):
            day = date.fromordinal(base_date.toordinal() + i)
            o = price + rng.uniform(-20, 20) + perturb_bps / 1e4 * price
            c = price + rng.uniform(-20, 20) + perturb_bps / 1e4 * price
            h = max(o, c) + abs(rng.normal(0, 10))
            low = min(o, c) - abs(rng.normal(0, 10))
            records.append({
                "root": root,
                "trading_day": day,
                "timestamp": pd.Timestamp(day, tz="UTC"),
                "open": o,
                "high": h,
                "low": low,
                "close": c,
                "volume": int(rng.uniform(100_000, 500_000)),
                "ret_co": c / o - 1 if o != 0 else 0.0,
                "ret_oc": c / o - 1 if o != 0 else 0.0,
            })
    return pd.DataFrame(records)


# ===========================================================================
# Session Semantics Tests
# ===========================================================================

class TestSessionSemantics:
    """Tests for session_semantics.compare_session_semantics."""

    def test_identical_bars_pass(self) -> None:
        from sleeves.cooc_reversal_futures.pipeline.session_semantics import (
            compare_session_semantics,
        )
        df = _make_bars(["ES", "NQ"], 30)
        report = compare_session_semantics(df, df, ["ES", "NQ"], sample_days=20, seed=42)
        assert report.gate_passed is True
        # All diffs should be zero
        for key, stats in report.per_field_stats.items():
            assert stats["median_bps"] == 0.0 or np.isnan(stats["median_bps"])

    def test_perturbed_bars_detect_diff(self) -> None:
        from sleeves.cooc_reversal_futures.pipeline.session_semantics import (
            compare_session_semantics,
        )
        df_a = _make_bars(["ES"], 30, seed=42)
        df_b = _make_bars(["ES"], 30, seed=42, perturb_bps=50.0)  # 50 bps off
        report = compare_session_semantics(
            df_a, df_b, ["ES"], sample_days=30, seed=42,
            thresholds={"open_close_median_bps_max": 5.0, "open_close_p95_bps_max": 25.0,
                        "frac_days_over_25bps_max": 0.05},
        )
        # Should detect non-zero differences
        assert any(s.get("median_bps", 0) > 0 for s in report.per_field_stats.values())

    def test_empty_overlap_skips(self) -> None:
        from sleeves.cooc_reversal_futures.pipeline.session_semantics import (
            compare_session_semantics,
        )
        df_a = _make_bars(["ES"], 10, seed=42)
        df_b = _make_bars(["NQ"], 10, seed=42)  # Different root
        report = compare_session_semantics(df_a, df_b, ["ES"], sample_days=10, seed=42)
        assert report.gate_passed is True  # No data to compare → vacuously true

    def test_output_artifacts(self) -> None:
        from sleeves.cooc_reversal_futures.pipeline.session_semantics import (
            compare_session_semantics,
        )
        with tempfile.TemporaryDirectory() as td:
            df = _make_bars(["ES"], 20, seed=42)
            report = compare_session_semantics(
                df, df, ["ES"], sample_days=15, seed=42, output_dir=td,
            )
            assert (Path(td) / "session_semantics_report.json").exists()


# ===========================================================================
# Feature Parity Tests
# ===========================================================================

class TestFeatureParity:
    """Tests for parity.compute_feature_parity."""

    def _make_gold(self, roots: list[str], n_days: int = 40) -> pd.DataFrame:
        """Create synthetic gold frame matching pipeline schema."""
        rng = np.random.default_rng(42)
        records = []
        base = date(2025, 6, 1)
        for root in roots:
            for i in range(n_days):
                day = date.fromordinal(base.toordinal() + i)
                ret_co = rng.normal(0, 0.01)
                records.append({
                    "trading_day": day,
                    "root": root,
                    "open": 5000 + rng.uniform(-20, 20),
                    "high": 5020,
                    "low": 4980,
                    "close": 5000 + rng.uniform(-20, 20),
                    "volume": 200000,
                    "ret_co": ret_co,
                    "ret_oc": rng.normal(0, 0.01),
                    "active_contract": f"{root}H26",
                    "days_to_expiry": max(0, 30 - i),
                })
        df = pd.DataFrame(records).sort_values(["root", "trading_day"]).reset_index(drop=True)
        return df

    def test_parity_report_structure(self) -> None:
        from sleeves.cooc_reversal_futures.pipeline.parity import compute_feature_parity

        gold = self._make_gold(["ES", "NQ"], 30)

        # Create a minimal mock sleeve
        sleeve = MagicMock()
        sleeve.compute_signal_frame.return_value = pd.DataFrame()  # Returns empty → no matches

        days = [date(2025, 6, 10), date(2025, 6, 15)]
        report = compute_feature_parity(
            gold_frame=gold, sleeve=sleeve, asof_days=days,
        )
        # Report should be well-formed
        assert hasattr(report, "per_feature_mismatch_rate")
        assert hasattr(report, "gate_passed")
        assert hasattr(report, "worst_offenders")
        assert isinstance(report.per_feature_mismatch_rate, dict)

    def test_unmapped_features_flagged(self) -> None:
        from sleeves.cooc_reversal_futures.pipeline.parity import (
            _TRAIN_ONLY, _RUNTIME_ONLY, compute_feature_parity,
        )
        gold = self._make_gold(["ES"], 20)
        sleeve = MagicMock()
        sleeve.compute_signal_frame.return_value = pd.DataFrame()

        report = compute_feature_parity(
            gold_frame=gold, sleeve=sleeve, asof_days=[date(2025, 6, 5)],
        )
        # Unmapped features should show up in report
        for feat in _TRAIN_ONLY:
            assert f"{feat} (train_only)" in report.per_feature_mismatch_rate
        for feat in _RUNTIME_ONLY:
            assert f"{feat} (runtime_only)" in report.per_feature_mismatch_rate

    def test_artifacts_written(self) -> None:
        from sleeves.cooc_reversal_futures.pipeline.parity import compute_feature_parity

        gold = self._make_gold(["ES"], 20)
        sleeve = MagicMock()
        sleeve.compute_signal_frame.return_value = pd.DataFrame()

        with tempfile.TemporaryDirectory() as td:
            compute_feature_parity(
                gold_frame=gold, sleeve=sleeve,
                asof_days=[date(2025, 6, 5)],
                output_dir=td,
            )
            assert (Path(td) / "feature_parity_report.json").exists()


# ===========================================================================
# Coverage Gate Tests
# ===========================================================================

class TestCoverageGate:
    """Tests for validation._coverage_gate."""

    def _make_dataset(self, roots_per_day: Dict[date, list[str]]) -> pd.DataFrame:
        records = []
        for day, roots in roots_per_day.items():
            for root in roots:
                records.append({
                    "trading_day": day,
                    "root": root,
                    "ret_co": 0.001,
                    "ret_oc": 0.002,
                    "signal": -0.001,
                    "rolling_std_ret_co": 0.01,
                    "rolling_std_ret_oc": 0.01,
                    "rolling_mean_volume": 100000,
                    "roll_window_flag": 0,
                    "days_to_expiry": 20,
                    "ret_co_rank_pct": 0.5,
                    "ret_co_cs_demean": 0.0,
                    "y": 0.001,
                })
        return pd.DataFrame(records)

    def test_full_coverage_passes(self) -> None:
        from sleeves.cooc_reversal_futures.pipeline.validation import _coverage_gate

        ds = self._make_dataset({
            date(2025, 6, 1): ["ES", "NQ", "YM", "RTY"],
            date(2025, 6, 2): ["ES", "NQ", "YM", "RTY"],
        })
        gate, report = _coverage_gate(ds, min_roots_per_day=4)
        assert gate.passed is True
        assert report.gate_passed is True
        assert report.days_below_threshold == 0

    def test_partial_coverage_fails(self) -> None:
        from sleeves.cooc_reversal_futures.pipeline.validation import _coverage_gate

        ds = self._make_dataset({
            date(2025, 6, 1): ["ES", "NQ", "YM", "RTY"],
            date(2025, 6, 2): ["ES", "NQ"],  # Only 2 roots
        })
        gate, report = _coverage_gate(ds, min_roots_per_day=4, allow_partial=False)
        assert gate.passed is False
        assert report.days_below_threshold == 1

    def test_partial_coverage_allowed(self) -> None:
        from sleeves.cooc_reversal_futures.pipeline.validation import _coverage_gate

        ds = self._make_dataset({
            date(2025, 6, 1): ["ES", "NQ", "YM", "RTY"],
            date(2025, 6, 2): ["ES", "NQ"],
        })
        gate, report = _coverage_gate(ds, min_roots_per_day=4, allow_partial=True)
        assert gate.passed is True

    def test_histogram_correct(self) -> None:
        from sleeves.cooc_reversal_futures.pipeline.validation import _coverage_gate

        ds = self._make_dataset({
            date(2025, 6, 1): ["ES", "NQ", "YM", "RTY"],
            date(2025, 6, 2): ["ES", "NQ", "YM"],
            date(2025, 6, 3): ["ES", "NQ", "YM", "RTY"],
        })
        _, report = _coverage_gate(ds, min_roots_per_day=3)
        # 3-root days: 1, 4-root days: 2
        assert report.histogram.get(3, 0) == 1
        assert report.histogram.get(4, 0) == 2

    def test_low_threshold_passes_sparse(self) -> None:
        from sleeves.cooc_reversal_futures.pipeline.validation import _coverage_gate

        ds = self._make_dataset({
            date(2025, 6, 1): ["ES", "NQ"],
            date(2025, 6, 2): ["ES", "NQ", "YM"],
        })
        gate, report = _coverage_gate(ds, min_roots_per_day=2)
        assert gate.passed is True


# ===========================================================================
# Trade Proxy Tests
# ===========================================================================

class TestTradeProxy:
    """Tests for trade_proxy.evaluate_trade_proxy."""

    def _make_dataset(self, n_days: int = 100) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        records = []
        base = date(2025, 1, 2)
        roots = ["ES", "NQ", "YM", "RTY"]
        for i in range(n_days):
            day = date.fromordinal(base.toordinal() + i)
            for root in roots:
                ret_co = rng.normal(0, 0.01)
                ret_oc = rng.normal(0, 0.01)
                records.append({
                    "trading_day": day,
                    "root": root,
                    "ret_co": ret_co,
                    "ret_oc": ret_oc,
                    "signal": -ret_co,
                    "close": 5000.0,
                    "multiplier": 50.0,
                })
        return pd.DataFrame(records)

    def test_baseline_equals_baseline(self) -> None:
        from sleeves.cooc_reversal_futures.pipeline.trade_proxy import evaluate_trade_proxy

        ds = self._make_dataset(80)
        # Under default score_semantics="alpha_low_long":
        #   model_alpha = -preds
        # Baseline uses r_co_meanrevert: baseline_alpha = -r_co = -ret_co
        # So for parity: -preds = -ret_co  →  preds = ret_co
        preds = pd.Series(ds["ret_co"].values, index=ds.index)
        report = evaluate_trade_proxy(ds, preds)

        # Model predictions == baseline predictions → same Sharpe (or very close)
        diff = abs(report.sharpe_model - report.sharpe_baseline)
        assert diff < 0.01

    def test_gate_passes_when_model_beats_baseline(self) -> None:
        from sleeves.cooc_reversal_futures.pipeline.trade_proxy import evaluate_trade_proxy

        ds = self._make_dataset(80)
        preds = pd.Series(ds["ret_co"].values, index=ds.index)  # Same as baseline under alpha_low_long
        report = evaluate_trade_proxy(
            ds, preds,
            config={"require_not_worse_than_baseline": True, "sharpe_tolerance": 0.1},
        )
        assert report.gate_passed is True

    def test_report_metrics_valid(self) -> None:
        from sleeves.cooc_reversal_futures.pipeline.trade_proxy import evaluate_trade_proxy

        ds = self._make_dataset(80)
        preds = pd.Series(-ds["ret_co"].values, index=ds.index)
        report = evaluate_trade_proxy(ds, preds)

        assert 0.0 <= report.hit_rate <= 1.0
        assert report.max_drawdown <= 0.0  # drawdown is negative
        assert isinstance(report.mean_daily_return, float)
        assert isinstance(report.worst_1pct_return, float)

    def test_output_artifacts(self) -> None:
        from sleeves.cooc_reversal_futures.pipeline.trade_proxy import evaluate_trade_proxy

        ds = self._make_dataset(50)
        preds = pd.Series(-ds["ret_co"].values, index=ds.index)
        with tempfile.TemporaryDirectory() as td:
            evaluate_trade_proxy(ds, preds, output_dir=td)
            assert (Path(td) / "trade_proxy_report.json").exists()
            assert (Path(td) / "trade_proxy_daily.parquet").exists()


# ===========================================================================
# IBKR Historical Provider Tests (mock-based)
# ===========================================================================

class TestIBKRHistProvider:
    """Tests for ibkr_hist_provider.IBKRHistoricalDataProvider."""

    def _mock_client(self) -> MagicMock:
        client = MagicMock()
        # Mock qualify_contracts to return a contract with conId
        mock_contract = MagicMock()
        mock_contract.conId = 12345
        client.qualify_contracts.return_value = [mock_contract]

        # Mock historical_bars to return synthetic data
        dates = pd.date_range("2025-06-01", periods=30, freq="B", tz="UTC")
        client.historical_bars.return_value = pd.DataFrame({
            "timestamp": dates,
            "open": np.random.uniform(4990, 5010, len(dates)),
            "high": np.random.uniform(5010, 5030, len(dates)),
            "low": np.random.uniform(4970, 4990, len(dates)),
            "close": np.random.uniform(4990, 5010, len(dates)),
            "volume": np.random.randint(100000, 500000, len(dates)),
        })
        return client

    def test_fetch_returns_correct_schema(self) -> None:
        from sleeves.cooc_reversal_futures.pipeline.ibkr_hist_provider import (
            IBKRHistoricalDataProvider,
        )
        client = self._mock_client()
        provider = IBKRHistoricalDataProvider(client)
        df = provider.fetch_daily_bars("ES", date(2025, 6, 1), date(2025, 7, 15))
        required = {"timestamp", "open", "high", "low", "close", "volume"}
        assert required.issubset(set(df.columns))

    def test_caching_works(self) -> None:
        from sleeves.cooc_reversal_futures.pipeline.ibkr_hist_provider import (
            IBKRHistoricalDataProvider,
        )
        with tempfile.TemporaryDirectory() as td:
            client = self._mock_client()
            provider = IBKRHistoricalDataProvider(client, cache_dir=td)

            # First fetch → download
            df1 = provider.fetch_daily_bars("ES", date(2025, 6, 1), date(2025, 7, 15))
            assert len(df1) > 0

            # Second fetch → cache hit (client not called again)
            call_count = client.historical_bars.call_count
            df2 = provider.fetch_daily_bars("ES", date(2025, 6, 1), date(2025, 7, 15))
            # Should not have made additional client calls
            assert client.historical_bars.call_count == call_count
            assert len(df2) == len(df1)

            # Sidecar JSON exists
            cache_files = list(Path(td).rglob("*.json"))
            assert len(cache_files) > 0

    def test_roll_segmentation(self) -> None:
        from sleeves.cooc_reversal_futures.pipeline.ibkr_hist_provider import (
            IBKRHistoricalDataProvider,
        )
        from sleeves.cooc_reversal_futures.contract_master import CONTRACT_MASTER

        segments = IBKRHistoricalDataProvider._build_roll_segments(
            "ES", date(2025, 6, 1), date(2025, 9, 30), CONTRACT_MASTER["ES"],
        )
        assert len(segments) >= 1
        # Each segment should be (start, end, symbol)
        for s_start, s_end, symbol in segments:
            assert s_start <= s_end
            assert symbol.startswith("ES")


# ===========================================================================
# Signal Mode with Phase 1.5 Gate Tests
# ===========================================================================

class TestSignalModePhase15:
    """Tests for signal_mode.resolve_signal_mode with require_phase15."""

    def _make_pack(self, td: str, phase15_status: str = "PASS") -> str:
        pack = Path(td) / "test_pack"
        (pack / "model").mkdir(parents=True)
        (pack / "model" / "model.pkl").write_bytes(b"fake_model")
        (pack / "validation_report.json").write_text(json.dumps({
            "all_passed": True,
            "gates": [{"name": "model_sanity", "passed": True, "detail": "ok"}],
            "baseline_ic": 0.05,
            "model_ic": 0.06,
        }))
        manifest = {
            "run_id": "test",
            "run_dir": str(pack),
            "seed": 42,
            "start_date": "2025-01-01",
            "end_date": "2025-12-31",
            "config_hash": "abc",
            "phase15_status": phase15_status,
        }
        (pack / "run_manifest.json").write_text(json.dumps(manifest))
        return str(pack)

    def test_phase15_pass_allows_model(self) -> None:
        from sleeves.cooc_reversal_futures.signal_mode import resolve_signal_mode, SignalMode

        with tempfile.TemporaryDirectory() as td:
            pack = self._make_pack(td, "PASS")
            mode = resolve_signal_mode(pack, require_phase15=True)
            assert mode == SignalMode.MODEL

    def test_phase15_fail_forces_heuristic(self) -> None:
        from sleeves.cooc_reversal_futures.signal_mode import resolve_signal_mode, SignalMode

        with tempfile.TemporaryDirectory() as td:
            pack = self._make_pack(td, "FAIL")
            mode = resolve_signal_mode(pack, require_phase15=True)
            assert mode == SignalMode.HEURISTIC

    def test_phase15_missing_forces_heuristic(self) -> None:
        from sleeves.cooc_reversal_futures.signal_mode import resolve_signal_mode, SignalMode

        with tempfile.TemporaryDirectory() as td:
            pack = self._make_pack(td, "PASS")
            # Remove run_manifest.json
            (Path(pack) / "run_manifest.json").unlink()
            mode = resolve_signal_mode(pack, require_phase15=True)
            assert mode == SignalMode.HEURISTIC

    def test_phase15_disabled_allows_model(self) -> None:
        from sleeves.cooc_reversal_futures.signal_mode import resolve_signal_mode, SignalMode

        with tempfile.TemporaryDirectory() as td:
            pack = self._make_pack(td, "FAIL")
            mode = resolve_signal_mode(pack, require_phase15=False)
            assert mode == SignalMode.MODEL


# ===========================================================================
# Type Serialization Tests
# ===========================================================================

class TestPhase15Types:
    """Tests for new Phase 1.5 types in types.py."""

    def test_session_semantics_report_serializable(self) -> None:
        from sleeves.cooc_reversal_futures.pipeline.types import SessionSemanticsReport

        r = SessionSemanticsReport(
            per_field_stats={"ES_open": {"median_bps": 1.5, "p95_bps": 4.0}},
            gate_passed=True,
        )
        d = r.to_dict()
        assert d["gate_passed"] is True
        assert "ES_open" in d["per_field_stats"]

    def test_phase15_report_serializable(self) -> None:
        from sleeves.cooc_reversal_futures.pipeline.types import (
            Phase15Report, CoverageReport, TradeProxyReport,
        )

        p15 = Phase15Report(
            coverage=CoverageReport(
                days_total=100, days_below_threshold=0,
                min_roots_per_day=4, gate_passed=True, histogram={4: 100},
            ),
            trade_proxy=TradeProxyReport(
                sharpe_model=1.2, sharpe_baseline=1.0,
                hit_rate=0.55, max_drawdown=-0.05,
                mean_daily_return=0.001, worst_1pct_return=-0.02,
                gate_passed=True,
            ),
            all_passed=True,
        )
        d = p15.to_dict()
        assert d["all_passed"] is True
        assert "coverage" in d
        assert "trade_proxy" in d

    def test_run_manifest_includes_phase15(self) -> None:
        from sleeves.cooc_reversal_futures.pipeline.types import (
            Phase15Report, RunManifest,
        )

        rm = RunManifest(
            run_id="test", run_dir="/tmp", seed=42,
            start_date="2025-01-01", end_date="2025-12-31",
            config_hash="abc",
            phase15=Phase15Report(all_passed=True),
        )
        d = rm.to_dict()
        assert d["phase15_status"] == "PASS"
        assert "phase15" in d

    def test_run_manifest_without_phase15(self) -> None:
        from sleeves.cooc_reversal_futures.pipeline.types import RunManifest

        rm = RunManifest(
            run_id="test", run_dir="/tmp", seed=42,
            start_date="2025-01-01", end_date="2025-12-31",
            config_hash="abc",
        )
        d = rm.to_dict()
        assert "phase15_status" not in d
        assert "phase15" not in d
