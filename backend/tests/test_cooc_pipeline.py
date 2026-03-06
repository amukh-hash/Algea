"""Comprehensive tests for the CO→OC reversal futures pipeline.

Covers:
  1. Bronze validation violation detection
  2. Canonicalization correctness (exact 3-day example)
  3. Roll mapping coverage
  4. Leakage guard per-row
  5. Feature/label alignment exactness
  6. Split chronology + embargo
  7. Baseline vs model IC gate logic
  8. Crash + CAUTION gating integration
  9. Artifact existence + schema
  10. Determinism (same seed → identical outputs)
"""
from __future__ import annotations

import json
import shutil
import tempfile
from datetime import date, datetime, time
from pathlib import Path
from typing import Dict, List
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

# --- Pipeline imports ---
from sleeves.cooc_reversal_futures.config import COOCReversalConfig
from sleeves.cooc_reversal_futures.contract_master import CONTRACT_MASTER
from sleeves.cooc_reversal_futures.pipeline.bronze_validate import validate_bronze_bars
from sleeves.cooc_reversal_futures.pipeline.canonicalize import (
    build_contract_map,
    build_gold_frame,
    build_silver_bars,
    normalize_bars,
)
from sleeves.cooc_reversal_futures.pipeline.dataset import (
    _trading_day_to_decision_ts,
    _trading_day_to_feature_cutoff_ts,
    _trading_day_to_label_ts,
    assert_no_leakage,
    assemble_dataset,
    build_features,
    build_labels,
)
from sleeves.cooc_reversal_futures.pipeline.ingest import CsvDataProvider, ingest_bronze
from sleeves.cooc_reversal_futures.pipeline.splits import time_based_split, walk_forward_cv
from sleeves.cooc_reversal_futures.pipeline.train import (
    FEATURE_COLUMNS,
    Preprocessor,
    _RidgeModel,
    train_model,
    save_model_bundle,
)
from sleeves.cooc_reversal_futures.pipeline.types import (
    BronzeManifest,
    BronzeValidationReport,
    DatasetManifest,
    GateResult,
    ModelBundle,
    RunManifest,
    SplitSpec,
    ValidationReport,
)
from sleeves.cooc_reversal_futures.pipeline.validation import (
    _ic,
    run_validation,
)
from sleeves.cooc_reversal_futures.pipeline.export import export_production_pack
from sleeves.cooc_reversal_futures.roll import active_contract_for_day, roll_week_flag
from sleeves.cooc_reversal_futures.sleeve import COOCReversalFuturesSleeve
from algae.data.options.vrp_features import VolRegime

ET = ZoneInfo("America/New_York")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Fixtures                                                                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def _make_3day_bars(root: str = "ES") -> pd.DataFrame:
    """Synthetic 3-day OHLCV bars with known values for exact testing."""
    dates = pd.bdate_range("2025-01-02", periods=3, tz="UTC")
    return pd.DataFrame({
        "timestamp": dates,
        "open":  [100.0, 101.0, 102.0],
        "high":  [103.0, 104.0, 105.0],
        "low":   [99.0,  100.0, 101.0],
        "close": [102.0, 103.0, 104.0],
        "volume": [1000, 1100, 1200],
    })


def _make_multi_root_bars() -> Dict[str, pd.DataFrame]:
    """Bars for ES and NQ covering 60 business days."""
    np.random.seed(42)
    dates = pd.bdate_range("2024-01-02", periods=60, tz="UTC")
    result = {}
    for root, base in [("ES", 5000.0), ("NQ", 17000.0)]:
        n = len(dates)
        rets = np.random.normal(0, 0.005, n)
        closes = base * np.cumprod(1 + rets)
        opens = closes * (1 + np.random.normal(0, 0.001, n))
        highs = np.maximum(opens, closes) + np.abs(np.random.normal(0, 10, n))
        lows = np.minimum(opens, closes) - np.abs(np.random.normal(0, 10, n))
        vols = np.random.poisson(50000, n)
        result[root] = pd.DataFrame({
            "timestamp": dates,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": vols,
        })
    return result


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def multi_root_bars():
    return _make_multi_root_bars()


@pytest.fixture
def gold_frame():
    """Gold frame from multi-root bars."""
    bars = _make_multi_root_bars()
    all_norm: List[pd.DataFrame] = []
    for root, df in sorted(bars.items()):
        df = df.copy()
        df["root"] = root
        df = normalize_bars(df)
        all_norm.append(df)
    combined = pd.concat(all_norm, ignore_index=True)
    cmap = build_contract_map(sorted(bars.keys()), date(2024, 1, 2), date(2024, 3, 29))
    silver = build_silver_bars(combined, cmap)
    return build_gold_frame(silver)


@pytest.fixture
def full_dataset(gold_frame):
    """Assembled dataset from gold frame."""
    config = COOCReversalConfig()
    ds, manifest = assemble_dataset(gold_frame, config, lookback=10)
    return ds


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  1. Bronze Validation Violation Detection                                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class TestBronzeValidation:
    def test_valid_bars_pass(self) -> None:
        bars = _make_3day_bars()
        report = validate_bronze_bars(bars, "ES")
        assert report.ok
        assert report.monotonic_ts
        assert report.no_duplicates
        assert report.ohlc_sane
        assert report.non_negative_volume

    def test_non_monotonic_detected(self) -> None:
        bars = _make_3day_bars()
        bars.loc[1, "timestamp"] = bars.loc[0, "timestamp"] - pd.Timedelta(hours=1)
        report = validate_bronze_bars(bars, "ES")
        assert not report.ok
        assert not report.monotonic_ts
        assert "monoton" in report.violations[0].lower()

    def test_duplicate_ts_detected(self) -> None:
        bars = _make_3day_bars()
        bars.loc[2, "timestamp"] = bars.loc[1, "timestamp"]
        report = validate_bronze_bars(bars, "ES")
        assert not report.ok
        assert not report.no_duplicates

    def test_ohlc_violation_detected(self) -> None:
        bars = _make_3day_bars()
        bars.loc[0, "low"] = 200.0  # low > high/open/close
        report = validate_bronze_bars(bars, "ES")
        assert not report.ok
        assert not report.ohlc_sane

    def test_negative_volume_detected(self) -> None:
        bars = _make_3day_bars()
        bars.loc[0, "volume"] = -5
        report = validate_bronze_bars(bars, "ES")
        assert not report.ok
        assert not report.non_negative_volume

    def test_gap_report(self) -> None:
        bars = _make_3day_bars()
        # Create a gap by skipping a day (already Mon–Wed → no gap)
        # Manually set dates: Mon, Thu (gap of Tue+Wed)
        bars.loc[0, "timestamp"] = pd.Timestamp("2025-01-06", tz="UTC")  # Mon
        bars.loc[1, "timestamp"] = pd.Timestamp("2025-01-09", tz="UTC")  # Thu
        bars.loc[2, "timestamp"] = pd.Timestamp("2025-01-10", tz="UTC")  # Fri
        report = validate_bronze_bars(bars, "ES")
        assert len(report.gap_report) > 0


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  2. Canonicalization Correctness                                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class TestCanonicalization:
    def test_normalize_bars_adds_trading_day(self) -> None:
        bars = _make_3day_bars()
        bars["root"] = "ES"
        out = normalize_bars(bars)
        assert "trading_day" in out.columns
        assert len(out) == 3

    def test_gold_frame_ret_co_ret_oc_exact(self) -> None:
        """Known 3-day example with exact expected computations."""
        bars = _make_3day_bars()
        bars["root"] = "ES"
        bars = normalize_bars(bars)
        cmap = build_contract_map(["ES"], date(2025, 1, 2), date(2025, 1, 6))
        silver = build_silver_bars(bars, cmap)
        gold = build_gold_frame(silver)

        # Day 2 (index 0 after drop): open=101, prev_close=102
        # ret_co = 101/102 - 1
        # ret_oc = 103/101 - 1
        row = gold[gold["trading_day"] == date(2025, 1, 3)]
        if len(row) > 0:
            assert np.isclose(row.iloc[0]["ret_co"], 101.0 / 102.0 - 1.0, atol=1e-10)
            assert np.isclose(row.iloc[0]["ret_oc"], 103.0 / 101.0 - 1.0, atol=1e-10)

    def test_gold_frame_first_day_dropped(self) -> None:
        """First day per root has no prev_close → dropped."""
        bars = _make_3day_bars()
        bars["root"] = "ES"
        bars = normalize_bars(bars)
        cmap = build_contract_map(["ES"], date(2025, 1, 2), date(2025, 1, 6))
        silver = build_silver_bars(bars, cmap)
        gold = build_gold_frame(silver)
        # 3 bars → 2 gold rows (first day dropped)
        assert len(gold) == 2


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  3. Roll Mapping Coverage                                                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class TestRollCoverage:
    def test_contract_map_covers_all_root_day_pairs(self) -> None:
        roots = ["ES", "NQ"]
        start, end = date(2024, 1, 2), date(2024, 3, 29)
        cmap = build_contract_map(roots, start, end)
        trading_days = pd.bdate_range(start, end)
        expected_rows = len(roots) * len(trading_days)
        assert len(cmap) == expected_rows

    def test_active_contract_format(self) -> None:
        spec = CONTRACT_MASTER["ES"]
        contract = active_contract_for_day("ES", date(2024, 3, 1), spec)
        assert contract.startswith("ES")
        assert len(contract) >= 4

    def test_roll_week_advances_contract(self) -> None:
        spec = CONTRACT_MASTER["ES"]
        normal = active_contract_for_day("ES", date(2024, 3, 7), spec)
        roll = active_contract_for_day("ES", date(2024, 3, 11), spec)
        assert normal != roll  # during roll week, contract advances


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  4. Leakage Guard Per-Row                                                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class TestLeakageGuard:
    def test_assembled_dataset_passes_leakage_check(self, full_dataset) -> None:
        """Every row must pass: feature_cutoff_ts <= decision_ts and label_ts > decision_ts."""
        assert_no_leakage(full_dataset)

    def test_feature_cutoff_before_decision(self, full_dataset) -> None:
        for _, row in full_dataset.iterrows():
            assert row["feature_cutoff_ts"] <= row["decision_ts"]

    def test_label_after_decision(self, full_dataset) -> None:
        for _, row in full_dataset.iterrows():
            assert row["label_ts"] > row["decision_ts"]

    def test_violated_leakage_raises(self) -> None:
        """Manually craft a leakage violation → must raise."""
        df = pd.DataFrame({
            "feature_cutoff_ts": [datetime(2024, 1, 2, 10, 0, tzinfo=ET)],
            "decision_ts": [datetime(2024, 1, 2, 9, 30, tzinfo=ET)],
            "label_ts": [datetime(2024, 1, 2, 16, 0, tzinfo=ET)],
        })
        with pytest.raises(AssertionError, match="LEAKAGE"):
            assert_no_leakage(df)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  5. Feature/Label Alignment                                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class TestFeatureLabelAlignment:
    def test_signal_equals_neg_ret_co(self, gold_frame) -> None:
        features = build_features(gold_frame, lookback=5)
        assert np.allclose(features["signal"].values, -features["ret_co"].values, equal_nan=True)

    def test_label_equals_neg_ret_oc(self, gold_frame) -> None:
        labels = build_labels(gold_frame)
        r_oc_col = "r_oc" if "r_oc" in gold_frame.columns else "ret_oc"
        assert np.allclose(labels["y"].values, -gold_frame[r_oc_col].values, equal_nan=True)

    def test_all_feature_columns_present(self, full_dataset) -> None:
        for col in FEATURE_COLUMNS:
            assert col in full_dataset.columns, f"Missing feature: {col}"

    def test_provenance_columns_present(self, full_dataset) -> None:
        for col in ("asof_ts", "feature_cutoff_ts", "decision_ts",
                     "label_ts", "data_version_hash", "code_version_hash", "config_hash"):
            assert col in full_dataset.columns, f"Missing provenance: {col}"


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  6. Split Chronology + Embargo                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class TestSplits:
    def test_time_based_split_chronological(self, full_dataset) -> None:
        split = time_based_split(full_dataset, train_frac=0.6, val_frac=0.2, embargo_days=2)
        assert split.train_end < split.val_start
        if split.test_start is not None:
            assert split.val_end < split.test_start

    def test_embargo_enforced(self, full_dataset) -> None:
        split = time_based_split(full_dataset, embargo_days=3)
        train_end = date.fromisoformat(split.train_end)
        val_start = date.fromisoformat(split.val_start)
        gap = (val_start - train_end).days
        assert gap >= 3

    def test_walk_forward_no_overlap(self, full_dataset) -> None:
        folds = walk_forward_cv(full_dataset, fold_size_days=10, embargo_days=2, min_train_days=20)
        for i in range(len(folds) - 1):
            assert folds[i].val_end <= folds[i + 1].val_start

    def test_walk_forward_expanding(self, full_dataset) -> None:
        folds = walk_forward_cv(full_dataset, fold_size_days=10, embargo_days=2, min_train_days=20)
        if len(folds) >= 2:
            # Training windows should expand
            d0_start = date.fromisoformat(folds[0].train_start)
            d0_end = date.fromisoformat(folds[0].train_end)
            d1_end = date.fromisoformat(folds[1].train_end)
            assert d1_end > d0_end
            assert d0_start == date.fromisoformat(folds[1].train_start)  # same start


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  7. Baseline vs Model IC Gate                                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class TestBaselineGate:
    def test_ic_computation(self) -> None:
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert np.isclose(_ic(y, y), 1.0)
        assert np.isclose(_ic(y, -y), -1.0)

    def test_ic_uncorrelated(self) -> None:
        np.random.seed(42)
        y = np.random.randn(100)
        x = np.random.randn(100)
        ic = _ic(y, x)
        assert abs(ic) < 0.3

    def test_model_training_produces_bundle(self, full_dataset) -> None:
        folds = walk_forward_cv(full_dataset, fold_size_days=10, embargo_days=1, min_train_days=20)
        if len(folds) >= 1:
            config = COOCReversalConfig()
            bundle_info, model, pp = train_model(config, full_dataset, folds, seed=42)
            assert bundle_info.chosen_params is not None
            assert len(bundle_info.trial_log) > 0


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  8. Crash + CAUTION Gating Integration                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class TestCrashCautionGating:
    def test_crash_risk_zero_exposure(self) -> None:
        sleeve = COOCReversalFuturesSleeve()
        r = sleeve.build_daily_orders(
            date_t=date(2025, 1, 2),
            pred_mu={"ES": 0.002, "NQ": -0.001},
            pred_sigma={"ES": 0.01, "NQ": 0.01},
            prices={"ES": 5000, "NQ": 18000},
            capital=1_000_000,
            regime=VolRegime.CRASH_RISK,
        )
        assert all(v == 0 for v in r["contracts"].values())
        assert len(r["orders"]) == 0

    def test_caution_scales_gross(self) -> None:
        sleeve = COOCReversalFuturesSleeve()
        r_normal = sleeve.build_daily_orders(
            date_t=date(2025, 1, 2),
            pred_mu={"ES": 0.003, "NQ": -0.003},
            pred_sigma={"ES": 0.01, "NQ": 0.01},
            prices={"ES": 5000, "NQ": 18000},
            capital=5_000_000,
            regime=VolRegime.NORMAL_CARRY,
        )
        r_caution = sleeve.build_daily_orders(
            date_t=date(2025, 1, 2),
            pred_mu={"ES": 0.003, "NQ": -0.003},
            pred_sigma={"ES": 0.01, "NQ": 0.01},
            prices={"ES": 5000, "NQ": 18000},
            capital=5_000_000,
            regime=VolRegime.CAUTION,
        )
        normal_gross = sum(abs(v) for v in r_normal["weights"].values())
        caution_gross = sum(abs(v) for v in r_caution["weights"].values())
        assert caution_gross <= normal_gross + 1e-6

    def test_validation_strategy_gates_pass(self) -> None:
        """Run strategy gates via validation module."""
        from sleeves.cooc_reversal_futures.pipeline.validation import (
            _strategy_gate_caps,
            _strategy_gate_caution,
            _strategy_gate_crash,
        )
        config = COOCReversalConfig()
        assert _strategy_gate_crash(config).passed
        assert _strategy_gate_caution(config).passed
        assert _strategy_gate_caps(config).passed


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  9. Artifact Existence + Schema                                          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class TestArtifactOutputs:
    def test_bronze_ingest_creates_artifacts(self, tmp_dir, multi_root_bars) -> None:
        # Write CSVs
        data_dir = tmp_dir / "raw"
        data_dir.mkdir()
        for root, df in multi_root_bars.items():
            df.to_csv(data_dir / f"{root}.csv", index=False)

        provider = CsvDataProvider(data_dir)
        bronze_dir = tmp_dir / "bronze"
        manifest = ingest_bronze(provider, list(multi_root_bars.keys()),
                                  date(2024, 1, 2), date(2024, 3, 29), bronze_dir)
        for root in multi_root_bars:
            assert Path(manifest.paths[root]).exists()
            meta_path = Path(manifest.paths[root]).parent / "_meta.json"
            assert meta_path.exists()
            meta = json.loads(meta_path.read_text())
            assert "sha256" in meta

    def test_types_to_dict_serializable(self) -> None:
        """All pipeline types must produce JSON-serializable dicts."""
        report = BronzeValidationReport(
            root="ES", ok=True, monotonic_ts=True, no_duplicates=True,
            ohlc_sane=True, non_negative_volume=True,
            gap_report=(), row_count=100,
        )
        assert json.dumps(report.to_dict())

        split = SplitSpec(
            train_start="2024-01-01", train_end="2024-06-30",
            val_start="2024-07-03", val_end="2024-09-30",
            test_start="2024-10-03", test_end="2024-12-31",
            embargo_days=2,
        )
        assert json.dumps(split.to_dict())

        gate = GateResult(name="test", passed=True, detail="ok")
        assert json.dumps(gate.to_dict())

    def test_export_production_pack(self, tmp_dir, full_dataset) -> None:
        """End-to-end: train → save → export pack."""
        config = COOCReversalConfig()
        folds = walk_forward_cv(full_dataset, fold_size_days=10, embargo_days=1, min_train_days=20)
        if not folds:
            pytest.skip("Not enough data for folds")
        bundle_info, model, pp = train_model(config, full_dataset, folds, seed=42)
        model_dir = tmp_dir / "model"
        bundle = save_model_bundle(bundle_info, model, pp, model_dir)

        manifest = RunManifest(
            run_id="test_run",
            run_dir=str(tmp_dir),
            seed=42,
            start_date="2024-01-02",
            end_date="2024-03-29",
            config_hash="abc123",
            model=bundle,
            splits=tuple(folds),
        )
        pack_dir = export_production_pack(manifest, tmp_dir)
        assert (pack_dir / "run_manifest.json").exists()
        assert (pack_dir / "feature_schema.json").exists()
        assert (pack_dir / "splits.json").exists()


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  10. Determinism                                                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class TestDeterminism:
    def test_same_seed_identical_model(self, full_dataset) -> None:
        config = COOCReversalConfig()
        folds = walk_forward_cv(full_dataset, fold_size_days=10, embargo_days=1, min_train_days=20)
        if not folds:
            pytest.skip("Not enough data for folds")

        b1, m1, p1 = train_model(config, full_dataset, folds, seed=42)
        b2, m2, p2 = train_model(config, full_dataset, folds, seed=42)

        assert b1.chosen_params == b2.chosen_params
        assert np.isclose(b1.primary_metric_value, b2.primary_metric_value)

    def test_dataset_hash_stable(self, gold_frame) -> None:
        config = COOCReversalConfig()
        ds1, m1 = assemble_dataset(gold_frame, config, lookback=10)
        ds2, m2 = assemble_dataset(gold_frame, config, lookback=10)
        assert m1.data_version_hash == m2.data_version_hash

    def test_gold_frame_deterministic(self) -> None:
        bars = _make_multi_root_bars()
        all_norm: List[pd.DataFrame] = []
        for root, df in sorted(bars.items()):
            df2 = df.copy()
            df2["root"] = root
            df2 = normalize_bars(df2)
            all_norm.append(df2)
        combined = pd.concat(all_norm, ignore_index=True)
        cmap = build_contract_map(sorted(bars.keys()), date(2024, 1, 2), date(2024, 3, 29))
        silver = build_silver_bars(combined, cmap)
        gold = build_gold_frame(silver)

        # Do it again
        all_norm2: List[pd.DataFrame] = []
        for root, df in sorted(bars.items()):
            df2 = df.copy()
            df2["root"] = root
            df2 = normalize_bars(df2)
            all_norm2.append(df2)
        combined2 = pd.concat(all_norm2, ignore_index=True)
        silver2 = build_silver_bars(combined2, cmap)
        gold2 = build_gold_frame(silver2)

        pd.testing.assert_frame_equal(gold, gold2)
