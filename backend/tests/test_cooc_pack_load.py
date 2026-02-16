"""Pack load smoke test for CO→OC reversal futures production pack.

Verifies that a production pack can be loaded, model infers correctly,
and outputs are well-shaped and finite — all without importing
training-only dependencies.
"""
from __future__ import annotations

import json
import os
import pickle
from datetime import date
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _latest_run_dir() -> Path:
    base = Path("data_lake/futures/runs")
    if not base.exists():
        pytest.skip("No run directory found — run Phase 1 first")
    runs = sorted(base.iterdir())
    if not runs:
        pytest.skip("No runs found")
    return runs[-1]


def _pack_dir() -> Path:
    pack = _latest_run_dir() / "production_pack"
    if not pack.exists():
        pytest.skip(f"No production_pack in {pack.parent.name} — run training first")
    return pack


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPackLoad:
    """Production pack load and inference smoke tests."""

    def test_pack_directory_exists(self) -> None:
        pack = _pack_dir()
        assert pack.exists(), f"Pack not found: {pack}"
        assert pack.is_dir()

    def test_run_manifest_valid_json(self) -> None:
        manifest_path = _pack_dir() / "run_manifest.json"
        assert manifest_path.exists()
        data = json.loads(manifest_path.read_text())
        assert "run_id" in data
        assert "seed" in data
        assert "start_date" in data
        assert "end_date" in data
        assert "config_hash" in data

    def test_feature_schema_valid(self) -> None:
        schema_path = _pack_dir() / "feature_schema.json"
        assert schema_path.exists()
        schema = json.loads(schema_path.read_text())
        assert "feature_order" in schema
        assert isinstance(schema["feature_order"], list)
        assert len(schema["feature_order"]) > 0
        assert "nan_fill_values" in schema
        assert "chosen_params" in schema

    def test_splits_valid(self) -> None:
        splits_path = _pack_dir() / "splits.json"
        assert splits_path.exists()
        splits = json.loads(splits_path.read_text())
        assert isinstance(splits, list)
        assert len(splits) > 0
        for split in splits:
            assert "train_start" in split
            assert "train_end" in split
            assert "val_start" in split
            assert "val_end" in split
            assert "embargo_days" in split

    def test_validation_report_exists(self) -> None:
        vr_path = _pack_dir() / "validation_report.json"
        assert vr_path.exists()
        report = json.loads(vr_path.read_text())
        assert "gates" in report
        assert isinstance(report["gates"], list)
        assert "baseline_ic" in report
        assert "model_ic" in report

    def test_model_loads(self) -> None:
        model_dir = _pack_dir() / "model"
        assert model_dir.exists()
        assert (model_dir / "model.pkl").exists()
        assert (model_dir / "preprocessor.pkl").exists()
        assert (model_dir / "model_manifest.json").exists()

        with open(model_dir / "model.pkl", "rb") as f:
            model = pickle.load(f)
        with open(model_dir / "preprocessor.pkl", "rb") as f:
            preprocessor = pickle.load(f)

        # Model should have a predict method
        assert hasattr(model, "predict"), "Model missing predict method"

    def test_model_inference_on_synthetic_slice(self) -> None:
        """Load model and run inference on a small synthetic slice."""
        model_dir = _pack_dir() / "model"
        schema_path = _pack_dir() / "feature_schema.json"

        with open(model_dir / "model.pkl", "rb") as f:
            model = pickle.load(f)
        with open(model_dir / "preprocessor.pkl", "rb") as f:
            preprocessor = pickle.load(f)

        schema = json.loads(schema_path.read_text())
        features = schema["feature_order"]
        nan_fill = schema["nan_fill_values"]

        # Create a small synthetic DataFrame matching feature schema
        n_rows = 5
        np.random.seed(42)
        data: Dict[str, Any] = {}
        for feat in features:
            # Use nan_fill as center, small noise
            center = nan_fill.get(feat, 0.0)
            data[feat] = np.random.normal(center, 0.01, n_rows)

        df = pd.DataFrame(data)

        # Transform through preprocessor
        X = preprocessor.transform(df)
        predictions = model.predict(X)

        assert len(predictions) == n_rows, f"Expected {n_rows} preds, got {len(predictions)}"
        assert np.all(np.isfinite(predictions)), "Non-finite predictions"
        assert predictions.dtype in (np.float32, np.float64), f"Bad dtype: {predictions.dtype}"

    def test_contract_master_snapshot(self) -> None:
        cm_path = _pack_dir() / "contract_master.json"
        assert cm_path.exists()
        cm = json.loads(cm_path.read_text())
        for root in ["ES", "NQ", "RTY", "YM"]:
            assert root in cm, f"Missing root {root}"
            spec = cm[root]
            assert "multiplier" in spec
            assert "tick_size" in spec
            assert spec["multiplier"] > 0

    def test_model_inference_on_real_dataset_slice(self) -> None:
        """Load model and run inference on a slice from the actual dataset."""
        run_dir = _latest_run_dir()
        dataset_path = run_dir / "dataset" / "dataset.parquet"
        if not dataset_path.exists():
            pytest.skip("Dataset not found in run dir")

        model_dir = _pack_dir() / "model"
        with open(model_dir / "model.pkl", "rb") as f:
            model = pickle.load(f)
        with open(model_dir / "preprocessor.pkl", "rb") as f:
            preprocessor = pickle.load(f)

        dataset = pd.read_parquet(dataset_path)
        # Take last 20 rows (most recent data — simulates "today" inference)
        recent = dataset.tail(20)

        X = preprocessor.transform(recent)
        predictions = model.predict(X)

        assert len(predictions) == len(recent)
        assert np.all(np.isfinite(predictions)), "Non-finite predictions on real data"
        # Predictions should be in a reasonable range for returns
        assert np.abs(predictions).max() < 1.0, \
            f"Suspiciously large prediction: {np.abs(predictions).max():.4f}"
