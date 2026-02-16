"""Tests for paper-fill shadow evaluator (Deliverable E)."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _create_fills_dir(base: Path, n_days: int = 3) -> Path:
    """Create synthetic fill parquets."""
    fills_dir = base / "reconcile"
    fills_dir.mkdir(parents=True, exist_ok=True)
    for d in range(n_days):
        day = f"2025-01-{10 + d:02d}"
        df = pd.DataFrame({
            "fill_date": [day] * 4,
            "instrument": ["ES", "NQ", "YM", "RTY"],
            "fill_price": [5000.0, 18000.0, 40000.0, 2200.0],
            "fill_qty": [1, 1, 1, 1],
            "pnl": np.random.randn(4) * 100,
        })
        df.to_parquet(fills_dir / f"fills_{day}.parquet", index=False)
    return fills_dir


def _create_intents_dir(base: Path, n_days: int = 3) -> Path:
    """Create synthetic intent JSONs."""
    intents_dir = base / "open"
    intents_dir.mkdir(parents=True, exist_ok=True)
    for d in range(n_days):
        day = f"2025-01-{10 + d:02d}"
        intents = [
            {"date": day, "instrument": "ES", "side": "BUY", "qty": 1},
            {"date": day, "instrument": "NQ", "side": "SELL", "qty": 1},
        ]
        (intents_dir / f"intents_{day}.json").write_text(json.dumps(intents))
    return intents_dir


class TestShadowEval:
    def test_outputs_created(self):
        """Shadow eval should create report JSON and parquets."""
        from backend.scripts.paper.run_shadow_eval_cooc_ibkr import run_shadow_eval

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            fills_dir = _create_fills_dir(base)
            intents_dir = _create_intents_dir(base)
            pack_dir = base / "pack"
            pack_dir.mkdir()
            output_dir = base / "output"

            report = run_shadow_eval(
                fills_dir=str(fills_dir),
                intents_dir=str(intents_dir),
                pack_dir=str(pack_dir),
                output_dir=str(output_dir),
            )

            assert report["status"] == "OK"
            assert (output_dir / "shadow_eval_report.json").exists()
            assert (output_dir / "shadow_fills.parquet").exists()

    def test_no_fills_graceful(self):
        """Should handle empty fills directory gracefully."""
        from backend.scripts.paper.run_shadow_eval_cooc_ibkr import run_shadow_eval

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            fills_dir = base / "reconcile"
            fills_dir.mkdir()
            intents_dir = base / "open"
            intents_dir.mkdir()
            pack_dir = base / "pack"
            pack_dir.mkdir()
            output_dir = base / "output"

            report = run_shadow_eval(
                fills_dir=str(fills_dir),
                intents_dir=str(intents_dir),
                pack_dir=str(pack_dir),
                output_dir=str(output_dir),
            )

            assert report["status"] == "NO_DATA"

    def test_metrics_computed(self):
        """HEURISTIC metrics should be computed from fills."""
        from backend.scripts.paper.run_shadow_eval_cooc_ibkr import run_shadow_eval

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            fills_dir = _create_fills_dir(base)
            intents_dir = _create_intents_dir(base)
            pack_dir = base / "pack"
            pack_dir.mkdir()
            output_dir = base / "output"

            report = run_shadow_eval(
                fills_dir=str(fills_dir),
                intents_dir=str(intents_dir),
                pack_dir=str(pack_dir),
                output_dir=str(output_dir),
            )

            h = report["heuristic"]
            assert "sharpe" in h
            assert "hit_rate" in h
            assert h["n_fills"] > 0

    def test_date_range_filter(self):
        """Date range filter should limit fills."""
        from backend.scripts.paper.run_shadow_eval_cooc_ibkr import run_shadow_eval

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            fills_dir = _create_fills_dir(base, n_days=10)
            intents_dir = _create_intents_dir(base)
            pack_dir = base / "pack"
            pack_dir.mkdir()
            output_dir = base / "output"

            report = run_shadow_eval(
                fills_dir=str(fills_dir),
                intents_dir=str(intents_dir),
                pack_dir=str(pack_dir),
                output_dir=str(output_dir),
                start_date="2025-01-12",
                end_date="2025-01-14",
            )

            assert report["status"] == "OK"
            # Should have fewer fill records due to filter
            assert report["n_fill_records"] > 0
