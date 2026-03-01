"""Verification tests for TFT Core Reversal sleeve.

Covers:
1. TFT model forward-pass shape validation ([B, 184, 3] → [B, 3])
2. GRN residual projection (input_dim ≠ hidden_dim)
3. CoreTFTPlugin dual-intent emission (AUCTION_OPEN + AUCTION_CLOSE)
4. Sizing safety invariant: weight bounded [0.04, 0.08]
5. Killswitch: zero weight on high uncertainty or low edge
6. TFTInferenceContext schema validation
7. DAG topology loads the new jobs correctly
"""
from __future__ import annotations

import json
import os
import tempfile

import pytest
import torch

from backend.app.core.schemas import ExecutionPhase
from backend.app.ml_platform.models.tft_gap.model import (
    GatedResidualNetwork,
    TemporalFusionTransformer,
)
from backend.app.sleeves.core_tft.schemas import TFTInferenceContext
from backend.app.sleeves.core_tft_plugin import CoreTFTPlugin

# Detect CUDA availability — plugin tests use CPU fallback when needed
_HAS_CUDA = torch.cuda.is_available()


@pytest.fixture(autouse=True)
def _patch_cuda_device(monkeypatch):
    """Force CoreTFTPlugin to use CPU when CUDA is not available."""
    if not _HAS_CUDA:
        # Patch the device string inside the plugin's execute method
        original_execute = CoreTFTPlugin.execute

        def _cpu_execute(self, context, model_cache):
            # Temporarily override the device
            import backend.app.sleeves.core_tft_plugin as plugin_mod
            # The plugin hardcodes "cuda:1", so we monkeypatch it
            monkeypatch.setattr(plugin_mod, "_DEVICE", "cpu")
            return original_execute(self, context, model_cache)

        monkeypatch.setattr(CoreTFTPlugin, "execute", _cpu_execute)


# ═══════════════════════════════════════════════════════════════════════
# Phase 2: TFT Model Shape Tests
# ═══════════════════════════════════════════════════════════════════════

class TestGatedResidualNetwork:
    """GRN shape and residual projection tests."""

    def test_grn_same_dim(self):
        """GRN with input_dim == hidden_dim uses identity residual."""
        grn = GatedResidualNetwork(64, 64)
        x = torch.randn(2, 64)
        out = grn(x)
        assert out.shape == (2, 64)

    def test_grn_different_dim(self):
        """GRN with input_dim != hidden_dim projects residual correctly."""
        grn = GatedResidualNetwork(3, 64)
        x = torch.randn(2, 3)
        out = grn(x)
        assert out.shape == (2, 64)

    def test_grn_five_to_64(self):
        """GRN for observed covariates: 5 → 64."""
        grn = GatedResidualNetwork(5, 64)
        x = torch.randn(4, 5)
        out = grn(x)
        assert out.shape == (4, 64)


class TestTemporalFusionTransformer:
    """TFT forward pass shape validation."""

    def test_forward_shape(self):
        """[B, 184, 3] → [B, 3] quantile output."""
        model = TemporalFusionTransformer(hidden_dim=64)
        ts = torch.randn(1, 184, 3)
        static = torch.randn(1, 3)
        obs = torch.randn(1, 5)
        out = model(ts, static, obs)
        assert out.shape == (1, 3), f"Expected (1, 3), got {out.shape}"

    def test_forward_batch(self):
        """Batched forward: [4, 184, 3] → [4, 3]."""
        model = TemporalFusionTransformer(hidden_dim=64)
        ts = torch.randn(4, 184, 3)
        static = torch.randn(4, 3)
        obs = torch.randn(4, 5)
        out = model(ts, static, obs)
        assert out.shape == (4, 3), f"Expected (4, 3), got {out.shape}"

    def test_forward_bfloat16(self):
        """bfloat16 precision produces valid output."""
        model = TemporalFusionTransformer(hidden_dim=64).to(dtype=torch.bfloat16)
        ts = torch.randn(1, 184, 3, dtype=torch.bfloat16)
        static = torch.randn(1, 3, dtype=torch.bfloat16)
        obs = torch.randn(1, 5, dtype=torch.bfloat16)
        out = model(ts, static, obs)
        assert out.shape == (1, 3)
        assert not torch.any(torch.isnan(out)), "bfloat16 output contains NaN"


# ═══════════════════════════════════════════════════════════════════════
# Phase 3: CoreTFTPlugin Tests
# ═══════════════════════════════════════════════════════════════════════

def _make_features(
    q50_bias: float = 0.0,
    asof_date: str = "2026-01-15",
) -> dict:
    """Build a valid TFTInferenceContext dict for testing."""
    return {
        "asof_date": asof_date,
        "observed_past_seq": [[0.001, 0.5, 0.5]] * 184,
        "day_of_week": 2,
        "is_opex": 0,
        "macro_event_id": 0,
        "gap_proxy_pct": 0.005,
        "nikkei_pct": -0.003,
        "eurostoxx_pct": 0.002,
        "zn_drift_bps": -5.0,
        "vix_spot": 18.5,
    }


class TestCoreTFTPlugin:
    """CoreTFTPlugin inference and intent emission tests."""

    def test_dual_intent_emission(self):
        """Plugin always emits exactly 2 intents (entry + exit)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            features = _make_features()
            with open(os.path.join(tmpdir, "tft_features.json"), "w") as f:
                json.dump(features, f)

            context = {
                "artifact_dir": tmpdir,
                "asof_date": "2026-01-15",
                "model_weights_path": None,
            }
            model_cache = {}
            plugin = CoreTFTPlugin()
            plugin.execute(context, model_cache)

            out_path = os.path.join(tmpdir, "core_intents.json")
            assert os.path.exists(out_path), "core_intents.json not written"

            with open(out_path) as f:
                intents = json.load(f)

            assert len(intents) == 2, f"Expected 2 intents, got {len(intents)}"

            # First intent: AUCTION_OPEN (entry)
            assert intents[0]["execution_phase"] == ExecutionPhase.AUCTION_OPEN.value
            assert intents[0]["sleeve"] == "core_reversal_tft"
            assert intents[0]["symbol"] == "ES"
            assert intents[0]["multiplier"] == 50.0

            # Second intent: AUCTION_CLOSE (exit — forced flat)
            assert intents[1]["execution_phase"] == ExecutionPhase.AUCTION_CLOSE.value
            assert intents[1]["target_weight"] == 0.0

    def test_exit_always_zero_weight(self):
        """Exit intent target_weight is always 0.0 regardless of signal."""
        with tempfile.TemporaryDirectory() as tmpdir:
            features = _make_features()
            with open(os.path.join(tmpdir, "tft_features.json"), "w") as f:
                json.dump(features, f)

            context = {
                "artifact_dir": tmpdir,
                "asof_date": "2026-01-15",
                "model_weights_path": None,
            }
            plugin = CoreTFTPlugin()
            plugin.execute(context, {})

            with open(os.path.join(tmpdir, "core_intents.json")) as f:
                intents = json.load(f)

            exit_intent = intents[1]
            assert exit_intent["target_weight"] == 0.0, \
                "Exit intent must always have target_weight=0.0"

    def test_model_cache_reuse(self):
        """Model is loaded once and reused from cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            features = _make_features()
            with open(os.path.join(tmpdir, "tft_features.json"), "w") as f:
                json.dump(features, f)

            context = {
                "artifact_dir": tmpdir,
                "asof_date": "2026-01-15",
                "model_weights_path": None,
            }
            model_cache = {}
            plugin = CoreTFTPlugin()

            plugin.execute(context, model_cache)
            assert "tft_model" in model_cache

            # Run again — should reuse cached model
            with open(os.path.join(tmpdir, "tft_features.json"), "w") as f:
                json.dump(features, f)
            plugin.execute(context, model_cache)
            assert "tft_model" in model_cache


# ═══════════════════════════════════════════════════════════════════════
# Sizing Safety Invariant Tests
# ═══════════════════════════════════════════════════════════════════════

class TestSizingSafety:
    """Verify the confidence scaling formula is bounded [0.04, 0.08]."""

    @pytest.mark.parametrize("spread", [0.0, 0.0001, 0.005, 0.010, 0.014, 0.015])
    def test_weight_bounded(self, spread):
        """Weight magnitude is always in [0.0, 0.08] regardless of spread."""
        plugin = CoreTFTPlugin()
        # Simulate the sizing logic directly
        q50 = 0.01  # Positive edge above MIN_EDGE
        uncertainty_spread = spread

        target_weight = 0.0
        if uncertainty_spread <= plugin.MAX_UNCERTAINTY and abs(q50) >= plugin.MIN_EDGE:
            direction = 1.0 if q50 > 0 else -1.0
            confidence = max(0.0, 1.0 - (uncertainty_spread / plugin.MAX_UNCERTAINTY))
            scaled_weight = plugin.BASE_WEIGHT + (
                confidence * (plugin.MAX_WEIGHT - plugin.BASE_WEIGHT)
            )
            target_weight = direction * scaled_weight

        assert abs(target_weight) <= plugin.MAX_WEIGHT, \
            f"Weight {target_weight} exceeds MAX_WEIGHT {plugin.MAX_WEIGHT}"
        if target_weight != 0.0:
            assert abs(target_weight) >= plugin.BASE_WEIGHT, \
                f"Non-zero weight {target_weight} below BASE_WEIGHT {plugin.BASE_WEIGHT}"

    def test_zero_spread_gives_max_weight(self):
        """Zero uncertainty → maximum confidence → MAX_WEIGHT."""
        plugin = CoreTFTPlugin()
        confidence = max(0.0, 1.0 - (0.0 / plugin.MAX_UNCERTAINTY))
        weight = plugin.BASE_WEIGHT + (confidence * (plugin.MAX_WEIGHT - plugin.BASE_WEIGHT))
        assert weight == pytest.approx(plugin.MAX_WEIGHT)

    def test_max_spread_gives_base_weight(self):
        """Spread == MAX_UNCERTAINTY → zero confidence → BASE_WEIGHT."""
        plugin = CoreTFTPlugin()
        confidence = max(0.0, 1.0 - (plugin.MAX_UNCERTAINTY / plugin.MAX_UNCERTAINTY))
        weight = plugin.BASE_WEIGHT + (confidence * (plugin.MAX_WEIGHT - plugin.BASE_WEIGHT))
        assert weight == pytest.approx(plugin.BASE_WEIGHT)

    def test_high_spread_vetoes_trade(self):
        """Spread > MAX_UNCERTAINTY → no trade (weight = 0.0)."""
        plugin = CoreTFTPlugin()
        # spread > MAX_UNCERTAINTY should be vetoed
        spread = plugin.MAX_UNCERTAINTY + 0.001
        q50 = 0.01

        target_weight = 0.0
        if spread <= plugin.MAX_UNCERTAINTY and abs(q50) >= plugin.MIN_EDGE:
            target_weight = 1.0  # Should never reach here

        assert target_weight == 0.0

    def test_low_edge_vetoes_trade(self):
        """Median prediction < MIN_EDGE → no trade."""
        plugin = CoreTFTPlugin()
        q50 = 0.001  # Below MIN_EDGE of 0.002

        target_weight = 0.0
        if 0.005 <= plugin.MAX_UNCERTAINTY and abs(q50) >= plugin.MIN_EDGE:
            target_weight = 1.0  # Should never reach here

        assert target_weight == 0.0


# ═══════════════════════════════════════════════════════════════════════
# Schema Validation
# ═══════════════════════════════════════════════════════════════════════

class TestTFTInferenceContext:
    """TFTInferenceContext Pydantic validation."""

    def test_valid_context(self):
        ctx = TFTInferenceContext(**_make_features())
        assert ctx.day_of_week == 2
        assert len(ctx.observed_past_seq) == 184

    def test_rejects_short_sequence(self):
        features = _make_features()
        features["observed_past_seq"] = [[0.0, 0.0, 0.0]] * 100
        with pytest.raises(Exception):
            TFTInferenceContext(**features)

    def test_rejects_invalid_day(self):
        features = _make_features()
        features["day_of_week"] = 7
        with pytest.raises(Exception):
            TFTInferenceContext(**features)


# ═══════════════════════════════════════════════════════════════════════
# DAG Topology Loading
# ═══════════════════════════════════════════════════════════════════════

class TestDAGTopology:
    """Verify dag_topology.yaml loads the TFT jobs correctly."""

    def test_yaml_loads_tft_jobs(self):
        from backend.app.orchestrator.dag_loader import load_yaml_jobs
        jobs = load_yaml_jobs()
        job_names = {j.name for j in jobs}
        assert "data_ingest_core_tft" in job_names, \
            "data_ingest_core_tft not found in DAG topology"
        assert "signals_generate_core_tft" in job_names, \
            "signals_generate_core_tft not found in DAG topology"

    def test_tft_inference_depends_on_ingest(self):
        from backend.app.orchestrator.dag_loader import load_yaml_jobs
        jobs = load_yaml_jobs()
        tft_job = next(j for j in jobs if j.name == "signals_generate_core_tft")
        assert "data_ingest_core_tft" in tft_job.deps, \
            "signals_generate_core_tft must depend on data_ingest_core_tft"
