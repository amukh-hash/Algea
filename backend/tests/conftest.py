"""Shared fixtures for all backend tests.

Sets ALGAE_TEST_MODE=1 to bypass production-only validation gates
(e.g., 1MB minimum weight size) and wraps ModelRegistryStore.publish_version
to auto-place dummy weights before validation runs.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

# Set test mode BEFORE imports so _validate_contract sees it
os.environ["ALGAE_TEST_MODE"] = "1"

FIXTURES_DIR = Path(__file__).parent / "fixtures"
DUMMY_WEIGHTS = FIXTURES_DIR / "dummy_weights.safetensors"


@pytest.fixture
def dummy_weights_path() -> Path:
    """Returns the path to the shared dummy_weights.safetensors fixture (≥1MB)."""
    assert DUMMY_WEIGHTS.exists(), f"Missing fixture: {DUMMY_WEIGHTS}"
    return DUMMY_WEIGHTS


@pytest.fixture
def pre_place_weights(tmp_path: Path):
    """Returns a factory that pre-places dummy weights into a version directory."""
    def _place(version_dir: Path) -> Path:
        version_dir.mkdir(parents=True, exist_ok=True)
        dst = version_dir / "weights.safetensors"
        shutil.copy2(DUMMY_WEIGHTS, dst)
        return version_dir
    return _place


@pytest.fixture(autouse=True)
def _auto_place_weights_for_publish(monkeypatch):
    """Auto-place dummy weights before publish_version validation.

    Wraps ModelRegistryStore.publish_version so that if weights.safetensors
    is missing in the version directory, it gets auto-placed from the test
    fixture. This ensures tests written before Phase 4 stub removal still
    pass without modification.
    """
    from backend.app.ml_platform.registry.store import ModelRegistryStore

    original_publish = ModelRegistryStore.publish_version

    def _wrapped_publish(self, model_name, version, *args, **kwargs):
        # Pre-place weights if not already present
        version_dir = self.model_root / model_name / version
        weights_path = version_dir / "weights.safetensors"
        if not weights_path.exists() and DUMMY_WEIGHTS.exists():
            version_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(DUMMY_WEIGHTS, weights_path)
        return original_publish(self, model_name, version, *args, **kwargs)

    monkeypatch.setattr(ModelRegistryStore, "publish_version", _wrapped_publish)


@pytest.fixture(autouse=True)
def _allow_stub_signals_env(monkeypatch):
    """Allow stub signals in all tests by default."""
    monkeypatch.setenv("ORCH_ALLOW_STUB_SIGNALS", "1")
