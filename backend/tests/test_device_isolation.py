"""Test device isolation and DataParallel poisoning.

Validates:
  1. DeviceManager is a true singleton
  2. DEVICE_HEAVY / DEVICE_FAST constants are correct
  3. DataParallel is forbidden (raises RuntimeError)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


class TestDeviceIsolation:
    def setup_method(self):
        """Reset singleton before each test."""
        from backend.app.ml_platform.device_manager import DeviceManager
        DeviceManager.reset_singleton()

    def test_singleton_pattern(self):
        from backend.app.ml_platform.device_manager import DeviceManager
        dm1 = DeviceManager()
        dm2 = DeviceManager()
        assert dm1 is dm2, "DeviceManager is not a singleton"

    def test_device_constants(self):
        from backend.app.ml_platform.device_manager import DeviceManager
        assert DeviceManager.HEAVY == torch.device("cuda:0")
        assert DeviceManager.FAST == torch.device("cuda:1")

    def test_data_parallel_forbidden(self):
        """torch.nn.DataParallel should raise RuntimeError after DeviceManager init."""
        from backend.app.ml_platform.device_manager import DeviceManager
        _ = DeviceManager()

        with pytest.raises(RuntimeError, match="[Ff]orbidden|DataParallel"):
            model = torch.nn.Linear(10, 10)
            torch.nn.DataParallel(model)

    def test_get_device_roles(self):
        from backend.app.ml_platform.device_manager import DeviceManager
        assert DeviceManager.get_device("heavy") == torch.device("cuda:0")
        assert DeviceManager.get_device("train") == torch.device("cuda:0")
        assert DeviceManager.get_device("fast") == torch.device("cuda:1")
        assert DeviceManager.get_device("infer") == torch.device("cuda:1")

    def test_invalid_role_raises(self):
        from backend.app.ml_platform.device_manager import DeviceManager
        with pytest.raises(ValueError, match="Unknown device role"):
            DeviceManager.get_device("invalid")

    def test_startup_assertions(self):
        from backend.app.ml_platform.device_manager import DeviceManager
        dm = DeviceManager()
        info = dm.startup_assertions()
        assert info["partition"] == "asymmetric"
        assert info["dp_poisoned"] == "true"

    def teardown_method(self):
        """Restore DataParallel after tests."""
        # Re-import to get the original
        import importlib
        import torch.nn
        from backend.app.ml_platform.device_manager import DeviceManager
        DeviceManager.reset_singleton()
