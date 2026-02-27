from datetime import datetime

from backend.app.ml_platform.config import MLPlatformConfig
from backend.app.ml_platform.device_manager import DeviceManager
from backend.app.ml_platform.inference_gateway.protocol import InferenceRequestBase


def test_protocol_serialization() -> None:
    req = InferenceRequestBase(
        asof=datetime(2026, 1, 1),
        universe_id="us_lg",
        features_hash="abc",
        trace_id="t1",
        payload={"x": 1},
    )
    data = req.to_dict()
    assert data["asof"].startswith("2026-01-01")
    assert data["universe_id"] == "us_lg"


def test_device_pinning_config() -> None:
    cfg = MLPlatformConfig(train_device="cuda:0", infer_device="cuda:1")
    dm = DeviceManager(cfg)
    train_binding = dm.binding_for("train")
    infer_binding = dm.binding_for("infer")
    assert train_binding.cuda_visible_devices == "0"
    assert infer_binding.cuda_visible_devices == "1"
