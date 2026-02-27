from __future__ import annotations

import os
from dataclasses import dataclass

from .config import MLPlatformConfig


@dataclass(frozen=True)
class DeviceBinding:
    role: str
    logical_device: str
    cuda_visible_devices: str


class DeviceManager:
    def __init__(self, cfg: MLPlatformConfig | None = None):
        self.cfg = cfg or MLPlatformConfig()

    def binding_for(self, role: str) -> DeviceBinding:
        if role not in {"train", "infer"}:
            raise ValueError(f"Unknown role: {role}")
        if role == "train":
            return DeviceBinding(role, self.cfg.train_device, self.cfg.train_device.split(":")[-1])
        return DeviceBinding(role, self.cfg.infer_device, self.cfg.infer_device.split(":")[-1])

    def pin_process(self, role: str) -> DeviceBinding:
        binding = self.binding_for(role)
        os.environ["CUDA_VISIBLE_DEVICES"] = binding.cuda_visible_devices
        os.environ["ML_ROLE"] = role
        return binding

    def startup_assertions(self) -> dict[str, str]:
        # Keep this portable for CI; deep CUDA checks happen in runtime services.
        return {
            "train_device": self.cfg.train_device,
            "infer_device": self.cfg.infer_device,
            "partition": "asymmetric",
        }
