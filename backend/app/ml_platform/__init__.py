"""ML platform package for training/inference plane separation."""

from .config import MLPlatformConfig
from .device_manager import DeviceManager

__all__ = ["MLPlatformConfig", "DeviceManager"]
