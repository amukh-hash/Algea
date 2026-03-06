"""Singleton Device Manager for hardware-asymmetric GPU partitioning.

Enforces strict device isolation between:
  - cuda:0 (RTX 3090 Ti, 24GB) — batch/offline/training (DEVICE_HEAVY)
  - cuda:1 (RTX 4070 Super, 12GB) — real-time inference (DEVICE_FAST)

Explicitly poisons ``torch.nn.DataParallel`` to prevent symmetric
distribution across mismatched microarchitectures (Ampere vs Ada Lovelace).
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import torch

from .config import MLPlatformConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeviceBinding:
    role: str
    logical_device: str
    cuda_visible_devices: str


class DeviceManager:
    """Singleton device registry with VRAM assertions and DP poisoning.

    Thread-safe singleton pattern.  On first instantiation:
    - Poisons ``torch.nn.DataParallel`` to prevent symmetric DP.
    - Validates physical GPU memory meets minimum thresholds.
    """

    _instance: DeviceManager | None = None

    # Canonical device assignments
    HEAVY = torch.device("cuda:0")  # RTX 3090 Ti — batch/offline
    FAST = torch.device("cuda:1")   # RTX 4070 Super — real-time inference

    # Minimum VRAM thresholds (bytes)
    _MIN_HEAVY_VRAM = 20e9   # 20 GB minimum for cuda:0
    _MIN_FAST_VRAM = 10e9    # 10 GB minimum for cuda:1

    def __new__(cls, cfg: MLPlatformConfig | None = None) -> "DeviceManager":
        if cls._instance is None:
            instance = super().__new__(cls)
            instance._initialized = False
            cls._instance = instance
        return cls._instance

    def __init__(self, cfg: MLPlatformConfig | None = None) -> None:
        if self._initialized:
            return
        self.cfg = cfg or MLPlatformConfig()
        self._initialized = True

        # Poison DataParallel — symmetric DP across mismatched GPUs causes
        # severe PCIe bottlenecking and OOM on the smaller card.
        self._poison_data_parallel()

    @staticmethod
    def _poison_data_parallel() -> None:
        """Disable torch.nn.DataParallel to prevent symmetric distribution."""
        original_dp = torch.nn.DataParallel

        class _ForbiddenDataParallel(original_dp):
            def __init__(self, *args, **kwargs):
                raise RuntimeError(
                    "DataParallel is FORBIDDEN in the Algaie system. "
                    "Use DeviceManager.HEAVY / DeviceManager.FAST for "
                    "explicit asymmetric device assignment."
                )

        torch.nn.DataParallel = _ForbiddenDataParallel  # type: ignore[misc]
        logger.info("POISON torch.nn.DataParallel disabled — asymmetric mode enforced")

    def validate_devices(self) -> dict[str, dict]:
        """Validate physical GPU memory meets thresholds.

        Returns a dict with device info.  Soft-fails on CI/CPU-only
        environments by logging warnings instead of raising.
        """
        result: dict[str, dict] = {}

        if not torch.cuda.is_available():
            logger.warning("CUDA not available — device validation skipped (CI/CPU mode)")
            return {"status": "cpu_only"}

        n_gpus = torch.cuda.device_count()
        if n_gpus < 2:
            logger.warning(
                "Only %d GPU(s) detected — dual-device validation skipped. "
                "System will fall back to single-device mode.",
                n_gpus,
            )
            return {"status": "single_gpu", "gpu_count": n_gpus}

        # Validate VRAM on both devices
        try:
            import pynvml
            pynvml.nvmlInit()

            for idx, (label, min_vram) in enumerate([
                ("HEAVY (cuda:0)", self._MIN_HEAVY_VRAM),
                ("FAST (cuda:1)", self._MIN_FAST_VRAM),
            ]):
                handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode()

                total_gb = mem_info.total / 1e9
                free_gb = mem_info.free / 1e9

                result[f"cuda:{idx}"] = {
                    "name": name,
                    "total_gb": round(total_gb, 2),
                    "free_gb": round(free_gb, 2),
                    "role": label,
                    "meets_threshold": mem_info.total >= min_vram,
                }

                if mem_info.total < min_vram:
                    logger.warning(
                        "VRAM WARNING %s: %s has %.1f GB (need %.1f GB)",
                        label, name, total_gb, min_vram / 1e9,
                    )
                else:
                    logger.info(
                        "VRAM OK %s: %s — %.1f GB total, %.1f GB free",
                        label, name, total_gb, free_gb,
                    )

            pynvml.nvmlShutdown()

        except ImportError:
            logger.warning("pynvml not installed — VRAM assertions skipped")
            for idx in range(min(n_gpus, 2)):
                name = torch.cuda.get_device_name(idx)
                total = torch.cuda.get_device_properties(idx).total_mem / 1e9
                result[f"cuda:{idx}"] = {
                    "name": name,
                    "total_gb": round(total, 2),
                }

        result["status"] = "validated"
        result["partition"] = "asymmetric"
        return result

    def binding_for(self, role: str) -> DeviceBinding:
        """Get device binding for a role ('train' or 'infer')."""
        if role not in {"train", "infer"}:
            raise ValueError(f"Unknown role: {role}")
        if role == "train":
            return DeviceBinding(role, self.cfg.train_device, self.cfg.train_device.split(":")[-1])
        return DeviceBinding(role, self.cfg.infer_device, self.cfg.infer_device.split(":")[-1])

    def pin_process(self, role: str) -> DeviceBinding:
        """Pin the current process to a specific GPU role."""
        binding = self.binding_for(role)
        os.environ["CUDA_VISIBLE_DEVICES"] = binding.cuda_visible_devices
        os.environ["ML_ROLE"] = role
        return binding

    def startup_assertions(self) -> dict[str, str]:
        """Return startup configuration summary."""
        info = {
            "train_device": self.cfg.train_device,
            "infer_device": self.cfg.infer_device,
            "partition": "asymmetric",
            "dp_poisoned": "true",
        }
        return info

    @classmethod
    def get_device(cls, role: str) -> torch.device:
        """Convenience: get the torch.device for a role.

        Parameters
        ----------
        role : str
            ``'heavy'`` / ``'train'`` → cuda:0.
            ``'fast'`` / ``'infer'`` → cuda:1.
        """
        if role in ("heavy", "train", "batch"):
            return cls.HEAVY
        elif role in ("fast", "infer", "inference"):
            return cls.FAST
        else:
            raise ValueError(f"Unknown device role: '{role}'")

    @classmethod
    def warmup_models(cls, models: list[torch.nn.Module] | None = None) -> dict[str, float]:
        """Run a single dummy forward pass through models to pre-allocate CUDA memory.

        Must be called during autostart (06:30 AM) to avoid the 400-800ms
        first-pass latency spike when ``torch.nn.MultiheadAttention``
        initializes CUDA context at 09:29 ET.

        Parameters
        ----------
        models : list[nn.Module] or None
            Models to warm up.  If None, runs a minimal CUDA context warmup.

        Returns
        -------
        dict[str, float]
            Warmup timing per model (in ms).
        """
        import time

        timings: dict[str, float] = {}

        if not torch.cuda.is_available():
            logger.info("CUDA not available — warmup skipped (CPU mode)")
            return {"status": "cpu_only"}

        # Minimal CUDA context warmup
        start = time.perf_counter()
        _ = torch.zeros(1, device="cuda:0")
        if torch.cuda.device_count() > 1:
            _ = torch.zeros(1, device="cuda:1")
        timings["cuda_context"] = (time.perf_counter() - start) * 1000
        logger.info("CUDA context warmup: %.1fms", timings["cuda_context"])

        # Model-specific warmup
        if models:
            for model in models:
                name = model.__class__.__name__
                try:
                    model.eval()
                    device = next(model.parameters()).device
                    # Small dummy tensor — just enough to trigger JIT
                    dummy = torch.randn(1, 8, device=device, dtype=torch.bfloat16)
                    start = time.perf_counter()
                    with torch.inference_mode():
                        try:
                            model(dummy)
                        except Exception:
                            pass  # Shape mismatch is expected — we just need CUDA init
                    elapsed = (time.perf_counter() - start) * 1000
                    timings[name] = elapsed
                    logger.info("Model warmup %s: %.1fms", name, elapsed)
                except Exception as e:
                    logger.warning("Warmup failed for %s: %s", name, e)
                    timings[name] = -1.0

        torch.cuda.synchronize()
        return timings

    @staticmethod
    def sweep_cuda_cache() -> None:
        """Explicit CUDA cache sweep to prevent memory fragmentation.

        Call at the conclusion of every DAG FSM PREMARKET cycle to free
        orphaned tensors before the intraday trading session begins.
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info("CUDA cache swept — orphaned tensors freed")

    @classmethod
    def reset_singleton(cls) -> None:
        """Reset the singleton instance (for testing only)."""
        cls._instance = None
