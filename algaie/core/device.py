"""
Centralised GPU device selection.

Reads ``ALGAE_CUDA_DEVICE`` from the environment to decide which GPU to use.
Default is ``cuda:1`` (3090 Ti).  Set to ``cuda:0`` to use the primary 4070 Super,
or ``cpu`` to disable GPU entirely.

Usage
-----
>>> from algaie.core.device import get_device
>>> device = get_device()          # cuda:1 by default
>>> device = get_device("cuda:0")  # explicit override
"""
from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)

_DEFAULT_CUDA_DEVICE = "cuda:1"  # 3090 Ti


def get_device(override: str | None = None) -> torch.device:
    """Return the project-wide torch device.

    Resolution order:
      1. ``override`` argument (if not None)
      2. ``ALGAE_CUDA_DEVICE`` env var (fallback ``ALGAIE_CUDA_DEVICE`` for compatibility)
      3. ``cuda:1`` if CUDA is available
      4. ``cpu`` as final fallback
    """
    if override is not None:
        choice = override
    else:
        choice = os.environ.get("ALGAE_CUDA_DEVICE") or os.environ.get("ALGAIE_CUDA_DEVICE", _DEFAULT_CUDA_DEVICE)

    if choice.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested (%s) but not available — falling back to CPU", choice)
        return torch.device("cpu")

    device = torch.device(choice)

    # Validate that the requested GPU index actually exists
    if device.type == "cuda":
        idx = device.index if device.index is not None else 0
        if idx >= torch.cuda.device_count():
            logger.warning(
                "Requested cuda:%d but only %d GPU(s) found — falling back to cuda:0",
                idx,
                torch.cuda.device_count(),
            )
            device = torch.device("cuda:0")

    logger.info("Device: %s", device)
    return device
