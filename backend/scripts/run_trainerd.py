#!/usr/bin/env python3
from __future__ import annotations

import json

from backend.app.ml_platform.device_manager import DeviceManager
from backend.app.ml_platform.training.jobs import build_job
from backend.app.ml_platform.training.trainerd import TrainerDaemon


if __name__ == "__main__":
    binding = DeviceManager().pin_process("train")
    daemon = TrainerDaemon()
    print(f"trainerd pinned to {binding.logical_device} (CUDA_VISIBLE_DEVICES={binding.cuda_visible_devices})")
    print("trainerd ready")
    # non-blocking noop worker process: can receive one job payload from stdin.
    try:
        payload = json.loads(input())
    except EOFError:
        payload = None
    if payload:
        job = build_job(payload)
        print(daemon.run_job(job))
