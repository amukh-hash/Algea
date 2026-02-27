#!/usr/bin/env python3
from __future__ import annotations

import os

import uvicorn

from backend.app.ml_platform.device_manager import DeviceManager
from backend.app.ml_platform.inference_gateway.server import InferenceGatewayServer


if __name__ == "__main__":
    binding = DeviceManager().pin_process("infer")
    server = InferenceGatewayServer()
    app = server.make_app()
    port = int(os.getenv("INFERENCED_PORT", "8111"))
    print(f"inferenced pinned to {binding.logical_device} (CUDA_VISIBLE_DEVICES={binding.cuda_visible_devices})")
    uvicorn.run(app, host="0.0.0.0", port=port)
