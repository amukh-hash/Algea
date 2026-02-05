"""Verify Chronos-2 authentication for protocol-aligned daily inference/training."""

from transformers import AutoModel
import sys

try:
    print("Attempting to load amazon/chronos-2...")
    m = AutoModel.from_pretrained(
        "amazon/chronos-2",
        trust_remote_code=True
    )
    print("Chronos-2 loaded successfully")
except Exception as e:
    print(f"Failed to load: {e}")
    sys.exit(1)
