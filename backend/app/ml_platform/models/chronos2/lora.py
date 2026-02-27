from __future__ import annotations


def train_lora_adapter_stub(series_windows: list[tuple[list[float], list[float]]], epochs: int) -> dict:
    # deterministic, cheap stand-in for CI environments without GPU.
    total = sum(sum(inp) + sum(tgt) for inp, tgt in series_windows) if series_windows else 0.0
    return {"adapter_scale": total / max(len(series_windows), 1), "epochs": epochs}
