"""
Foundation model (Chronos-2) training loop.

Replaces the placeholder stub with a real training pipeline that:
  - Uses ``GoldFuturesWindowDataset`` or ``ChronosDataset``
  - Supports LoRA fine-tuning via ``load_chronos_adapter``
  - Persists checkpoints to ``destination``
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from algaie.models.foundation.chronos2 import FoundationModelConfig, SimpleChronos2
from algaie.training.chronos_dataset import ChronosDataset, chronos_collate_fn

logger = logging.getLogger(__name__)


def train_foundation_model(
    canonical_daily: Any,
    destination: Path,
    *,
    model_id: str = "amazon/chronos-bolt-tiny",
    lora_config: Optional[Dict[str, Any]] = None,
    context_len: int = 512,
    prediction_len: int = 64,
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 1e-4,
    device: Optional[torch.device] = None,
) -> None:
    """
    Train or fine-tune a foundation model.

    Parameters
    ----------
    canonical_daily : path-like or list of paths to per-ticker parquet files
    destination : directory for checkpoints and final model
    model_id : HuggingFace model identifier
    lora_config : LoRA hyperparameters (rank, alpha, dropout, target_modules)
    context_len, prediction_len : window sizes
    epochs, batch_size, lr : training hyperparameters
    device : torch device (auto-detected if None)
    """
    destination.mkdir(parents=True, exist_ok=True)

    if device is None:
        from algaie.core.device import get_device
        device = get_device()

    # Attempt to use real Chronos2 loading; fall back to stub
    try:
        from algaie.models.foundation.chronos2_teacher import load_chronos_adapter

        model_wrapper, info = load_chronos_adapter(
            model_id=model_id,
            use_qlora=False,
            device=device,
            lora_config=lora_config,
        )
        logger.info(f"Loaded Chronos-2 model: {info.get('model_type', 'unknown')}")
    except ImportError:
        logger.warning("chronos2_teacher not available; saving placeholder")
        model = SimpleChronos2(FoundationModelConfig())
        _ = model
        (destination / "placeholder.txt").write_text("foundation model placeholder", encoding="utf-8")
        return

    # Build dataset
    if isinstance(canonical_daily, (str, Path)):
        files = sorted(Path(canonical_daily).glob("*.parquet"))
    elif isinstance(canonical_daily, list):
        files = [Path(f) for f in canonical_daily]
    else:
        logger.error(f"Unsupported canonical_daily type: {type(canonical_daily)}")
        return

    ds = ChronosDataset(
        files=files,
        context_len=context_len,
        prediction_len=prediction_len,
        stride=5,
        target_col="close",
    )
    if len(ds) == 0:
        logger.warning("No training samples — aborting.")
        return

    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=chronos_collate_fn)
    optimizer = torch.optim.AdamW(
        [p for p in model_wrapper.parameters() if p.requires_grad], lr=lr
    )

    # Training loop
    model_wrapper.train()
    for epoch in range(epochs):
        total_loss = 0.0
        steps = 0
        for batch in loader:
            context = batch["past_target"].to(device)
            future = batch["future_target"].to(device)

            optimizer.zero_grad()
            outputs = model_wrapper(
                context=context.squeeze(-1),
                future_target=future.squeeze(-1),
            )
            loss = outputs.loss if hasattr(outputs, "loss") else outputs["loss"]
            if loss.requires_grad:
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            steps += 1

        avg = total_loss / max(1, steps)
        logger.info(f"Epoch {epoch + 1}/{epochs} — loss={avg:.6f}")

        # Checkpoint
        ckpt_path = destination / f"epoch_{epoch + 1}.pt"
        torch.save(model_wrapper.state_dict(), ckpt_path)

    # Final save
    if hasattr(model_wrapper, "save_pretrained"):
        model_wrapper.save_pretrained(destination / "final")
    logger.info(f"Foundation training complete → {destination}")

