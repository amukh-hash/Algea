"""lock_production_weights.py — Serialize model weights to safetensors format.

Converts all .pt checkpoints to .safetensors for:
- Deterministic, zero-copy memory-mapped loading
- Prevention of arbitrary code execution (pickle-free)
- Production deployment on ASGI inference engine

Sets read-only permissions to prevent accidental overwrites.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("lock_weights")

PROD_DIR = PROJECT_ROOT / "backend" / "artifacts" / "production_weights"
MODEL_DIR = PROJECT_ROOT / "backend" / "artifacts" / "models"


def _set_readonly(path: Path) -> None:
    """Set file to read-only (cross-platform)."""
    if sys.platform == "win32":
        subprocess.run(["attrib", "+R", str(path)], check=True, capture_output=True)
    else:
        os.chmod(path, 0o444)
    logger.info("LOCK  %s → read-only", path.name)


def lock_kronos() -> Path | None:
    """Convert Kronos PatchTST checkpoint to safetensors."""
    src = MODEL_DIR / "kronos" / "patchtst_kronos_best.pt"
    if not src.exists():
        logger.warning("SKIP  Kronos — %s not found", src)
        return None

    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)

    dst = PROD_DIR / "kronos_locked.safetensors"
    save_file(state_dict, str(dst))
    logger.info("SAVE  Kronos → %s (%d tensors)", dst, len(state_dict))

    # Also save config metadata
    config = ckpt.get("config", {})
    if config:
        import json
        config_dst = PROD_DIR / "kronos_config.json"
        with open(config_dst, "w") as f:
            json.dump(config, f, indent=2)
        logger.info("  config: %s", config)

    _set_readonly(dst)
    return dst


def lock_mera() -> Path | None:
    """Convert MERA SMoE checkpoint to safetensors."""
    # Find best MERA checkpoint
    mera_dir = MODEL_DIR / "mera"
    candidates = list(mera_dir.glob("*.pt")) + list(mera_dir.glob("*.pth"))
    if not candidates:
        logger.warning("SKIP  MERA — no checkpoints in %s", mera_dir)
        return None

    src = candidates[0]
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)

    dst = PROD_DIR / "mera_locked.safetensors"
    save_file(state_dict, str(dst))
    logger.info("SAVE  MERA → %s (%d tensors)", dst, len(state_dict))
    _set_readonly(dst)
    return dst


def lock_vrp() -> Path | None:
    """Convert VRP ST-Transformer + TD3 checkpoint to safetensors."""
    vrp_dir = MODEL_DIR / "vrp"

    # Lock pre-trained encoder
    encoder_path = vrp_dir / "st_transformer_pretrained.pt"
    if encoder_path.exists():
        ckpt = torch.load(encoder_path, map_location="cpu", weights_only=False)
        state_dict = ckpt if isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()) else ckpt.get("model_state_dict", ckpt)
        dst = PROD_DIR / "vrp_encoder_locked.safetensors"
        save_file(state_dict, str(dst))
        logger.info("SAVE  VRP encoder → %s (%d tensors)", dst, len(state_dict))
        _set_readonly(dst)
    else:
        logger.warning("SKIP  VRP encoder — %s not found", encoder_path)

    # Lock actor-critic
    actor_path = vrp_dir / "td3_actor_best.pt"
    critic_path = vrp_dir / "td3_critic_best.pt"

    saved = []
    for name, path in [("actor", actor_path), ("critic", critic_path)]:
        if path.exists():
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
            state_dict = ckpt if isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()) else ckpt.get("model_state_dict", ckpt)
            dst = PROD_DIR / f"vrp_{name}_locked.safetensors"
            save_file(state_dict, str(dst))
            logger.info("SAVE  VRP %s → %s (%d tensors)", name, dst, len(state_dict))
            _set_readonly(dst)
            saved.append(dst)
        else:
            logger.warning("SKIP  VRP %s — %s not found", name, path)

    return saved[0] if saved else None


def main() -> None:
    PROD_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("=" * 60)
    logger.info("PRODUCTION WEIGHT SERIALIZATION")
    logger.info("Output: %s", PROD_DIR)
    logger.info("=" * 60)

    results = {}
    results["kronos"] = lock_kronos()
    results["mera"] = lock_mera()
    results["vrp"] = lock_vrp()

    logger.info("=" * 60)
    for sleeve, path in results.items():
        status = f"✓ {path}" if path else "✗ skipped"
        logger.info("  %s: %s", sleeve, status)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
