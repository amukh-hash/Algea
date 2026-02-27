"""
Post-training evaluation of Chronos-2 Teacher models.
Computes Coverage80, IC (Spearman), and quantile metrics using the native
pipeline.predict_quantiles API.

Usage:
    python backend/scripts/eval_chronos_metrics.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from algea.training.chronos_dataset import ChronosDataset, chronos_collate_fn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Quantile levels used by Chronos-2 (21 levels)
CHRONOS2_QUANTILES = [
    0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
    0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99,
]
Q10_IDX = CHRONOS2_QUANTILES.index(0.1)   # 2
Q50_IDX = CHRONOS2_QUANTILES.index(0.5)   # 10
Q90_IDX = CHRONOS2_QUANTILES.index(0.9)   # 18


def load_pipeline_with_checkpoint(run_dir: Path, device: torch.device):
    """Load Chronos2Pipeline + apply LoRA checkpoint from run_dir."""
    from chronos import Chronos2Pipeline
    from peft import LoraConfig, inject_adapter_in_model

    config = json.loads((run_dir / "config.json").read_text())
    model_id = config["model_id"]

    logger.info(f"Loading pipeline: {model_id}")
    pipeline = Chronos2Pipeline.from_pretrained(
        model_id, device_map=None, dtype=torch.float32,
    )
    model = pipeline.model.to(device)

    # Inject LoRA with same config as training
    target_modules = ["q", "v", "k", "o"]
    if config.get("lora_ffn", False):
        from backend.scripts._cli_utils import detect_ffn_modules
        ffn_mods = detect_ffn_modules(model)
        if ffn_mods:
            target_modules += ffn_mods

    peft_cfg = LoraConfig(
        r=config["lora_rank"],
        lora_alpha=config["lora_alpha"],
        target_modules=target_modules,
        lora_dropout=config.get("lora_dropout", 0.05),
        bias="none",
    )
    model = inject_adapter_in_model(peft_cfg, model)

    # Load best checkpoint
    best_ckpt = run_dir / "checkpoints" / "best" / "model.pt"
    if best_ckpt.exists():
        state = torch.load(best_ckpt, map_location=device, weights_only=True)
        model.load_state_dict(state, strict=False)
        logger.info(f"Loaded best checkpoint: {best_ckpt}")
    else:
        final_ckpt = run_dir / "checkpoints" / "final" / "model.pt"
        state = torch.load(final_ckpt, map_location=device, weights_only=True)
        model.load_state_dict(state, strict=False)
        logger.info(f"Loaded final checkpoint: {final_ckpt}")

    model.eval()
    pipeline.model = model
    return pipeline, config


def build_val_dataset(config: dict) -> ChronosDataset:
    """Rebuild the validation dataset using the same split as training."""
    ticker_dir = ROOT / "backend" / "data" / "canonical" / "per_ticker"
    universe_path = ROOT / "backend" / "data" / "canonical" / "universe_frame.parquet"

    import pandas as pd
    uf = pd.read_parquet(universe_path)
    uf["date"] = pd.to_datetime(uf["date"]).dt.date
    mask_col = config.get("mask_col", "auto")
    if mask_col == "auto":
        mask_col = "is_tradable" if "is_tradable" in uf.columns else "is_observable"
    if mask_col not in uf.columns:
        mask_col = "is_observable"
    obs_lookup = uf[uf[mask_col]].groupby("symbol")["date"].apply(set).to_dict()

    all_files = sorted(ticker_dir.glob("*.parquet"))
    seed = config.get("seed", 42)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(all_files))
    n_val = max(1, int(len(all_files) * config.get("val_frac", 0.10)))
    val_files = [all_files[i] for i in idx[-n_val:]]

    val_ds = ChronosDataset(
        files=val_files,
        context_len=config["context_len"],
        prediction_len=config["prediction_len"],
        stride=config.get("stride", 10),
        target_col=config.get("target_col", "close"),
        obs_lookup=obs_lookup,
        max_samples_per_file=config.get("max_samples_per_file", 100),
        sampling_mode=config.get("sampling_mode", "random_anchors"),
        seed=seed + 1,
    )
    logger.info(f"Val dataset: {len(val_ds)} samples")
    return val_ds


@torch.no_grad()
def evaluate(pipeline, val_ds, config, device, max_batches=50):
    """Compute Coverage80, IC, and quantile metrics."""
    prediction_len = config["prediction_len"]
    batch_size = config.get("batch_size", 4)

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=chronos_collate_fn,
        num_workers=0,
        pin_memory=False,
    )

    realized_list = []
    q10_list = []
    q50_list = []
    q90_list = []

    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break

        context = batch["past_target"].squeeze(-1)  # [B, C], keep on CPU
        future = batch["future_target"].squeeze(-1)  # [B, P], keep on CPU
        y_end = future[:, -1]  # [B]
        realized_list.append(y_end)

        # Pass as list of 1D tensors (pipeline handles batching internally)
        context_list = [context[b] for b in range(context.shape[0])]  # list of [C]
        
        try:
            predictions = pipeline.predict(
                context_list,
                prediction_length=prediction_len,
                limit_prediction_length=False,
            )
        except Exception as e:
            logger.warning(f"  Batch {i} predict failed: {e}")
            # Remove the realized values we already appended
            realized_list.pop()
            continue

        # predictions is a list of B tensors, each (1, 21, P) for univariate
        for b_idx in range(len(predictions)):
            pred = predictions[b_idx]  # (1, 21, P) or (21, P)
            if pred.ndim == 3:
                pred = pred.squeeze(0)  # (21, P)
            # Terminal step quantiles
            q10_list.append(pred[Q10_IDX, -1].item())
            q50_list.append(pred[Q50_IDX, -1].item())
            q90_list.append(pred[Q90_IDX, -1].item())

        if (i + 1) % 10 == 0:
            logger.info(f"  Batch {i+1}/{max_batches}...")

    realized = torch.cat(realized_list).numpy()
    q10 = np.array(q10_list)
    q50 = np.array(q50_list)
    q90 = np.array(q90_list)

    n = len(realized)
    logger.info(f"  Total samples evaluated: {n}")

    # Coverage80: fraction where realized in [q10, q90]
    in_interval = (realized >= q10) & (realized <= q90)
    coverage80 = float(in_interval.mean())

    # IC: Spearman correlation between q50 and realized
    mask = np.isfinite(q50) & np.isfinite(realized)
    if mask.sum() > 2:
        ic, p_val = spearmanr(q50[mask], realized[mask])
        ic = float(ic) if np.isfinite(ic) else 0.0
    else:
        ic = 0.0
        p_val = 1.0

    # Quantile calibration: fraction below each quantile
    frac_below_q10 = float((realized < q10).mean())
    frac_below_q50 = float((realized < q50).mean())
    frac_below_q90 = float((realized < q90).mean())

    metrics = {
        "n_samples": n,
        "coverage80": coverage80,
        "ic_end": ic,
        "ic_pvalue": float(p_val),
        "frac_below_q10": frac_below_q10,
        "frac_below_q50": frac_below_q50,
        "frac_below_q90": frac_below_q90,
    }
    return metrics


def main():
    from algea.core.device import get_device
    device = get_device()

    runs_dir = ROOT / "backend" / "data" / "runs"
    run_ids = [
        "RUN-2026-02-09-175844",  # Teacher-10d
        "RUN-2026-02-09-181337",  # Teacher-30d
    ]

    for run_id in run_ids:
        run_dir = runs_dir / run_id
        if not run_dir.exists():
            logger.warning(f"Run {run_id} not found, skipping")
            continue

        config = json.loads((run_dir / "config.json").read_text())
        horizon = config["prediction_len"]
        logger.info("=" * 60)
        logger.info(f"Evaluating {run_id} (horizon={horizon}d)")
        logger.info("=" * 60)

        pipeline, config = load_pipeline_with_checkpoint(run_dir, device)
        val_ds = build_val_dataset(config)
        metrics = evaluate(pipeline, val_ds, config, device, max_batches=50)

        logger.info(f"  Coverage80:    {metrics['coverage80']:.4f}")
        logger.info(f"  IC (Spearman): {metrics['ic_end']:.4f} (p={metrics['ic_pvalue']:.4e})")
        logger.info(f"  Calibration:   q10={metrics['frac_below_q10']:.3f}  q50={metrics['frac_below_q50']:.3f}  q90={metrics['frac_below_q90']:.3f}")

        # Save metrics to run directory
        out_path = run_dir / "eval_metrics.json"
        out_path.write_text(json.dumps(metrics, indent=2))
        logger.info(f"  Saved to: {out_path}")

        # Clean up GPU memory
        del pipeline
        torch.cuda.empty_cache()

    logger.info("Done.")


if __name__ == "__main__":
    main()
