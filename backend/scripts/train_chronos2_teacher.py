"""
Chronos-2 Teacher Training — Full pipeline (v2).

Uses:
  - ChronosDataset (per-ticker parquet + universe gating via is_tradable/is_observable)
  - Chronos2Pipeline.from_pretrained("amazon/chronos-2") -> LoRA injection
  - Native pinball (quantile) loss on 21 quantiles
  - Gradient accumulation, cosine LR, AMP + GradScaler, checkpointing + validation
  - Optional past covariates from canonical market_covariates.parquet
  - Validation: pinball loss + calibration coverage80 + cross-sectional IC proxy
  - Multi-horizon presets: teacher10 (10d) / teacher30 (30d)

Usage:
    python backend/scripts/train_chronos2_teacher.py
    python backend/scripts/train_chronos2_teacher.py --preset teacher10 --use-covariates
    python backend/scripts/train_chronos2_teacher.py --preset teacher30 --sampling-mode random_anchors
    python backend/scripts/train_chronos2_teacher.py --epochs 3 --batch-size 8 --lora-ffn
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from algea.training.chronos_dataset import ChronosDataset, chronos_collate_fn
from backend.scripts._cli_utils import detect_ffn_modules

# --- Logging ----------------------------------------------------------------
log_dir = ROOT / "backend" / "data" / "runs"
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_dir / "chronos2_train.log"),
    ],
)
logger = logging.getLogger(__name__)

# --- Defaults ---------------------------------------------------------------
DEFAULTS = {
    "model_id": "amazon/chronos-2",
    "context_len": 252,
    "prediction_len": 30,
    "stride": 10,
    "target_col": "close",
    "batch_size": 4,
    "grad_accum": 8,         # effective batch = 32
    "epochs": 2,
    "lr": 5e-5,
    "warmup_steps": 100,
    "val_every": 500,        # global steps
    "ckpt_every": 500,
    "log_every": 25,
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "max_samples_per_file": 100,
    "val_frac": 0.10,
    "seed": 42,
}

# Horizon presets
PRESETS = {
    "teacher10": {"prediction_len": 10, "stride": 10},
    "teacher30": {"prediction_len": 30, "stride": 10},
}

# Standard quantile levels used by Chronos-2
QUANTILE_LEVELS = [
    0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
    0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99,
]


# --- Seeds ------------------------------------------------------------------
def set_seeds(seed: int) -> None:
    """Set seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --- Universe gating --------------------------------------------------------
def build_mask_lookup(universe_path: Path, mask_col: str = "auto") -> dict:
    """
    Load universe frame -> {symbol: set(datetime.date)} for eligible anchor gating.

    mask_col:
      "auto" -> prefer is_tradable if present, else is_observable
      "is_tradable" -> use is_tradable column
      "is_observable" -> use is_observable column
    """
    logger.info(f"Loading universe frame: {universe_path}")
    uf = pd.read_parquet(universe_path)
    uf["date"] = pd.to_datetime(uf["date"]).dt.date

    # Resolve mask column
    if mask_col == "auto":
        if "is_tradable" in uf.columns:
            mask_col = "is_tradable"
        else:
            mask_col = "is_observable"
    if mask_col not in uf.columns:
        logger.warning(
            f"Mask column '{mask_col}' not found, "
            f"falling back to is_observable"
        )
        mask_col = "is_observable"

    logger.info(f"Universe gating: using '{mask_col}' column")
    obs = uf[uf[mask_col]].groupby("symbol")["date"].apply(set).to_dict()
    logger.info(f"Eligible symbols: {len(obs)}")
    return obs


def build_tier_lookup(universe_path: Path) -> dict | None:
    """Load tier assignments {symbol: tier_int} from universe frame."""
    try:
        uf = pd.read_parquet(universe_path, columns=["symbol", "tier"])
        if "tier" not in uf.columns:
            return None
        # Take the modal tier per symbol
        tier_map = uf.groupby("symbol")["tier"].agg(
            lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 3
        ).to_dict()
        logger.info(f"Tier lookup: {len(tier_map)} symbols")
        return tier_map
    except Exception:
        return None


# --- Data -------------------------------------------------------------------
def make_datasets(args, obs_lookup, covariates_df, tier_lookup):
    """Create train/val ChronosDataset splits (file-level split)."""
    ticker_dir = ROOT / "backend" / "data" / "canonical" / "per_ticker"
    all_files = sorted(ticker_dir.glob("*.parquet"))
    logger.info(f"Total per-ticker files: {len(all_files)}")

    # Deterministic shuffle + split
    rng = np.random.RandomState(args.seed)
    idx = rng.permutation(len(all_files))
    n_val = max(1, int(len(all_files) * args.val_frac))
    train_files = [all_files[i] for i in idx[:-n_val]]
    val_files = [all_files[i] for i in idx[-n_val:]]
    logger.info(
        f"Train tickers: {len(train_files)}, Val tickers: {len(val_files)}"
    )

    common_kwargs = dict(
        context_len=args.context_len,
        prediction_len=args.prediction_len,
        stride=args.stride,
        target_col=args.target_col,
        obs_lookup=obs_lookup,
        max_samples_per_file=args.max_samples_per_file,
        sampling_mode=args.sampling_mode,
        covariates_df=covariates_df,
        tier_lookup=tier_lookup,
    )
    train_ds = ChronosDataset(
        files=train_files,
        seed=args.seed,
        **common_kwargs,
    )
    val_ds = ChronosDataset(
        files=val_files,
        seed=args.seed + 1,
        **common_kwargs,
    )
    return train_ds, val_ds


# --- Model ------------------------------------------------------------------
def load_model(args, device):
    """Load amazon/chronos-2 via Chronos2Pipeline, inject LoRA, freeze base."""
    from chronos import Chronos2Pipeline
    from peft import LoraConfig, inject_adapter_in_model

    logger.info(f"Loading model: {args.model_id}")
    # Load pipeline without device_map (single GPU)
    pipeline = Chronos2Pipeline.from_pretrained(
        args.model_id,
        device_map=None,
        dtype=torch.float32,
    )
    model = pipeline.model  # Chronos2Model
    model = model.to(device)

    # Read patch config
    chronos_cfg = model.config.chronos_config
    output_patch_size = chronos_cfg.get("output_patch_size", 16)
    num_output_patches = math.ceil(args.prediction_len / output_patch_size)
    logger.info(
        f"output_patch_size={output_patch_size}, "
        f"num_output_patches={num_output_patches}"
    )

    # Check covariate support
    n_cov = chronos_cfg.get("num_future_covariates", 0)
    if args.use_covariates and n_cov == 0:
        logger.warning(
            f"Model config has num_future_covariates={n_cov}, but "
            f"--use-covariates is set. Disabling covariates to prevent "
            f"shape mismatch."
        )
        args.use_covariates = False
    elif args.use_covariates and n_cov > 0:
        logger.info(f"Model supports {n_cov} future covariates.")


    # LoRA target modules
    target_modules = ["q", "v", "k", "o"]
    if args.lora_ffn:
        ffn_mods = detect_ffn_modules(model)
        if ffn_mods:
            target_modules = target_modules + ffn_mods
            logger.info(f"LoRA FFN modules detected: {ffn_mods}")
        else:
            logger.warning(
                "No FFN modules detected for LoRA; proceeding with "
                "attention-only"
            )

    peft_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
    )
    model = inject_adapter_in_model(peft_cfg, model)

    # Freeze base, keep LoRA trainable
    trainable = 0
    frozen = 0
    for name, param in model.named_parameters():
        if "lora_" in name.lower():
            param.requires_grad = True
            trainable += param.numel()
        else:
            param.requires_grad = False
            frozen += param.numel()

    logger.info(
        f"Parameters: {trainable:,} trainable, {frozen:,} frozen "
        f"({100*trainable/(trainable+frozen):.2f}%)"
    )
    logger.info(f"LoRA target_modules: {target_modules}")
    return pipeline, model, num_output_patches, target_modules


# --- Loss -------------------------------------------------------------------
def compute_loss(model, batch, device, num_output_patches, use_amp=False):
    """
    Forward pass through Chronos2Model with native quantile/pinball loss.
    Returns loss tensor.
    """
    context = batch["past_target"].squeeze(-1).to(device)       # [B, C]
    target = batch["future_target"].squeeze(-1).to(device)      # [B, P]

    context_mask = torch.ones_like(context, dtype=torch.bool)
    target_mask = torch.ones_like(target, dtype=torch.bool)

    with torch.amp.autocast("cuda", enabled=use_amp):
        # Inject future covariates if available in batch
        kwargs = {}
        if "future_covariates" in batch:
            kwargs["future_covariates"] = batch["future_covariates"].to(device)

        outputs = model(
            context=context,
            context_mask=context_mask,
            future_target=target,
            future_target_mask=target_mask,
            num_output_patches=num_output_patches,
            **kwargs,
        )
    return outputs.loss


# --- Validation -------------------------------------------------------------
@torch.no_grad()
def validate(
    model, pipeline, val_loader, device, num_output_patches,
    prediction_len, max_batches=50, use_amp=False,
):
    """
    Validate: pinball loss + calibration coverage80 + cross-sectional IC.

    Returns dict with val_loss, coverage80, ic_end, quantile_method.
    """
    model.eval()
    total_loss = 0.0
    n_loss = 0

    # Accumulators for coverage and IC
    realized_list = []
    pred_q10_list = []
    pred_q50_list = []
    pred_q90_list = []
    quantile_method = "none"

    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        loss = compute_loss(
            model, batch, device, num_output_patches, use_amp=use_amp
        )
        total_loss += loss.item()
        n_loss += 1

    # Quantile extraction for coverage/IC (pipeline fallback, limited batches)
    ic_batches = min(max_batches, 20)
    try:
        quantile_method = "pipeline_predict"
        for i, batch in enumerate(val_loader):
            if i >= ic_batches:
                break

            context = batch["past_target"].squeeze(-1).to(device)  # [B, C]
            future = batch["future_target"].squeeze(-1).to(device)  # [B, P]

            # Realized end-of-horizon value
            y_end = future[:, -1]  # [B]
            realized_list.append(y_end.cpu())

            # Use pipeline.predict for quantile forecasts
            kw = {}
            if "future_covariates" in batch:
                kw["future_covariates"] = batch["future_covariates"].to(device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                forecasts = pipeline.predict(
                    context,
                    prediction_length=prediction_len,
                    num_samples=100,
                    **kw,
                )  # [B, S, P]

            if forecasts.ndim == 4:
                forecasts = forecasts.squeeze(-1)

            # Terminal step quantiles
            terminal = forecasts[:, :, -1]  # [B, S]
            q10 = torch.quantile(terminal, 0.1, dim=1)
            q50 = torch.quantile(terminal, 0.5, dim=1)
            q90 = torch.quantile(terminal, 0.9, dim=1)

            pred_q10_list.append(q10.cpu())
            pred_q50_list.append(q50.cpu())
            pred_q90_list.append(q90.cpu())

    except Exception as e:
        logger.warning(f"Quantile extraction failed ({e}); skipping coverage/IC")
        quantile_method = "failed"

    metrics = {
        "val_loss": total_loss / max(1, n_loss),
        "coverage80": float("nan"),
        "ic_end": float("nan"),
        "quantile_method": quantile_method,
    }

    if realized_list and pred_q10_list:
        realized = torch.cat(realized_list)
        q10 = torch.cat(pred_q10_list)
        q50 = torch.cat(pred_q50_list)
        q90 = torch.cat(pred_q90_list)

        # Coverage80: fraction where realized y_end in [q10, q90]
        in_interval = (realized >= q10) & (realized <= q90)
        metrics["coverage80"] = float(in_interval.float().mean().item())

        # IC: Spearman-like correlation between q50 and realized
        if len(realized) > 2:
            q50_np = q50.numpy()
            real_np = realized.numpy()
            mask = np.isfinite(q50_np) & np.isfinite(real_np)
            if mask.sum() > 2:
                from scipy.stats import spearmanr
                ic, _ = spearmanr(q50_np[mask], real_np[mask])
                metrics["ic_end"] = float(ic) if np.isfinite(ic) else 0.0
            else:
                metrics["ic_end"] = 0.0
        else:
            metrics["ic_end"] = 0.0

    model.train()
    return metrics


# --- Setup helpers ----------------------------------------------------------
def setup_device():
    """Configure CUDA device with TF32 if available, return device."""
    from algea.core.device import get_device
    device = get_device()
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        logger.info("TF32 enabled for matmul and cuDNN")
    return device


def create_run_dir(args):
    """Create timestamped run directory with config snapshot."""
    run_id = datetime.now().strftime("RUN-%Y-%m-%d-%H%M%S")
    run_dir = ROOT / "backend" / "data" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(vars(args), indent=2, default=str), encoding="utf-8"
    )
    return run_id, run_dir, ckpt_dir


def load_optional_covariates(args):
    """Load or build covariates if --use-covariates is set."""
    if not args.use_covariates:
        return None

    cov_path = Path(args.covariates_path)
    if not cov_path.is_absolute():
        cov_path = ROOT / cov_path

    if cov_path.exists():
        covariates_df = pd.read_parquet(cov_path)
        logger.info(
            f"Covariates loaded: {len(covariates_df)} rows, "
            f"columns={list(covariates_df.columns)}"
        )
        return covariates_df

    logger.info(f"Covariates not found at {cov_path}, attempting to build...")
    try:
        from algea.data.market.covariates import build_and_persist_covariates
        ticker_dir = ROOT / "backend" / "data" / "canonical" / "per_ticker"
        return build_and_persist_covariates(ticker_dir, cov_path, overwrite=False)
    except Exception as e:
        logger.warning(f"Could not build covariates: {e}. Proceeding without covariates.")
        return None


# --- Finalization -----------------------------------------------------------
def _finalize_training(
    model, pipeline, val_loader, device, num_output_patches,
    args, use_amp, run_id, run_dir, ckpt_dir, metrics_log,
    global_step, best_val, best_coverage, best_ic,
    target_modules, train_ds, val_ds, t0,
):
    """Save final checkpoint, run final validation, and write manifest."""
    final_path = ckpt_dir / "final"
    final_path.mkdir(exist_ok=True)
    torch.save(model.state_dict(), final_path / "model.pt")

    final_val = validate(
        model, pipeline, val_loader, device, num_output_patches,
        args.prediction_len, use_amp=use_amp,
    )

    with open(run_dir / "metrics.jsonl", "w") as f:
        for m in metrics_log:
            f.write(json.dumps(m) + "\n")

    elapsed = time.time() - t0
    manifest = {
        "run_id": run_id,
        "model_id": args.model_id,
        "pipeline": "chronos2_teacher",
        "status": "COMPLETED",
        "global_steps": global_step,
        "best_val_loss": best_val,
        "best_coverage80": best_coverage,
        "best_ic_end": best_ic,
        "final_val_loss": final_val["val_loss"],
        "final_coverage80": final_val["coverage80"],
        "final_ic_end": final_val["ic_end"],
        "quantile_method": final_val["quantile_method"],
        "elapsed_seconds": elapsed,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "num_output_patches": num_output_patches,
        "prediction_len": args.prediction_len,
        "lora_target_modules": target_modules,
        "sampling_mode": args.sampling_mode,
        "mask_col": args.mask_col,
        "use_covariates": args.use_covariates,
        "amp": use_amp,
        "config": vars(args),
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )

    logger.info("=" * 60)
    logger.info(f"TRAINING COMPLETE -- {run_id}")
    logger.info(
        f"  Steps: {global_step}  |  Best val: {best_val:.4f}  |  "
        f"Time: {elapsed/60:.1f} min"
    )
    logger.info(f"  Coverage80: {best_coverage:.3f}  |  IC: {best_ic:.4f}")
    logger.info(f"  Artifacts: {run_dir}")
    logger.info("=" * 60)


# --- Training loop ----------------------------------------------------------
def train(args):
    set_seeds(args.seed)

    device = setup_device()
    logger.info(f"Config: {vars(args)}")

    # AMP
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    logger.info(f"AMP: {'ON' if use_amp else 'OFF'}")

    # Model (load first to check config for covariates support)
    pipeline, model, num_output_patches, target_modules = load_model(
        args, device
    )

    # Run directory
    run_id, run_dir, ckpt_dir = create_run_dir(args)

    # Universe gating
    universe_path = (
        ROOT / "backend" / "data" / "canonical" / "universe_frame.parquet"
    )
    obs_lookup = build_mask_lookup(universe_path, args.mask_col)
    tier_lookup = build_tier_lookup(universe_path)

    # Covariates (optional)
    covariates_df = load_optional_covariates(args)

    # Data
    train_ds, val_ds = make_datasets(
        args, obs_lookup, covariates_df, tier_lookup
    )
    logger.info(
        f"Train samples: {len(train_ds):,}  |  Val samples: {len(val_ds):,}"
    )

    if len(train_ds) == 0:
        logger.error("No training samples -- aborting.")
        return

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=chronos_collate_fn,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=chronos_collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    # Model loaded earlier
    model.train()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.lr, weight_decay=0.01
    )

    # LR schedule: linear warmup -> cosine decay
    steps_per_epoch = max(1, len(train_loader) // args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = min(
        args.warmup_steps, max(10, int(0.02 * total_steps))
    )

    def lr_fn(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)

    # State
    global_step = 0
    best_val = float("inf")
    best_coverage = float("nan")
    best_ic = float("nan")
    metrics_log = []
    t0 = time.time()

    logger.info("=" * 60)
    logger.info(f"TRAINING START -- {run_id}")
    logger.info(
        f"steps_per_epoch={steps_per_epoch}  "
        f"total_steps={total_steps}  warmup={warmup_steps}"
    )
    logger.info(f"Model: {args.model_id}  |  LoRA rank: {args.lora_rank}")
    logger.info(
        f"Sampling: {args.sampling_mode}  |  Mask: {args.mask_col}  |  "
        f"Covariates: {args.use_covariates}"
    )
    logger.info("=" * 60)

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        micro_steps = 0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            loss = compute_loss(
                model, batch, device, num_output_patches,
                use_amp=use_amp,
            )
            loss_scaled = loss / args.grad_accum
            scaler.scale(loss_scaled).backward()
            epoch_loss += loss.item()
            micro_steps += 1

            if micro_steps % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Log
                if global_step % args.log_every == 0:
                    avg = epoch_loss / micro_steps
                    lr_now = scheduler.get_last_lr()[0]
                    elapsed = time.time() - t0
                    throughput = (
                        global_step * args.batch_size * args.grad_accum
                    ) / elapsed
                    logger.info(
                        f"[E{epoch+1}] step={global_step:5d}  "
                        f"loss={avg:.4f}  lr={lr_now:.2e}  "
                        f"throughput={throughput:.0f} samples/s"
                    )
                    metrics_log.append({
                        "step": global_step,
                        "epoch": epoch + 1,
                        "train_loss": avg,
                        "lr": lr_now,
                    })

                # Validate
                if global_step % args.val_every == 0:
                    val_metrics = validate(
                        model, pipeline, val_loader, device,
                        num_output_patches, args.prediction_len,
                        use_amp=use_amp,
                    )
                    val_loss = val_metrics["val_loss"]
                    cov80 = val_metrics["coverage80"]
                    ic_end = val_metrics["ic_end"]
                    q_method = val_metrics["quantile_method"]

                    logger.info(
                        f"  -- val_loss={val_loss:.4f}  "
                        f"coverage80={cov80:.3f}  ic_end={ic_end:.4f}  "
                        f"[{q_method}]"
                    )
                    if metrics_log:
                        metrics_log[-1].update({
                            "val_loss": val_loss,
                            "coverage80": cov80,
                            "ic_end": ic_end,
                            "quantile_method": q_method,
                        })

                    if val_loss < best_val:
                        best_val = val_loss
                        best_coverage = cov80
                        best_ic = ic_end
                        best_path = ckpt_dir / "best"
                        best_path.mkdir(exist_ok=True)
                        torch.save(
                            model.state_dict(), best_path / "model.pt"
                        )
                        logger.info(
                            f"  -- NEW BEST val_loss={best_val:.4f} "
                            f"-> {best_path}"
                        )

                # Checkpoint
                if global_step % args.ckpt_every == 0:
                    step_path = ckpt_dir / f"step-{global_step}"
                    step_path.mkdir(exist_ok=True)
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "scaler": scaler.state_dict(),
                            "global_step": global_step,
                            "best_val": best_val,
                            "rng_torch": torch.random.get_rng_state(),
                        },
                        step_path / "checkpoint.pt",
                    )

        # Handle partial accumulation at epoch end
        if micro_steps % args.grad_accum != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

        avg_epoch = epoch_loss / max(1, micro_steps)
        logger.info(
            f"Epoch {epoch+1} complete -- avg_loss={avg_epoch:.4f}"
        )

    # Finalize: save model, validate, and write manifest
    _finalize_training(
        model, pipeline, val_loader, device, num_output_patches,
        args, use_amp, run_id, run_dir, ckpt_dir, metrics_log,
        global_step, best_val, best_coverage, best_ic,
        target_modules, train_ds, val_ds, t0,
    )


# --- CLI --------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Chronos-2 Teacher Training (v2)"
    )

    # Standard defaults
    for key, default in DEFAULTS.items():
        arg_name = f"--{key.replace('_', '-')}"
        parser.add_argument(
            arg_name, type=type(default), default=default
        )

    # Preset
    parser.add_argument(
        "--preset",
        type=str,
        default="none",
        choices=["teacher10", "teacher30", "none"],
        help="Horizon preset: teacher10 (10d), teacher30 (30d), or none",
    )

    # Sampling & gating
    parser.add_argument(
        "--sampling-mode",
        type=str,
        default="random_anchors",
        choices=["sliding", "random_anchors"],
        help="Sampling mode for training windows",
    )
    parser.add_argument(
        "--mask-col",
        type=str,
        default="auto",
        choices=["is_tradable", "is_observable", "auto"],
        help="Universe gating column",
    )

    # Covariates
    parser.add_argument(
        "--use-covariates",
        action="store_true",
        default=False,
        help="Enable market covariates as past features",
    )
    parser.add_argument(
        "--covariates-path",
        type=str,
        default="backend/data/canonical/market_covariates.parquet",
        help="Path to canonical covariates parquet",
    )

    # LoRA FFN
    parser.add_argument(
        "--lora-ffn",
        action="store_true",
        default=False,
        help="Also apply LoRA to FFN projection modules",
    )

    # AMP
    parser.add_argument(
        "--amp",
        action="store_true",
        default=True,
        help="Enable AMP (default if CUDA)",
    )
    parser.add_argument(
        "--no-amp",
        dest="amp",
        action="store_false",
        help="Disable AMP",
    )

    args = parser.parse_args()

    # Apply preset overrides (user CLI flags take precedence)
    if args.preset != "none" and args.preset in PRESETS:
        preset = PRESETS[args.preset]
        # Only override if user did not explicitly set these
        for key, val in preset.items():
            cli_key = key.replace("_", "-")
            # Check if the arg was explicitly provided on CLI
            if f"--{cli_key}" not in sys.argv:
                setattr(args, key, val)
        logger.info(f"Applied preset '{args.preset}': {preset}")

    return args


# --- Shell snippet for multi-horizon runs ---
# Teacher 10d:
#   python backend/scripts/train_chronos2_teacher.py --preset teacher10 \
#       --sampling-mode random_anchors --use-covariates --epochs 3
#
# Teacher 30d:
#   python backend/scripts/train_chronos2_teacher.py --preset teacher30 \
#       --sampling-mode random_anchors --use-covariates --epochs 2


if __name__ == "__main__":
    args = parse_args()
    train(args)
