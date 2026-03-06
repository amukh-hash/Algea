"""
Institutional Optimization: Chronos-2 Deep LoRA + Optuna + 5-Fold Purged CV
Target: cuda:0 (RTX 3090 Ti). Do not disrupt cuda:1 live inference.

4-Pillar Plan:
  1. Cross-asset multiplexed data (ES+NQ+RTY+YM, ~5,000 windows)
  2. Deep LoRA r=64, alpha=128 on ALL linear layers (q/k/v/o/wi/wo)
  3. 5-Fold Purged Walk-Forward CV (gap=128 embargo)
  4. Optuna HPO: 50 trials × 5 folds = 250 training runs (~12 hours)

Final output: 5 production ensemble adapters (chronos2_adapter_f{1-5}.pt)
"""
import gc
import logging
import os
import sys
from pathlib import Path

import numpy as np
import optuna
import torch
import torch.nn as nn
from sklearn.model_selection import TimeSeriesSplit
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Hardware Isolation: Protect cuda:1 live trading
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("Optuna_Chronos2")

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DATA_DIR = Path("data_lake/chronos2_training")
OUTPUT_DIR = Path("backend/artifacts/model_weights")


# ═══════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════

class CrossAssetGapDataset(Dataset):
    """Cross-asset overnight gap dataset. Each window is pure single-asset."""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.bfloat16)
        self.y = torch.tensor(y, dtype=torch.bfloat16)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Native Chronos-2 forward expects context [T] (2D when batched)
        return self.X[idx], self.y[idx]


# ═══════════════════════════════════════════════════════════════════════════
# Pinball (Quantile) Loss
# ═══════════════════════════════════════════════════════════════════════════

class PinballLoss(nn.Module):
    """Quantile (Pinball) Loss for P10/P50/P90 probabilistic forecasting."""
    def __init__(self, quantiles=(0.10, 0.50, 0.90)):
        super().__init__()
        self.register_buffer("quantiles", torch.tensor(quantiles).view(1, -1))

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if preds.ndim == 3:
            preds = preds.squeeze(1)
        target_exp = target.unsqueeze(1)
        err = target_exp - preds
        return torch.max(self.quantiles * err, (self.quantiles - 1.0) * err).mean()


# ═══════════════════════════════════════════════════════════════════════════
# Model Factory (Deep LoRA via load_chronos_adapter)
# ═══════════════════════════════════════════════════════════════════════════

def build_model(lora_dropout: float):
    """Load amazon/chronos-2 with Deep LoRA r=64 on all linear layers."""
    from algae.models.foundation.chronos2_teacher import load_chronos_adapter

    model_wrapper, info = load_chronos_adapter(
        model_id="amazon/chronos-2",
        use_qlora=False,
        device=DEVICE,
        lora_config={
            "rank": 64,
            "alpha": 128,
            "dropout": lora_dropout,
            # Deep LoRA: every linear layer in the T5 encoder
            # q/k/v/o = attention, wi/wo = feed-forward network
            "target_modules": ["q", "k", "v", "o", "wi", "wo"],
        },
        eval_mode=False,
    )

    # Enable quantile head and cast to bfloat16
    model_wrapper._enable_q10d_head = True
    if hasattr(model_wrapper, "quantile_head"):
        model_wrapper.quantile_head.to(dtype=torch.bfloat16)
        for param in model_wrapper.quantile_head.parameters():
            param.requires_grad = True

    model_wrapper.to(DEVICE)

    total = sum(p.numel() for p in model_wrapper.parameters())
    trainable = sum(p.numel() for p in model_wrapper.parameters() if p.requires_grad)
    logger.info(
        "Deep LoRA: %s / %s params trainable (%.1f%%)",
        f"{trainable:,}", f"{total:,}", 100 * trainable / total,
    )

    return model_wrapper


# ═══════════════════════════════════════════════════════════════════════════
# Training Loop for a Single Fold
# ═══════════════════════════════════════════════════════════════════════════

def train_fold(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    epochs: int,
    trial=None,
    fold_idx: int = 0,
):
    """Train one fold, return (best_val_loss, best_coverage_error)."""
    best_cov_err = float("inf")
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

            optimizer.zero_grad()

            # Native Chronos-2 forward: context [B, T], future_target [B, 1]
            future_target = y_batch.unsqueeze(-1)  # [B] → [B, 1]
            outputs = model(context=X_batch, future_target=future_target)

            # Extract NLL loss from native forward
            if hasattr(outputs, "loss") and outputs.loss is not None:
                loss = outputs.loss
            elif isinstance(outputs, dict) and "loss" in outputs:
                loss = outputs["loss"]
            else:
                if hasattr(outputs, "logits"):
                    pred = outputs.logits.mean(dim=-1).squeeze()
                elif isinstance(outputs, torch.Tensor):
                    pred = outputs.squeeze()
                else:
                    continue
                loss = nn.functional.huber_loss(pred, y_batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        # --- Validate (Coverage Ratio via generate) ---
        model.eval()
        val_loss = 0.0
        inside_bands = 0
        total_samples = 0

        with torch.inference_mode():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(DEVICE), y_val.to(DEVICE)

                # generate() expects [B, T, F] on CPU (pipeline pin_memory fix)
                context_3d = X_val.cpu().unsqueeze(-1).float()
                q_all = model.generate(
                    context=context_3d,
                    prediction_length=1,
                    num_samples=50,
                )
                # Output: [B, 1, 21, 1] → [B, 21]
                q_all = q_all.to(DEVICE).squeeze(-1).squeeze(1)
                if q_all.ndim == 1:
                    q_all = q_all.unsqueeze(0)

                n_q = q_all.shape[-1]
                if n_q >= 21:
                    q_out = torch.stack([
                        q_all[:, 2],   # ~P10
                        q_all[:, 10],  # ~P50
                        q_all[:, 18],  # ~P90
                    ], dim=-1).float()
                else:
                    q_out = torch.stack([
                        q_all[:, 0],
                        q_all[:, n_q // 2],
                        q_all[:, -1],
                    ], dim=-1).float()

                val_loss += criterion(q_out, y_val.float()).item()

                within = (y_val.float() >= q_out[:, 0]) & (y_val.float() <= q_out[:, 2])
                inside_bands += within.sum().item()
                total_samples += len(y_val)

        cov = inside_bands / max(total_samples, 1)
        cov_err = abs(0.80 - cov)
        avg_train = train_loss / max(len(train_loader), 1)
        avg_val = val_loss / max(len(val_loader), 1)

        if cov_err < best_cov_err:
            best_cov_err = cov_err
            best_val_loss = avg_val
            # Save state for production ensemble
            best_state = {
                k: v.cpu()
                for k, v in model.state_dict().items()
                if "lora_" in k.lower() or "quantile_head" in k.lower()
            }

        # Optuna pruning: kill hopeless trials early
        if trial is not None:
            trial.report(avg_val + cov_err * 200.0, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return best_val_loss, best_cov_err, best_state


# ═══════════════════════════════════════════════════════════════════════════
# Optuna Objective
# ═══════════════════════════════════════════════════════════════════════════

def objective(trial):
    """Optuna objective: minimize Pinball + 200× coverage penalty across 5 folds."""
    # 1. Hyperparameter Search Space
    lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    lora_dropout = trial.suggest_float("lora_dropout", 0.05, 0.3)

    X_raw = np.load(DATA_DIR / "X_multi.npy")
    y_raw = np.load(DATA_DIR / "y_multi.npy")

    # 2. Purged 5-Fold Walk-Forward Split
    # gap=128: 32 days × 4 assets → ensures strict zero temporal overlap
    tscv = TimeSeriesSplit(n_splits=5, gap=128)

    criterion = PinballLoss().to(DEVICE)
    fold_losses = []
    fold_coverages = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_raw)):
        logger.info(
            "Trial %d | Fold %d/5 | lr=%.2e wd=%.2e drop=%.3f | train=%d val=%d",
            trial.number, fold + 1, lr, weight_decay, lora_dropout,
            len(train_idx), len(val_idx),
        )

        model = build_model(lora_dropout)
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )

        train_loader = DataLoader(
            CrossAssetGapDataset(X_raw[train_idx], y_raw[train_idx]),
            batch_size=64, shuffle=True, drop_last=True,
        )
        val_loader = DataLoader(
            CrossAssetGapDataset(X_raw[val_idx], y_raw[val_idx]),
            batch_size=64, shuffle=False,
        )

        # 12 epochs per fold: enough for Optuna to gauge trajectory
        val_loss, cov_err, _ = train_fold(
            model, train_loader, val_loader, optimizer, criterion,
            epochs=12, trial=trial, fold_idx=fold,
        )

        fold_losses.append(val_loss)
        fold_coverages.append(cov_err)

        # Strict garbage collection between folds
        del model, optimizer, train_loader, val_loader
        torch.cuda.empty_cache()
        gc.collect()

    avg_loss = float(np.mean(fold_losses))
    avg_cov_err = float(np.mean(fold_coverages))

    # 3. Objective = Pinball Loss + 200× coverage penalty
    coverage_penalty = avg_cov_err * 200.0

    trial.set_user_attr("coverage_error", avg_cov_err)
    trial.set_user_attr("pinball_loss", avg_loss)

    score = avg_loss + coverage_penalty
    logger.info(
        "Trial %d COMPLETE | Score: %.4f | Pinball: %.4f | CovErr: %.4f",
        trial.number, score, avg_loss, avg_cov_err,
    )
    return score


# ═══════════════════════════════════════════════════════════════════════════
# Final Production Ensemble
# ═══════════════════════════════════════════════════════════════════════════

def train_final_ensemble(best_params: dict):
    """Train and save 5 production adapters using the Optuna-optimal hyperparams."""
    logger.info("=" * 70)
    logger.info("PRODUCTION ENSEMBLE: %s", best_params)
    logger.info("=" * 70)

    X_raw = np.load(DATA_DIR / "X_multi.npy")
    y_raw = np.load(DATA_DIR / "y_multi.npy")

    tscv = TimeSeriesSplit(n_splits=5, gap=128)
    criterion = PinballLoss().to(DEVICE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_raw)):
        logger.info("Production Fold %d/5 (train=%d, val=%d)",
                     fold + 1, len(train_idx), len(val_idx))

        model = build_model(best_params["lora_dropout"])
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=best_params["lr"],
            weight_decay=best_params["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)

        train_loader = DataLoader(
            CrossAssetGapDataset(X_raw[train_idx], y_raw[train_idx]),
            batch_size=64, shuffle=True, drop_last=True,
        )
        val_loader = DataLoader(
            CrossAssetGapDataset(X_raw[val_idx], y_raw[val_idx]),
            batch_size=64, shuffle=False,
        )

        # 25 epochs for production: fully settle into optimal basin
        _, best_cov_err, best_state = train_fold(
            model, train_loader, val_loader, optimizer, criterion,
            epochs=25,
        )
        scheduler.step()

        if best_state:
            save_path = OUTPUT_DIR / f"chronos2_adapter_f{fold + 1}.pt"
            torch.save(best_state, save_path)
            logger.info(
                "  [Fold %d] Saved %s (CovErr: %.3f, size: %.1f KB)",
                fold + 1, save_path.name, best_cov_err,
                save_path.stat().st_size / 1024,
            )
        else:
            logger.warning("  [Fold %d] No valid state to save!", fold + 1)

        del model, optimizer, train_loader, val_loader
        torch.cuda.empty_cache()
        gc.collect()

    logger.info("=" * 70)
    logger.info("PRODUCTION ENSEMBLE COMPLETE: 5 adapters saved to %s", OUTPUT_DIR)
    logger.info("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logger.info("Sequence 1 Deep Optimization: 50 Trials × 5 Folds = 250 Runs")
    logger.info("GPU: cuda:0 (RTX 3090 Ti), Data: %s", DATA_DIR)

    # Optuna study with TPE sampler and median pruning
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
    )
    study.optimize(objective, n_trials=50)

    logger.info("=" * 70)
    logger.info("OPTUNA SEARCH COMPLETE")
    logger.info("  Best Score:    %.6f", study.best_trial.value)
    logger.info("  Best Params:   %s", study.best_trial.params)
    logger.info("  Coverage Err:  %.4f", study.best_trial.user_attrs.get("coverage_error", -1))
    logger.info("  Pinball Loss:  %.4f", study.best_trial.user_attrs.get("pinball_loss", -1))
    logger.info("=" * 70)

    # Train final 5-fold production ensemble with best params
    train_final_ensemble(study.best_trial.params)

    logger.info("Sequence 1: Institutional Ensemble Generation Complete.")
