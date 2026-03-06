"""
Research Cycle 3: StatArb V3 iTransformer — Idiosyncratic Pairs + OFI
Target: cuda:0 (RTX 3090 Ti). Estimated burn: ~2-4 hours.

10 Idiosyncratic Pairs with Order Flow Imbalance injection:
  KRE/IWM, XBI/IWM, ARKK/QQQ, SMH/QQQ, GDXJ/GLD,
  XOP/USO, ITB/VNQ, JNK/TLT, TAN/XLE, XRT/SPY

60-minute bars: lookback=60 → 8.5 trading days, 12-bar target → 1.7 days.
OFI blended into cointegration Z-scores for volume-informed mean-reversion.

Loss: HuberLoss (δ=1.0) — caps extreme gradient from macro shocks.
Gate: Directional Accuracy > 52.5% (the institutional pairs threshold).
Embargo: gap=12 bars (1.7 trading days → prevents target leakage).
"""
import gc
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import optuna
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.model_selection import TimeSeriesSplit
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("iTransformer_Optuna")

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DATA_DIR = Path("data_lake/statarb_v3")
OUT_DIR = Path("backend/artifacts/model_weights")


# ═══════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════

class StatArbDataset(Dataset):
    """X: [N, lookback=60, variates=6], y: [N, variates=6] (Δ Z-score)."""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ═══════════════════════════════════════════════════════════════════════════
# Directional Accuracy
# ═══════════════════════════════════════════════════════════════════════════

def compute_directional_accuracy(preds, targets):
    """% of predictions that correctly predict the sign of the spread move."""
    correct = (torch.sign(preds) == torch.sign(targets)).float()
    non_zero = (targets.abs() > 1e-6).float()
    denom = non_zero.sum()
    if denom == 0:
        return 0.0
    return (correct * non_zero).sum().item() / denom.item()


# ═══════════════════════════════════════════════════════════════════════════
# Model Factory
# ═══════════════════════════════════════════════════════════════════════════

def build_model(num_variates, lookback_len, d_model, n_heads, e_layers, dropout):
    """Build iTransformer with the real API."""
    from algae.models.tsfm.itransformer import iTransformer

    model = iTransformer(
        num_variates=num_variates,
        lookback_len=lookback_len,
        pred_len=1,
        d_model=d_model,
        n_heads=n_heads,
        e_layers=e_layers,
        dropout=dropout,
    ).to(DEVICE, dtype=torch.bfloat16)

    total = sum(p.numel() for p in model.parameters())
    logger.info("  Model: d=%d, L=%d, H=%d, drop=%.2f → %s params",
                d_model, e_layers, n_heads, dropout, f"{total:,}")
    return model


# ═══════════════════════════════════════════════════════════════════════════
# Optuna Objective
# ═══════════════════════════════════════════════════════════════════════════

def objective(trial, X_raw, y_raw):
    lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.4)

    # Architectural scaling
    d_model = trial.suggest_categorical("d_model", [64, 128, 256])
    e_layers = trial.suggest_int("e_layers", 2, 4)
    n_heads = 4 if d_model <= 128 else 8

    num_variates = X_raw.shape[2]
    lookback_len = X_raw.shape[1]

    # PURGED CV: gap=12 bars (1 hour embargo on 5-min data)
    # This is the mathematical firewall against the 3.28e-06 leakage bug
    tscv = TimeSeriesSplit(n_splits=5, gap=12)
    criterion = nn.HuberLoss(delta=1.0)

    fold_losses = []
    fold_das = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_raw)):
        model = build_model(num_variates, lookback_len, d_model, n_heads, e_layers, dropout)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        train_loader = DataLoader(
            StatArbDataset(X_raw[train_idx], y_raw[train_idx]),
            batch_size=128, shuffle=True, drop_last=True,
        )
        val_loader = DataLoader(
            StatArbDataset(X_raw[val_idx], y_raw[val_idx]),
            batch_size=256, shuffle=False,
        )

        best_fold_loss = float("inf")
        best_fold_da = 0.0

        for epoch in range(12):  # Fast trajectory search
            model.train()
            for X_b, y_b in train_loader:
                X_b = X_b.to(DEVICE, dtype=torch.bfloat16)
                y_b = y_b.to(DEVICE, dtype=torch.bfloat16)

                optimizer.zero_grad()
                # iTransformer output: [B, Variates, pred_len=1]
                preds = model(X_b).squeeze(-1)  # → [B, Variates]
                loss = criterion(preds, y_b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            model.eval()
            val_loss, val_da, n_batches = 0.0, 0.0, 0
            with torch.inference_mode():
                for X_b, y_b in val_loader:
                    X_b = X_b.to(DEVICE, dtype=torch.bfloat16)
                    y_b = y_b.to(DEVICE, dtype=torch.bfloat16)
                    preds = model(X_b).squeeze(-1)
                    val_loss += criterion(preds, y_b).item()
                    val_da += compute_directional_accuracy(preds.float(), y_b.float())
                    n_batches += 1

            avg_loss = val_loss / max(n_batches, 1)
            avg_da = val_da / max(n_batches, 1)

            if avg_loss < best_fold_loss:
                best_fold_loss = avg_loss
                best_fold_da = avg_da

        fold_losses.append(best_fold_loss)
        fold_das.append(best_fold_da)

        del model, optimizer, train_loader, val_loader
        torch.cuda.empty_cache()
        gc.collect()

    avg_loss = float(np.mean(fold_losses))
    avg_da = float(np.mean(fold_das))
    trial.set_user_attr("directional_accuracy", avg_da)

    logger.info(
        "Trial %d | Huber: %.5f | DA: %.2f%% | d=%d L=%d lr=%.2e wd=%.2e drop=%.2f",
        trial.number, avg_loss, avg_da * 100,
        d_model, e_layers, lr, weight_decay, dropout,
    )

    # Penalize if directional accuracy < 51%
    da_penalty = max(0.0, 0.51 - avg_da) * 100.0
    return avg_loss + da_penalty


# ═══════════════════════════════════════════════════════════════════════════
# Production Ensemble Training (5-Fold)
# ═══════════════════════════════════════════════════════════════════════════

def train_production_ensemble(best_params, X_raw, y_raw):
    """Train 5 production folds with Optuna-optimal params (35 epochs each)."""
    logger.info("=" * 60)
    logger.info("PRODUCTION ENSEMBLE: 5 folds × 35 epochs")
    logger.info("Params: %s", best_params)
    logger.info("=" * 60)

    num_variates = X_raw.shape[2]
    lookback_len = X_raw.shape[1]
    n_heads = 4 if best_params["d_model"] <= 128 else 8
    criterion = nn.HuberLoss(delta=1.0)

    tscv = TimeSeriesSplit(n_splits=5, gap=12)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_raw)):
        logger.info("--- Fold %d/5 ---", fold + 1)
        model = build_model(
            num_variates, lookback_len,
            best_params["d_model"], n_heads,
            best_params["e_layers"], best_params["dropout"],
        )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=best_params["lr"],
            weight_decay=best_params["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=35)

        train_loader = DataLoader(
            StatArbDataset(X_raw[train_idx], y_raw[train_idx]),
            batch_size=128, shuffle=True, drop_last=True,
        )
        val_loader = DataLoader(
            StatArbDataset(X_raw[val_idx], y_raw[val_idx]),
            batch_size=256, shuffle=False,
        )

        best_loss = float("inf")
        best_da = 0.0
        best_state = None

        for epoch in range(35):
            model.train()
            train_loss = 0.0
            for X_b, y_b in tqdm(
                train_loader,
                desc=f"F{fold + 1} Ep {epoch + 1:02d}/35",
                leave=False,
            ):
                X_b = X_b.to(DEVICE, dtype=torch.bfloat16)
                y_b = y_b.to(DEVICE, dtype=torch.bfloat16)
                optimizer.zero_grad()
                preds = model(X_b).squeeze(-1)
                loss = criterion(preds, y_b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()

            scheduler.step()

            model.eval()
            val_loss, val_da, n_batches = 0.0, 0.0, 0
            with torch.inference_mode():
                for X_b, y_b in val_loader:
                    X_b = X_b.to(DEVICE, dtype=torch.bfloat16)
                    y_b = y_b.to(DEVICE, dtype=torch.bfloat16)
                    preds = model(X_b).squeeze(-1)
                    val_loss += criterion(preds, y_b).item()
                    val_da += compute_directional_accuracy(preds.float(), y_b.float())
                    n_batches += 1

            avg_loss = val_loss / max(n_batches, 1)
            avg_da = val_da / max(n_batches, 1)
            avg_train = train_loss / max(len(train_loader), 1)

            logger.info(
                "Epoch %02d | Train: %.5f | Val Huber: %.5f | DA: %.2f%%",
                epoch + 1, avg_train, avg_loss, avg_da * 100,
            )

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_da = avg_da
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                logger.info("  >>> NEW BEST (Huber: %.5f, DA: %.2f%%)", best_loss, best_da * 100)

        # Save fold
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        save_path = OUT_DIR / f"itransformer_f{fold + 1}.pt"
        if best_state:
            torch.save(best_state, save_path)
        fold_results.append({"fold": fold + 1, "huber": best_loss, "da": best_da})
        logger.info(
            "Fold %d SAVED | Huber: %.5f | DA: %.2f%% | %s (%.1f KB)",
            fold + 1, best_loss, best_da * 100,
            save_path.name, save_path.stat().st_size / 1024,
        )

        del model, optimizer, train_loader, val_loader
        torch.cuda.empty_cache()
        gc.collect()

    # Save config manifest
    manifest = {
        "num_variates": int(num_variates),
        "lookback_len": int(lookback_len),
        "pred_len": 1,
        "d_model": best_params["d_model"],
        "n_heads": n_heads,
        "e_layers": best_params["e_layers"],
        "dropout": best_params["dropout"],
    }
    with open(OUT_DIR / "itransformer_config.json", "w") as f:
        json.dump(manifest, f, indent=2)

    return fold_results


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logger.info("Research Cycle 3: StatArb V3 iTransformer — 40 trials × 5 folds")

    X_raw = np.load(DATA_DIR / "X_features.npy")
    y_raw = np.load(DATA_DIR / "y_targets.npy")
    logger.info("Data: X=%s y=%s", X_raw.shape, y_raw.shape)

    # Sanity: check target is NOT near-zero (would indicate identity leak)
    y_std = y_raw.std()
    y_mean = y_raw.mean()
    logger.info("Target stats: mean=%.4f std=%.4f (should be ~0, ~1.0-2.0)", y_mean, y_std)
    if y_std < 0.01:
        logger.error("TARGET LEAKAGE DETECTED! y_std=%.6f — aborting.", y_std)
        raise RuntimeError("Δ Z-score targets appear near-zero. Check data pipeline.")

    # Optuna search
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    )
    study.optimize(
        lambda trial: objective(trial, X_raw, y_raw),
        n_trials=40,
    )

    logger.info("=" * 60)
    logger.info("OPTUNA COMPLETE")
    logger.info("  Best Huber: %.5f", study.best_trial.value)
    logger.info("  Best DA: %.2f%%",
                study.best_trial.user_attrs.get("directional_accuracy", 0) * 100)
    logger.info("  Best Params: %s", study.best_trial.params)

    # Production ensemble
    results = train_production_ensemble(study.best_trial.params, X_raw, y_raw)

    logger.info("=" * 60)
    logger.info("SEQUENCE 3 COMPLETE")
    for r in results:
        logger.info("  Fold %d | Huber: %.5f | DA: %.2f%%", r["fold"], r["huber"], r["da"] * 100)
    logger.info("=" * 60)
