"""
Institutional Optimization: Deep SMoE RankTransformer + Optuna + 5-Fold CV
Target: cuda:0 (RTX 3090 Ti). Estimated burn: ~4-8 hours.

Per-expert independent Optuna sweeps with architectural scaling:
  - d_model: [128, 256]
  - n_layers: [2, 4, 6]
  - n_heads: 4 (d_model=128) or 8 (d_model=256)
  - lr, weight_decay, dropout, pairwise margin

30 trials × 4 experts × 5 folds × 12 epochs = 7,200 training runs
Final production: 35 epochs with Optuna-optimal params per expert.
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
logger = logging.getLogger("SMoE_Optuna")

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DATA_DIR = Path("data_lake/smoe_training")
OUT_DIR = Path("backend/artifacts/model_weights")


# ═══════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════

class SMoEDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, mask: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.mask = torch.tensor(mask, dtype=torch.bool)  # True = padded

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.mask[idx]


# ═══════════════════════════════════════════════════════════════════════════
# Vectorized Pairwise Ranking Loss
# ═══════════════════════════════════════════════════════════════════════════

class PairwiseRankingLoss(nn.Module):
    def __init__(self, margin: float = 0.05):
        super().__init__()
        self.margin = margin

    def forward(self, preds, targets, pad_mask):
        diff_p = preds.unsqueeze(2) - preds.unsqueeze(1)
        diff_y = targets.unsqueeze(2) - targets.unsqueeze(1)
        target_greater = (diff_y > 0).float()
        loss = torch.relu(self.margin - diff_p)
        rank_diff = torch.abs(diff_y)
        valid = (~pad_mask).float()
        valid_pairs = valid.unsqueeze(2) * valid.unsqueeze(1)
        loss = loss * target_greater * valid_pairs * rank_diff
        num_valid = (target_greater * valid_pairs).sum()
        if num_valid > 0:
            return loss.sum() / num_valid
        return torch.tensor(0.0, device=preds.device, requires_grad=True)


# ═══════════════════════════════════════════════════════════════════════════
# Spearman IC
# ═══════════════════════════════════════════════════════════════════════════

def compute_batch_ic(preds, targets, mask):
    ics = []
    for b in range(preds.shape[0]):
        valid = ~mask[b]
        if valid.sum() > 2:
            p, t = preds[b][valid], targets[b][valid]
            if np.std(p) > 1e-6 and np.std(t) > 1e-6:
                corr, _ = spearmanr(p, t)
                if not np.isnan(corr):
                    ics.append(corr)
    return float(np.mean(ics)) if ics else 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Model Factory
# ═══════════════════════════════════════════════════════════════════════════

def build_model(d_model: int, n_layers: int, n_heads: int, dropout: float):
    """Build RankTransformer with the real API."""
    from algae.models.ranker.rank_transformer import RankTransformer

    model = RankTransformer(
        d_input=8,
        d_model=d_model,
        n_head=n_heads,
        n_layers=n_layers,
        dropout=dropout,
        max_len=512,
        pooling="none",
    ).to(DEVICE, dtype=torch.bfloat16)

    total = sum(p.numel() for p in model.parameters())
    logger.info("  Model: d=%d, L=%d, H=%d, drop=%.2f → %s params",
                d_model, n_layers, n_heads, dropout, f"{total:,}")
    return model


def run_forward(model, X_b, pad_mask_b):
    """Forward pass with real API: mask 1=valid, 0=padding."""
    X_b[pad_mask_b] = 0.0
    valid_mask = (~pad_mask_b).float()
    outputs = model(X_b, mask=valid_mask)
    # outputs["score"] is [B, N, 1]
    return outputs["score"].squeeze(-1).float()


# ═══════════════════════════════════════════════════════════════════════════
# Optuna Objective (per expert)
# ═══════════════════════════════════════════════════════════════════════════

def get_expert_objective(expert_id, X_raw, y_raw, mask_raw):
    """Returns an Optuna objective closure for the given expert."""

    def objective(trial):
        # Hyperparameter search space
        lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
        dropout = trial.suggest_float("dropout", 0.1, 0.4)
        margin = trial.suggest_float("margin", 0.01, 0.15)

        # Architectural scaling
        d_model = trial.suggest_categorical("d_model", [128, 256])
        n_layers = trial.suggest_int("n_layers", 2, 6, step=2)
        n_heads = 4 if d_model == 128 else 8

        # 5-fold walk-forward (gap=3 for 3-day forward target)
        tscv = TimeSeriesSplit(n_splits=5, gap=3)
        fold_ics = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_raw)):
            model = build_model(d_model, n_layers, n_heads, dropout)
            criterion = PairwiseRankingLoss(margin=margin)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr, weight_decay=weight_decay,
            )

            train_loader = DataLoader(
                SMoEDataset(X_raw[train_idx], y_raw[train_idx], mask_raw[train_idx]),
                batch_size=16, shuffle=True, drop_last=True,
            )
            val_loader = DataLoader(
                SMoEDataset(X_raw[val_idx], y_raw[val_idx], mask_raw[val_idx]),
                batch_size=32, shuffle=False,
            )

            best_fold_ic = -1.0

            for epoch in range(12):  # Fast burn for trajectory assessment
                model.train()
                for X_b, y_b, m_b in train_loader:
                    X_b = X_b.to(DEVICE, dtype=torch.bfloat16)
                    y_b = y_b.to(DEVICE, dtype=torch.float32)
                    m_b = m_b.to(DEVICE)

                    optimizer.zero_grad()
                    scores = run_forward(model, X_b, m_b)
                    loss = criterion(scores, y_b, m_b)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                model.eval()
                val_ics = []
                with torch.inference_mode():
                    for X_b, y_b, m_b in val_loader:
                        X_b = X_b.to(DEVICE, dtype=torch.bfloat16)
                        y_b = y_b.to(DEVICE, dtype=torch.float32)
                        m_b = m_b.to(DEVICE)
                        scores = run_forward(model, X_b, m_b)
                        ic = compute_batch_ic(
                            scores.cpu().numpy(), y_b.cpu().numpy(), m_b.cpu().numpy(),
                        )
                        val_ics.append(ic)

                avg_ic = float(np.mean(val_ics))
                if avg_ic > best_fold_ic:
                    best_fold_ic = avg_ic

            fold_ics.append(best_fold_ic)

            del model, optimizer, train_loader, val_loader
            torch.cuda.empty_cache()
            gc.collect()

        mean_ic = float(np.mean(fold_ics))
        logger.info(
            "Expert %d Trial %d | 5-Fold IC: %.4f | d=%d L=%d lr=%.2e wd=%.2e drop=%.2f margin=%.3f",
            expert_id, trial.number, mean_ic,
            d_model, n_layers, lr, weight_decay, dropout, margin,
        )
        return mean_ic

    return objective


# ═══════════════════════════════════════════════════════════════════════════
# Production Final Training
# ═══════════════════════════════════════════════════════════════════════════

def train_final_expert(expert_id, best_params, X_raw, y_raw, mask_raw):
    """Train production expert with Optuna-optimal params (35 epochs)."""
    logger.info("=" * 60)
    logger.info("PRODUCTION Expert %d: %s", expert_id, best_params)
    logger.info("=" * 60)

    # 80/20 chronological split + 3-day embargo
    split_idx = int(len(y_raw) * 0.8)
    embargo_idx = split_idx + 3

    train_loader = DataLoader(
        SMoEDataset(X_raw[:split_idx], y_raw[:split_idx], mask_raw[:split_idx]),
        batch_size=16, shuffle=True, drop_last=True,
    )
    val_loader = DataLoader(
        SMoEDataset(X_raw[embargo_idx:], y_raw[embargo_idx:], mask_raw[embargo_idx:]),
        batch_size=32, shuffle=False,
    )

    n_heads = 4 if best_params["d_model"] == 128 else 8
    model = build_model(
        best_params["d_model"], best_params["n_layers"],
        n_heads, best_params["dropout"],
    )
    criterion = PairwiseRankingLoss(margin=best_params["margin"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=best_params["lr"],
        weight_decay=best_params["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=35)

    best_ic = -1.0
    best_state = None

    for epoch in range(35):
        model.train()
        train_loss = 0.0
        for X_b, y_b, m_b in tqdm(
            train_loader,
            desc=f"Prod Exp {expert_id} Ep {epoch + 1:02d}/35",
            leave=False,
        ):
            X_b = X_b.to(DEVICE, dtype=torch.bfloat16)
            y_b = y_b.to(DEVICE, dtype=torch.float32)
            m_b = m_b.to(DEVICE)

            optimizer.zero_grad()
            scores = run_forward(model, X_b, m_b)
            loss = criterion(scores, y_b, m_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        model.eval()
        val_ics = []
        with torch.inference_mode():
            for X_b, y_b, m_b in val_loader:
                X_b = X_b.to(DEVICE, dtype=torch.bfloat16)
                y_b = y_b.to(DEVICE, dtype=torch.float32)
                m_b = m_b.to(DEVICE)
                scores = run_forward(model, X_b, m_b)
                val_ics.append(compute_batch_ic(
                    scores.cpu().numpy(), y_b.cpu().numpy(), m_b.cpu().numpy(),
                ))

        avg_ic = float(np.mean(val_ics))
        avg_loss = train_loss / max(len(train_loader), 1)
        logger.info("Epoch %02d | Loss: %.4f | Val IC: %.4f", epoch + 1, avg_loss, avg_ic)

        if avg_ic > best_ic:
            best_ic = avg_ic
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            logger.info("  >>> NEW BEST Expert %d (IC: %.4f)", expert_id, best_ic)

    # Save weights
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if best_state:
        torch.save(best_state, OUT_DIR / f"expert_{expert_id}.pt")

    # Save architecture manifest for live inference auto-detection
    manifest = {
        "d_input": 8,
        "d_model": best_params["d_model"],
        "n_head": n_heads,
        "n_layers": best_params["n_layers"],
        "dropout": best_params["dropout"],
        "max_len": 512,
    }
    with open(OUT_DIR / f"expert_{expert_id}_config.json", "w") as f:
        json.dump(manifest, f, indent=2)

    sz = (OUT_DIR / f"expert_{expert_id}.pt").stat().st_size / 1024
    logger.info(
        "Expert %d SAVED | IC: %.4f | %s (%.1f KB)",
        expert_id, best_ic, f"expert_{expert_id}.pt", sz,
    )

    del model, optimizer, train_loader, val_loader
    torch.cuda.empty_cache()
    gc.collect()

    return best_ic


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logger.info("Sequence 2.5: Deep SMoE Optuna — 30 trials × 4 experts × 5 folds")

    results = {}

    for expert_id in range(4):
        logger.info("\n" + "=" * 60)
        logger.info("OPTUNA STUDY: Expert %d (30 trials × 5 folds)", expert_id)
        logger.info("=" * 60)

        X_raw = np.load(DATA_DIR / f"X_expert_{expert_id}.npy")
        y_raw = np.load(DATA_DIR / f"y_expert_{expert_id}.npy")
        mask_raw = np.load(DATA_DIR / f"mask_expert_{expert_id}.npy")

        study = optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
        )
        study.optimize(
            get_expert_objective(expert_id, X_raw, y_raw, mask_raw),
            n_trials=30,
        )

        logger.info("Expert %d Optuna COMPLETE | Best 5-Fold IC: %.4f", expert_id, study.best_trial.value)
        logger.info("Expert %d Best Params: %s", expert_id, study.best_trial.params)

        # Train final production model with best params
        final_ic = train_final_expert(
            expert_id, study.best_trial.params, X_raw, y_raw, mask_raw,
        )
        results[expert_id] = {
            "optuna_ic": study.best_trial.value,
            "production_ic": final_ic,
            "params": study.best_trial.params,
        }

        # Free data arrays between experts
        del X_raw, y_raw, mask_raw
        gc.collect()

    logger.info("\n" + "=" * 60)
    logger.info("SEQUENCE 2.5: DEEP SMoE OPTIMIZATION COMPLETE")
    for eid, res in results.items():
        logger.info(
            "  Expert %d | Optuna IC: %.4f | Prod IC: %.4f | d=%d L=%d",
            eid, res["optuna_ic"], res["production_ic"],
            res["params"]["d_model"], res["params"]["n_layers"],
        )
    logger.info("=" * 60)
