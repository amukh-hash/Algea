"""
NO COMPROMISE SMoE OPTIMIZATION: Transfer Learning + Noise Augmentation
Target: cuda:0 (RTX 3090 Ti). Expected Burn: 8-14 Hours.

Phase 1: Pre-train a global RankTransformer (d=256, L=6) on ALL 500 stocks
Phase 2: Fine-tune 4 experts by freezing bottom 4 layers, training top 2 + heads
         with Gaussian noise injection for augmentation
"""
import gc
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SMoE_Transfer")

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DATA_DIR = Path("data_lake/smoe_training")
OUT_DIR = Path("backend/artifacts/model_weights")

# Frozen layers during fine-tuning (bottom 4 of 6)
FROZEN_LAYERS = {"encoder.layers.0", "encoder.layers.1",
                 "encoder.layers.2", "encoder.layers.3"}


# ═══════════════════════════════════════════════════════════════════════════
# Dataset & Loss
# ═══════════════════════════════════════════════════════════════════════════

class SMoEDataset(Dataset):
    def __init__(self, X, y, mask):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.mask = torch.tensor(mask, dtype=torch.bool)  # True = padded

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.mask[idx]


class PairwiseRankingLoss(nn.Module):
    def __init__(self, margin=0.05):
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


def build_model(d_input=8):
    """Build d=256, L=6 RankTransformer with the real API."""
    from algae.models.ranker.rank_transformer import RankTransformer
    model = RankTransformer(
        d_input=d_input, d_model=256, n_head=8, n_layers=6,
        dropout=0.15, max_len=512, pooling="none",
    ).to(DEVICE, dtype=torch.bfloat16)
    total = sum(p.numel() for p in model.parameters())
    logger.info("  Model: d=256, L=6, H=8, drop=0.15 → %s params", f"{total:,}")
    return model


def run_forward(model, X_b, pad_mask_b):
    """Forward with real API: mask 1=valid, 0=padding, extract score."""
    X_b[pad_mask_b] = 0.0
    valid_mask = (~pad_mask_b).float()
    outputs = model(X_b, mask=valid_mask)
    return outputs["score"].squeeze(-1).float()


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1: Global Foundation Pre-Training
# ═══════════════════════════════════════════════════════════════════════════

def train_global_foundation():
    logger.info("=" * 60)
    logger.info("PHASE 1: Pre-Training Global Foundation Ranker")
    logger.info("  500 stocks × 2028 days → universal cross-sectional physics")
    logger.info("=" * 60)

    X_global = np.load(DATA_DIR / "X_global.npy")
    y_global = np.load(DATA_DIR / "y_global.npy")
    mask_global = np.load(DATA_DIR / "mask_global.npy")
    logger.info("  Data: X=%s y=%s mask=%s", X_global.shape, y_global.shape, mask_global.shape)

    # 85/15 chronological split + 3-day embargo
    split_idx = int(len(y_global) * 0.85)
    embargo_idx = split_idx + 3

    train_loader = DataLoader(
        SMoEDataset(X_global[:split_idx], y_global[:split_idx], mask_global[:split_idx]),
        batch_size=16, shuffle=True, drop_last=True,
    )
    val_loader = DataLoader(
        SMoEDataset(X_global[embargo_idx:], y_global[embargo_idx:], mask_global[embargo_idx:]),
        batch_size=32, shuffle=False,
    )

    model = build_model(d_input=X_global.shape[-1])
    criterion = PairwiseRankingLoss(margin=0.08)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)

    best_ic = -1.0

    for epoch in range(40):
        model.train()
        train_loss = 0.0

        for X_b, y_b, m_b in tqdm(
            train_loader,
            desc=f"Global Ep {epoch + 1:02d}/40",
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
        logger.info("Global Ep %02d | Loss: %.4f | Val IC: %.4f", epoch + 1, avg_loss, avg_ic)

        if avg_ic > best_ic:
            best_ic = avg_ic
            torch.save(model.state_dict(), OUT_DIR / "base_ranker.pt")
            logger.info("  >>> NEW BEST Foundation (IC: %.4f)", best_ic)

    logger.info("Global Foundation COMPLETE | Best IC: %.4f", best_ic)
    del model, optimizer, train_loader, val_loader
    torch.cuda.empty_cache()
    gc.collect()
    return best_ic


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2: Expert Fine-Tuning (Freeze Bottom 4, Noise Augmentation)
# ═══════════════════════════════════════════════════════════════════════════

def finetune_expert(expert_id: int):
    logger.info("=" * 60)
    logger.info("PHASE 2: Fine-Tuning Expert %d", expert_id)
    logger.info("  Frozen: encoder.layers.{0-3} | Trainable: layers.{4-5} + heads")
    logger.info("=" * 60)

    X_raw = np.load(DATA_DIR / f"X_expert_{expert_id}.npy")
    y_raw = np.load(DATA_DIR / f"y_expert_{expert_id}.npy")
    mask_raw = np.load(DATA_DIR / f"mask_expert_{expert_id}.npy")
    d_input = X_raw.shape[-1]
    logger.info("  Data: X=%s", X_raw.shape)

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

    # 1. Build model matching foundation architecture
    model = build_model(d_input=d_input)

    # 2. Load foundation weights
    state = torch.load(OUT_DIR / "base_ranker.pt", map_location=DEVICE, weights_only=True)
    model.load_state_dict(state, strict=True)
    logger.info("  Loaded foundation weights from base_ranker.pt")

    # 3. Freeze bottom 4 encoder layers (universal feature extractors)
    frozen_count = 0
    trainable_count = 0
    for name, param in model.named_parameters():
        if any(name.startswith(layer) for layer in FROZEN_LAYERS):
            param.requires_grad = False
            frozen_count += param.numel()
        else:
            param.requires_grad = True
            trainable_count += param.numel()

    logger.info("  Frozen params: %s | Trainable: %s",
                f"{frozen_count:,}", f"{trainable_count:,}")

    # 4. Dynamic margin by expert type
    expert_margins = {0: 0.08, 1: 0.08, 2: 0.08, 3: 0.129}
    margin = expert_margins.get(expert_id, 0.08)
    logger.info("  Pairwise margin: %.3f", margin)

    criterion = PairwiseRankingLoss(margin=margin)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=3e-5, weight_decay=1e-2,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    best_ic = -1.0
    best_state = None

    for epoch in range(30):
        model.train()
        train_loss = 0.0

        for X_b, y_b, m_b in tqdm(
            train_loader,
            desc=f"Expert {expert_id} Ep {epoch + 1:02d}/30",
            leave=False,
        ):
            X_b = X_b.to(DEVICE, dtype=torch.bfloat16)
            y_b = y_b.to(DEVICE, dtype=torch.float32)
            m_b = m_b.to(DEVICE)

            # DATA AUGMENTATION: Gaussian noise on valid (non-padded) features
            noise = torch.randn_like(X_b) * 0.05
            X_b[~m_b] = X_b[~m_b] + noise[~m_b]

            optimizer.zero_grad()
            scores = run_forward(model, X_b, m_b)
            loss = criterion(scores, y_b, m_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        # Validation (no noise)
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
        logger.info("Expert %d Ep %02d | Loss: %.4f | Val IC: %.4f",
                    expert_id, epoch + 1, avg_loss, avg_ic)

        if avg_ic > best_ic:
            best_ic = avg_ic
            # Save full state (including frozen layers) for inference
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            logger.info("  >>> NEW BEST Expert %d (IC: %.4f)", expert_id, best_ic)

    # Save weights + config manifest
    if best_state:
        torch.save(best_state, OUT_DIR / f"expert_{expert_id}.pt")

    manifest = {
        "d_input": d_input, "d_model": 256, "n_head": 8,
        "n_layers": 6, "dropout": 0.15, "max_len": 512,
    }
    with open(OUT_DIR / f"expert_{expert_id}_config.json", "w") as f:
        json.dump(manifest, f, indent=2)

    sz = (OUT_DIR / f"expert_{expert_id}.pt").stat().st_size / 1024
    logger.info("Expert %d SAVED | IC: %.4f | %.1f KB", expert_id, best_ic, sz)

    del model, optimizer, train_loader, val_loader
    torch.cuda.empty_cache()
    gc.collect()
    return best_ic


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Sequence 2.5R: Transfer Learning — Global Pre-Train + Expert Fine-Tune")

    # Phase 1: Pre-train foundation
    if not (OUT_DIR / "base_ranker.pt").exists():
        foundation_ic = train_global_foundation()
    else:
        logger.info("Foundation already trained — skipping Phase 1")

    # Phase 2: Fine-tune experts
    results = {}
    for i in range(4):
        ic = finetune_expert(i)
        results[i] = ic

    logger.info("=" * 60)
    logger.info("SEQUENCE 2.5R TRANSFER LEARNING COMPLETE")
    for eid, ic in results.items():
        logger.info("  Expert %d | Prod IC: %.4f", eid, ic)
    logger.info("=" * 60)
