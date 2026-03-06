"""
Sequence 5 Training: VRP Spatial-Temporal Transformer
Target: cuda:0 (RTX 3090 Ti). Burns ~10-20 minutes.

Architecture: SpatialTemporalTransformer
  Input:  [B, Temporal=10, Spatial=125]  (5 tenors × 25 deltas flattened)
  Output: [B, 256] dense state embedding → regression head → VRP scalar

The ST-Transformer uses dual-attention (PatchTST variant):
  - Spatial attention: learns dependencies across delta buckets and tenors
  - Temporal attention: learns how the vol surface evolves over 10 days
  - Output: 256-dim embedding piped into a regression head

Target: VRP = VIX - Forward Realized Vol (scalar, in vol points)
  VRP > 0 → options overpriced → sell premium alpha
  VRP < 0 → options underpriced → buy protection
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
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("VRP_ST_Trainer")

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DATA_DIR = Path("data_lake/vrp_history")
OUT_DIR = Path("backend/artifacts/model_weights")


class VRPDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        # Flatten spatial dims: [N, 10, 5, 25] → [N, 10, 125]
        self.X = torch.tensor(
            X.reshape(X.shape[0], X.shape[1], -1), dtype=torch.float32,
        )
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train():
    from algae.models.tsfm.st_transformer import SpatialTemporalTransformer

    logger.info("=" * 60)
    logger.info("SEQUENCE 5: VRP Spatial-Temporal Transformer")
    logger.info("=" * 60)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    X_raw = np.load(DATA_DIR / "X_grid.npy")
    y_raw = np.load(DATA_DIR / "y_vrp.npy")
    logger.info("Data: X=%s  y=%s", X_raw.shape, y_raw.shape)
    logger.info("  VRP target: mean=%.2f std=%.2f range=[%.1f, %.1f]",
                y_raw.mean(), y_raw.std(), y_raw.min(), y_raw.max())

    # 80/20 chronological split with 21-day embargo
    split_idx = int(len(X_raw) * 0.80)
    embargo = 21
    X_train, y_train = X_raw[:split_idx], y_raw[:split_idx]
    X_val, y_val = X_raw[split_idx + embargo:], y_raw[split_idx + embargo:]
    logger.info("  Train: %d  Val: %d  (embargo=%d)", len(X_train), len(X_val), embargo)

    train_loader = DataLoader(
        VRPDataset(X_train, y_train), batch_size=64, shuffle=True, drop_last=True,
    )
    val_loader = DataLoader(
        VRPDataset(X_val, y_val), batch_size=128, shuffle=False,
    )

    # ST-Transformer: spatial_dim=125 (5×25 flattened), temporal_dim=10 (lookback)
    model = SpatialTemporalTransformer(
        spatial_dim=125,
        temporal_dim=10,
        d_model=128,
        nhead=4,
        num_layers=3,
    ).to(DEVICE, dtype=torch.bfloat16)

    # Regression head: 256 (ST-Transformer output) → scalar VRP prediction
    regression_head = nn.Sequential(
        nn.Linear(256, 64),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(64, 1),
    ).to(DEVICE, dtype=torch.bfloat16)

    total_params = sum(p.numel() for p in model.parameters()) + \
                   sum(p.numel() for p in regression_head.parameters())
    logger.info("  Model: %s params (ST d=128 L=3 H=4 + head)", f"{total_params:,}")

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(regression_head.parameters()),
        lr=3e-4, weight_decay=1e-3,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)
    criterion = nn.HuberLoss(delta=2.0)

    best_loss = float("inf")
    best_da = 0.0

    for epoch in range(40):
        model.train()
        regression_head.train()
        train_loss = 0.0

        for X_b, y_b in tqdm(train_loader, desc=f"VRP Ep {epoch + 1:02d}/40", leave=False):
            X_b = X_b.to(DEVICE, dtype=torch.bfloat16)
            y_b = y_b.to(DEVICE, dtype=torch.bfloat16)
            optimizer.zero_grad()

            # ST-Transformer: [B, 10, 125] → [B, 256]
            embeds = model(X_b)
            preds = regression_head(embeds).squeeze(-1)

            loss = criterion(preds, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(regression_head.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        regression_head.eval()
        val_loss = 0.0
        correct_dir = 0
        total = 0
        n_batches = 0

        with torch.inference_mode():
            for X_b, y_b in val_loader:
                X_b = X_b.to(DEVICE, dtype=torch.bfloat16)
                y_b = y_b.to(DEVICE, dtype=torch.bfloat16)
                embeds = model(X_b)
                preds = regression_head(embeds).squeeze(-1)
                val_loss += criterion(preds, y_b).item()

                # Directional accuracy: correctly predict VRP sign
                correct_dir += (torch.sign(preds) == torch.sign(y_b)).sum().item()
                total += len(y_b)
                n_batches += 1

        avg_loss = val_loss / max(n_batches, 1)
        da = correct_dir / max(total, 1)
        avg_train = train_loss / max(len(train_loader), 1)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_da = da
            torch.save({
                "transformer": {k: v.cpu() for k, v in model.state_dict().items()},
                "head": {k: v.cpu() for k, v in regression_head.state_dict().items()},
            }, OUT_DIR / "vrp_st_transformer.pt")

        if epoch % 5 == 0 or epoch == 39:
            logger.info(
                "Epoch %02d | Train: %.4f | Val Huber: %.4f | VRP Dir Acc: %.1f%%",
                epoch + 1, avg_train, avg_loss, da * 100,
            )

    # Save config manifest
    manifest = {
        "spatial_dim": 125,
        "temporal_dim": 10,
        "d_model": 128,
        "nhead": 4,
        "num_layers": 3,
        "output_dim": 256,
        "head_arch": "256→64→1",
        "best_huber": float(best_loss),
        "best_da": float(best_da),
    }
    with open(OUT_DIR / "vrp_config.json", "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("=" * 60)
    logger.info("SEQUENCE 5 COMPLETE | Best Huber: %.4f | DA: %.1f%%",
                best_loss, best_da * 100)
    logger.info("=" * 60)


if __name__ == "__main__":
    train()
