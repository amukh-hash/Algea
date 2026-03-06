"""
Research Cycle 2: Macro-Aware Asymmetric RL Policy
Target: cuda:0. Burns ~5 mins.

Fixes two fundamental weaknesses:
1. Blindness: Expands from SPY/VIX-only to 10-feature macro state
   (adds TNX, UUP, HYG/LQD credit spread, 200d trend)
2. Apathy: AsymmetricRiskLoss with 15x False Negative penalty
   forces the network to respect crash days instead of ignoring them

Action space: Tanh-bounded [-1, 1]² →
  act[0] = size_multiplier via (x+1)/2 → [0, 1]
  act[1] = veto flag (>0 = veto), abs(x) = confidence
"""
import gc
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RL_Macro_Agent")

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
OUT_DIR = Path("backend/artifacts/model_weights")


# ═══════════════════════════════════════════════════════════════════════════
# Dataset & Loss
# ═══════════════════════════════════════════════════════════════════════════

class MacroOracleDataset(Dataset):
    def __init__(self, states, actions):
        self.states = torch.tensor(states, dtype=torch.float32)
        self.actions = torch.tensor(actions, dtype=torch.float32)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]


class AsymmetricRiskLoss(nn.Module):
    """MSE with 15x penalty on False Negatives (failing to veto a crash).

    Standard MSE optimizes 95% of Calm days, yielding lethal 27.5% Veto Accuracy.
    This loss forces the network to never miss a crash, even at the cost of
    occasional false alarms.
    """
    def __init__(self, fn_penalty=15.0):
        super().__init__()
        self.fn_penalty = fn_penalty
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, pred, target):
        loss_size = self.mse(pred[:, 0], target[:, 0]).mean()

        veto_err = self.mse(pred[:, 1], target[:, 1])
        is_crash = (target[:, 1] > 0.0).float()
        asymmetric_weight = 1.0 + (is_crash * (self.fn_penalty - 1.0))
        loss_veto = (veto_err * asymmetric_weight).mean()

        return loss_size + loss_veto


# ═══════════════════════════════════════════════════════════════════════════
# Oracle Environment: Multi-Asset Macro State
# ═══════════════════════════════════════════════════════════════════════════

def extract_macro_oracle():
    """Build 10-feature macro state from SPY/VIX/TNX/UUP/HYG/LQD."""
    import yfinance as yf

    logger.info("Extracting Multi-Asset Macro Environment (2016-present)...")
    tickers = ["SPY", "^VIX", "^TNX", "UUP", "HYG", "LQD"]
    df = yf.download(tickers, start="2016-01-01", end="2026-03-01", progress=False)["Close"]
    df = df.ffill().dropna()
    logger.info("  Raw data: %d rows × %d assets", len(df), len(df.columns))

    # ── 10 Macro State Features (backward-looking only) ──
    features = pd.DataFrame(index=df.index)
    features["SPY_ret_1d"] = df["SPY"].pct_change()
    features["SPY_ret_5d"] = df["SPY"].pct_change(5)
    features["SPY_vol_21d"] = features["SPY_ret_1d"].rolling(21).std() * np.sqrt(252)
    features["VIX_change"] = df["^VIX"].pct_change()
    features["VIX_level"] = df["^VIX"] / 100.0
    features["TNX_shock"] = df["^TNX"].diff(5) / 10.0  # 5-day yield shock
    features["USD_shock"] = df["UUP"].pct_change(5)    # 5-day dollar shock
    features["Credit_Spread"] = (df["HYG"] / df["LQD"]).pct_change(21)  # Junk vs IG
    features["Trend_200d"] = (df["SPY"] / df["SPY"].rolling(200).mean()) - 1.0
    features["Drift_Proxy"] = features["SPY_vol_21d"].rolling(60).mean()

    # ── Hindsight Targets (future knowledge — training only) ──
    targets = pd.DataFrame(index=df.index)
    targets["Future_5d_Ret"] = df["SPY"].shift(-5) / df["SPY"] - 1.0
    targets["Future_5d_Min"] = df["SPY"].rolling(5).min().shift(-5) / df["SPY"] - 1.0

    df_clean = pd.concat([features, targets], axis=1).dropna()
    states = df_clean.iloc[:, :10].values.astype(np.float32)
    logger.info("  Clean features: %d rows × %d features", len(states), states.shape[1])

    # ── Asymmetric Action Mapping ──
    actions = np.zeros((len(df_clean), 2), dtype=np.float32)
    for i in range(len(df_clean)):
        fwd_ret = df_clean["Future_5d_Ret"].iloc[i]
        fwd_dd = df_clean["Future_5d_Min"].iloc[i]
        vix_lvl = df_clean["VIX_level"].iloc[i] * 100.0

        if fwd_dd < -0.035 or vix_lvl > 30.0:
            size, veto = 0.01, 1.0  # CRASH
        elif fwd_ret > 0.01 and fwd_dd > -0.01:
            size, veto = 1.0, 0.0   # RALLY
        else:
            size, veto = 0.5, 0.0   # NEUTRAL

        actions[i, 0] = (size * 2.0) - 1.0  # Map [0,1] → [-1,1]
        actions[i, 1] = (veto * 2.0) - 1.0

    n_veto = (actions[:, 1] > 0).sum()
    logger.info("  Actions: %d veto days (%.1f%%), %d total days",
                n_veto, 100 * n_veto / len(actions), len(actions))

    return states, actions


# ═══════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════

def train():
    logger.info("=" * 60)
    logger.info("SEQUENCE 4R: Macro-Aware Asymmetric RL Policy")
    logger.info("  Loss: AsymmetricRiskLoss (15x FN penalty)")
    logger.info("  Features: 10 macro dimensions")
    logger.info("=" * 60)

    from algae.models.rl.td3 import TD3Actor
    from backend.app.ml_platform.models.rl_policy.model import RLStateProjector

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    X_states, y_actions = extract_macro_oracle()
    raw_dim = X_states.shape[1]  # Now exactly 10

    # 80/20 chronological split
    split_idx = int(len(X_states) * 0.8)
    train_loader = DataLoader(
        MacroOracleDataset(X_states[:split_idx], y_actions[:split_idx]),
        batch_size=64, shuffle=True, drop_last=True,
    )
    val_loader = DataLoader(
        MacroOracleDataset(X_states[split_idx:], y_actions[split_idx:]),
        batch_size=128, shuffle=False,
    )

    projector = RLStateProjector(raw_dim=raw_dim, embed_dim=256).to(DEVICE, dtype=torch.bfloat16)
    actor = TD3Actor(state_dim=256, action_dim=2).to(DEVICE, dtype=torch.bfloat16)

    total_params = sum(p.numel() for p in projector.parameters()) + \
                   sum(p.numel() for p in actor.parameters())
    logger.info("  Projector + Actor: %s params (raw_dim=%d)", f"{total_params:,}", raw_dim)

    optimizer = torch.optim.AdamW(
        list(projector.parameters()) + list(actor.parameters()),
        lr=1e-3, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    criterion = AsymmetricRiskLoss(fn_penalty=15.0)

    best_veto_acc = 0.0
    best_state = None

    for epoch in range(50):
        projector.train()
        actor.train()
        train_loss = 0.0

        for s, a in tqdm(train_loader, desc=f"RL Ep {epoch + 1:02d}/50", leave=False):
            s = s.to(DEVICE, dtype=torch.bfloat16)
            a = a.to(DEVICE, dtype=torch.bfloat16)
            optimizer.zero_grad()
            loss = criterion(actor(projector(s)), a)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(projector.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        # Validation
        projector.eval()
        actor.eval()
        veto_correct = 0
        total_crashes = 0
        val_loss = 0.0
        n_batches = 0

        with torch.inference_mode():
            for s, a in val_loader:
                s = s.to(DEVICE, dtype=torch.bfloat16)
                a = a.to(DEVICE, dtype=torch.bfloat16)
                pred = actor(projector(s))
                val_loss += criterion(pred, a).item()

                t_veto = a[:, 1] > 0.0
                p_veto = pred[:, 1] > 0.0
                total_crashes += t_veto.sum().item()
                if t_veto.sum() > 0:
                    veto_correct += (t_veto & p_veto).sum().item()
                n_batches += 1

        acc = veto_correct / max(1, total_crashes)
        avg_loss = val_loss / max(n_batches, 1)

        if acc > best_veto_acc:
            best_veto_acc = acc
            best_state = {
                "projector": {k: v.cpu() for k, v in projector.state_dict().items()},
                "actor": {k: v.cpu() for k, v in actor.state_dict().items()},
            }
            torch.save(best_state, OUT_DIR / "td3_policy.pt")

        if epoch % 5 == 0 or epoch == 49:
            logger.info(
                "Epoch %02d | Loss: %.4f | Crisis Veto Acc: %.1f%% (target: >80%%)",
                epoch + 1, avg_loss, acc * 100,
            )

    # Save config manifest
    import json
    manifest = {"raw_dim": raw_dim, "embed_dim": 256, "action_dim": 2,
                "features": ["SPY_ret_1d", "SPY_ret_5d", "SPY_vol_21d",
                             "VIX_change", "VIX_level", "TNX_shock",
                             "USD_shock", "Credit_Spread", "Trend_200d", "Drift_Proxy"]}
    with open(OUT_DIR / "td3_config.json", "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("=" * 60)
    logger.info("SEQUENCE 4R COMPLETE | Best Veto Acc: %.1f%%", best_veto_acc * 100)
    logger.info("=" * 60)


if __name__ == "__main__":
    train()
