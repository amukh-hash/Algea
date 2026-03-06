"""
Sequence 4: Offline RL Policy — TD3 Behavioral Cloning
Target: cuda:0. Trains in ~5 minutes.

Synthesizes an "Oracle" dataset from historical VIX/SPY using hindsight-optimal
actions, then trains the RLStateProjector + TD3Actor via supervised MSE.

Action space: Tanh-bounded [-1, 1]² mapped to:
  act[0] → size_multiplier via (x+1)/2 → [0, 1]
  act[1] → veto flag (>0 = veto), abs(x) = confidence
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
logger = logging.getLogger("RL_BehavioralCloning")

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
OUT_DIR = Path("backend/artifacts/model_weights")


# ═══════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════

class OracleDataset(Dataset):
    def __init__(self, states, actions):
        self.states = torch.tensor(states, dtype=torch.float32)
        self.actions = torch.tensor(actions, dtype=torch.float32)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]


# ═══════════════════════════════════════════════════════════════════════════
# Oracle Environment: Hindsight-Optimal Actions
# ═══════════════════════════════════════════════════════════════════════════

def extract_oracle_environment():
    """Build state matrix and calculate hindsight-optimal actions from SPY/VIX."""
    import yfinance as yf

    logger.info("Extracting macro state environment from yfinance (2018-present)...")

    spy = yf.download("SPY", start="2018-01-01", end="2026-03-01", progress=False)["Close"]
    vix = yf.download("^VIX", start="2018-01-01", end="2026-03-01", progress=False)["Close"]

    if isinstance(spy, pd.DataFrame):
        spy = spy.iloc[:, 0]
    if isinstance(vix, pd.DataFrame):
        vix = vix.iloc[:, 0]

    df = pd.DataFrame({"SPY": spy, "VIX": vix}).dropna()
    logger.info("  Raw data: %d rows", len(df))

    # ── State features (backward-looking only, matches Orchestrator context) ──
    df["SPY_ret_1d"] = df["SPY"].pct_change()
    df["SPY_ret_5d"] = df["SPY"].pct_change(5)
    df["SPY_vol_21d"] = df["SPY_ret_1d"].rolling(21).std() * np.sqrt(252)
    df["VIX_change"] = df["VIX"].pct_change()
    df["VIX_level"] = df["VIX"] / 100.0  # Normalized
    df["Drift_Proxy"] = df["SPY_vol_21d"].rolling(60).mean()

    # ── Hindsight targets (future knowledge — training only) ──
    df["Future_5d_Ret"] = df["SPY"].shift(-5) / df["SPY"] - 1.0
    df["Future_5d_Min"] = df["SPY"].rolling(5).min().shift(-5) / df["SPY"] - 1.0

    df = df.dropna()
    logger.info("  Clean data: %d rows", len(df))

    feature_cols = ["SPY_ret_1d", "SPY_ret_5d", "SPY_vol_21d",
                    "VIX_change", "VIX_level", "Drift_Proxy"]
    states = df[feature_cols].values.astype(np.float32)

    # ── Calculate optimal actions in hindsight ──
    actions = np.zeros((len(df), 2), dtype=np.float32)

    for i in range(len(df)):
        fwd_ret = df["Future_5d_Ret"].iloc[i]
        fwd_dd = df["Future_5d_Min"].iloc[i]
        vix_lvl = df["VIX_level"].iloc[i] * 100.0

        # Risk Off (Veto): crash incoming or extreme VIX
        if fwd_dd < -0.03 or vix_lvl > 30.0:
            size, veto = 0.01, 1.0
        # Risk On: strong bull with no drawdown
        elif fwd_ret > 0.01 and fwd_dd > -0.01:
            size, veto = 1.0, 0.0
        # Neutral
        else:
            size, veto = 0.5, 0.0

        # Map [0,1] targets to [-1,1] Tanh space for the TD3Actor
        actions[i, 0] = (size * 2.0) - 1.0
        actions[i, 1] = (veto * 2.0) - 1.0

    n_veto = (actions[:, 1] > 0).sum()
    n_risk_on = (actions[:, 0] > 0.5).sum()
    logger.info("  Actions: %d veto days (%.1f%%), %d risk-on days (%.1f%%)",
                n_veto, 100 * n_veto / len(actions),
                n_risk_on, 100 * n_risk_on / len(actions))

    return states, actions, len(feature_cols)


# ═══════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════

def train_rl_policy():
    logger.info("=" * 60)
    logger.info("SEQUENCE 4: Offline RL Policy (Behavioral Cloning)")
    logger.info("=" * 60)

    from algae.models.rl.td3 import TD3Actor
    from backend.app.ml_platform.models.rl_policy.model import RLStateProjector

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    X_states, y_actions, raw_dim = extract_oracle_environment()

    # 80/20 chronological split
    split_idx = int(len(X_states) * 0.8)
    train_loader = DataLoader(
        OracleDataset(X_states[:split_idx], y_actions[:split_idx]),
        batch_size=64, shuffle=True, drop_last=True,
    )
    val_loader = DataLoader(
        OracleDataset(X_states[split_idx:], y_actions[split_idx:]),
        batch_size=128, shuffle=False,
    )

    # Build models
    projector = RLStateProjector(raw_dim=raw_dim, embed_dim=256).to(DEVICE, dtype=torch.bfloat16)
    actor = TD3Actor(state_dim=256, action_dim=2).to(DEVICE, dtype=torch.bfloat16)

    total_params = sum(p.numel() for p in projector.parameters()) + \
                   sum(p.numel() for p in actor.parameters())
    logger.info("  Projector + Actor: %s params", f"{total_params:,}")

    optimizer = torch.optim.AdamW(
        list(projector.parameters()) + list(actor.parameters()),
        lr=1e-3, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)
    criterion = nn.MSELoss()

    best_loss = float("inf")
    best_state = None

    for epoch in range(40):
        projector.train()
        actor.train()
        train_loss = 0.0

        for s, a in tqdm(train_loader, desc=f"RL Ep {epoch + 1:02d}/40", leave=False):
            s = s.to(DEVICE, dtype=torch.bfloat16)
            a = a.to(DEVICE, dtype=torch.bfloat16)

            optimizer.zero_grad()
            pred = actor(projector(s))
            loss = criterion(pred, a)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(projector.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        # Validation
        projector.eval()
        actor.eval()
        val_loss = 0.0
        veto_correct = 0
        veto_total = 0
        n_batches = 0

        with torch.inference_mode():
            for s, a in val_loader:
                s = s.to(DEVICE, dtype=torch.bfloat16)
                a = a.to(DEVICE, dtype=torch.bfloat16)
                pred = actor(projector(s))
                val_loss += criterion(pred, a).item()

                # Veto accuracy: did the model correctly predict veto?
                t_veto = a[:, 1] > 0.0
                p_veto = pred[:, 1] > 0.0
                if t_veto.sum() > 0:
                    veto_correct += (t_veto & p_veto).sum().item()
                    veto_total += t_veto.sum().item()
                n_batches += 1

        avg_val = val_loss / max(n_batches, 1)
        avg_train = train_loss / max(len(train_loader), 1)
        veto_acc = veto_correct / max(veto_total, 1)

        if avg_val < best_loss:
            best_loss = avg_val
            best_state = {
                "projector": {k: v.cpu() for k, v in projector.state_dict().items()},
                "actor": {k: v.cpu() for k, v in actor.state_dict().items()},
            }

        if epoch % 5 == 0 or epoch == 39:
            logger.info(
                "Epoch %02d | Train: %.4f | Val: %.4f | Veto Acc: %.1f%%",
                epoch + 1, avg_train, avg_val, veto_acc * 100,
            )

    # Save
    if best_state:
        torch.save(best_state, OUT_DIR / "td3_policy.pt")
        sz = (OUT_DIR / "td3_policy.pt").stat().st_size / 1024
        logger.info("TD3 Policy SAVED | Val MSE: %.4f | %.1f KB", best_loss, sz)

    # Save config manifest
    import json
    manifest = {"raw_dim": raw_dim, "embed_dim": 256, "action_dim": 2}
    with open(OUT_DIR / "td3_config.json", "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Sequence 4 COMPLETE. TD3 Risk Agent Armed.")


if __name__ == "__main__":
    train_rl_policy()
