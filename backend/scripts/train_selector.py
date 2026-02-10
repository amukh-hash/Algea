"""
Selector (rank transformer) training script.

Trains on the priors-frame produced by ``build_priors_frame.py``, using
time-based splits and ranking-aware objectives.

Usage
-----
::

    python backend/scripts/train_selector.py \\
        --priors-frame backend/data/selector/priors_frame \\
        --train-end 2021-12-31 --val-end 2023-12-31 \\
        --epochs 30 --lr 3e-4 --batch-size 8 --alpha 0.7 \\
        --target-horizon 5
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.stats import spearmanr
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from algaie.data.priors.selector_schema import MODEL_FEATURE_COLS
from algaie.training.selector_dataset import (
    SelectorDataset,
    make_time_split,
    selector_collate_fn,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Losses
# ═══════════════════════════════════════════════════════════════════════════

def pairwise_ranking_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    top_bottom_q: float = 0.2,
    max_pairs: int = 10_000,
    pairwise_mode: str = "uniform",
    max_pairs_per_symbol: int = 0,
    stratify_top_q: float = 0.05,
    stratify_bot_q: float = 0.05,
) -> torch.Tensor:
    """Pairwise logistic ranking loss within each date (batch element).

    Samples top-vs-bottom pairs from the ``top_bottom_q`` quantiles.
    **Does not build a full meshgrid** — samples indices directly.

    Parameters
    ----------
    scores : Tensor [B, N]
        Model scores.
    targets : Tensor [B, N]
        Ground-truth returns (y_ret).
    mask : Tensor [B, N]
        1 = valid ticker, 0 = padding.
    top_bottom_q : float
        Fraction of universe defining top/bottom sets (e.g. 0.2 = top/bottom 20%).
    max_pairs : int
        Maximum sampled pairs per date.  Default 10,000.
    pairwise_mode : str
        ``"uniform"`` — uniform random from top×bottom (default, backward compat).
        ``"stratified"`` — 50/50 mix of extreme-vs-extreme and broad top-vs-bottom.
    max_pairs_per_symbol : int
        Cap how many times any single symbol can appear on either side.
        0 = disabled (default).
    stratify_top_q : float
        Top fraction of *full universe* defining the "extreme top" subset
        for stratified mode (default 0.05).
    stratify_bot_q : float
        Bottom fraction of *full universe* defining the "extreme bottom" subset
        for stratified mode (default 0.05).
    """
    B = scores.shape[0]
    total_loss = torch.tensor(0.0, device=scores.device)
    valid = 0

    for b in range(B):
        m = mask[b].bool()
        if m.sum() < 4:
            continue
        s = scores[b][m]
        y = targets[b][m]
        N = s.shape[0]
        k = max(1, int(N * top_bottom_q))
        if k < 1:
            continue

        sorted_idx = torch.argsort(y, descending=True)
        top_idx = sorted_idx[:k]
        bot_idx = sorted_idx[-k:]

        if pairwise_mode == "stratified":
            # Extreme subsets: top of top, bottom of bottom
            k_ext_top = max(1, int(N * stratify_top_q))
            k_ext_bot = max(1, int(N * stratify_bot_q))
            top_ext = sorted_idx[:k_ext_top]
            bot_ext = sorted_idx[-k_ext_bot:]

            # 50/50 split: extreme pairs + broad pairs
            n_extreme = max_pairs // 2
            n_broad = max_pairs - n_extreme

            # Sample extreme pairs
            ext_i = top_ext[torch.randint(len(top_ext), (n_extreme,), device=s.device)]
            ext_j = bot_ext[torch.randint(len(bot_ext), (n_extreme,), device=s.device)]

            # Sample broad pairs
            broad_i = top_idx[torch.randint(len(top_idx), (n_broad,), device=s.device)]
            broad_j = bot_idx[torch.randint(len(bot_idx), (n_broad,), device=s.device)]

            pair_i = torch.cat([ext_i, broad_i])
            pair_j = torch.cat([ext_j, broad_j])
        else:
            # Uniform sampling — no meshgrid, direct index sampling
            pair_i = top_idx[torch.randint(len(top_idx), (max_pairs,), device=s.device)]
            pair_j = bot_idx[torch.randint(len(bot_idx), (max_pairs,), device=s.device)]

        # Per-symbol appearance cap (rejection sampling)
        if max_pairs_per_symbol > 0 and len(pair_i) > 0:
            cap = max_pairs_per_symbol
            counts = torch.zeros(N, dtype=torch.long, device=s.device)
            keep = torch.ones(len(pair_i), dtype=torch.bool, device=s.device)

            for idx in range(len(pair_i)):
                ci = pair_i[idx].item()
                cj = pair_j[idx].item()
                if counts[ci] >= cap or counts[cj] >= cap:
                    keep[idx] = False
                else:
                    counts[ci] += 1
                    counts[cj] += 1

            pair_i = pair_i[keep]
            pair_j = pair_j[keep]

        if len(pair_i) == 0:
            continue

        diff = s[pair_i] - s[pair_j]
        loss = F.softplus(-diff).mean()
        total_loss = total_loss + loss
        valid += 1

    return total_loss / max(1, valid)


# ═══════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════

def compute_spearman_ic(scores: np.ndarray, targets: np.ndarray) -> float:
    """Spearman rank correlation, handling NaN/constant arrays."""
    finite = np.isfinite(scores) & np.isfinite(targets)
    if finite.sum() < 3:
        return 0.0
    s, t = scores[finite], targets[finite]
    if np.std(s) < 1e-12 or np.std(t) < 1e-12:
        return 0.0
    ic, _ = spearmanr(s, t)
    return float(ic) if np.isfinite(ic) else 0.0


def compute_ndcg(scores: np.ndarray, targets: np.ndarray, k: int = 50) -> float:
    """NDCG@K for a single date."""
    n = len(scores)
    if n < 2:
        return 0.0
    k = min(k, n)
    # Relevance = targets shifted to be non-negative
    rel = targets - targets.min() + 1e-8

    # DCG of predicted ranking
    pred_order = np.argsort(-scores)
    dcg = sum(rel[pred_order[i]] / np.log2(i + 2) for i in range(k))

    # Ideal DCG
    ideal_order = np.argsort(-rel)
    idcg = sum(rel[ideal_order[i]] / np.log2(i + 2) for i in range(k))

    return float(dcg / max(idcg, 1e-12))


def compute_decile_spread(scores: np.ndarray, targets: np.ndarray) -> float:
    """Top minus bottom decile average realised return."""
    n = len(scores)
    if n < 10:
        return 0.0
    k = max(1, n // 10)
    order = np.argsort(-scores)
    top_ret = targets[order[:k]].mean()
    bot_ret = targets[order[-k:]].mean()
    return float(top_ret - bot_ret)


def compute_topk_return(scores: np.ndarray, targets: np.ndarray, k: int = 50) -> float:
    """Average realised return for top K stocks by predicted score."""
    k = min(k, len(scores))
    if k < 1:
        return 0.0
    order = np.argsort(-scores)
    return float(targets[order[:k]].mean())


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    ndcg_k: int = 50,
    topk: int = 50,
) -> Dict[str, float]:
    """Compute validation metrics for one epoch."""
    model.eval()
    all_ic = []
    all_ndcg = []
    all_spread = []
    all_topk_ret = []

    for batch in loader:
        X = batch["X"].to(device)
        y_ret = batch["y_ret"]
        mask = batch["mask"]

        out = model(X, mask.to(device))
        scores_t = out["score"].squeeze(-1).cpu()

        for b in range(scores_t.shape[0]):
            m = mask[b].numpy().astype(bool)
            if m.sum() < 10:
                continue
            s = scores_t[b].numpy()[m]
            y = y_ret[b].numpy()[m]
            all_ic.append(compute_spearman_ic(s, y))
            all_ndcg.append(compute_ndcg(s, y, k=ndcg_k))
            all_spread.append(compute_decile_spread(s, y))
            all_topk_ret.append(compute_topk_return(s, y, k=topk))

    n = max(len(all_ic), 1)
    return {
        "ic": float(np.mean(all_ic)) if all_ic else 0.0,
        "ndcg": float(np.mean(all_ndcg)) if all_ndcg else 0.0,
        "decile_spread": float(np.mean(all_spread)) if all_spread else 0.0,
        "topk_return": float(np.mean(all_topk_ret)) if all_topk_ret else 0.0,
        "n_dates": len(all_ic),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Linear ridge baseline (recommended by user for comparison)
# ═══════════════════════════════════════════════════════════════════════════

def run_ridge_baseline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    out_dir: Path,
):
    """Run a linear ridge regression baseline on the same splits.

    Reports IC, NDCG@50, and decile spread — a simple check against the
    RankTransformer. If competitive, a simpler model may be preferred.
    """
    from sklearn.linear_model import Ridge

    logger.info("=== Ridge Baseline ===")

    def _per_date_metrics(df, name):
        dates = df["date"].unique()
        ics, spreads, ndcgs = [], [], []
        for d in dates:
            sub = df[df["date"] == d]
            if len(sub) < 10:
                continue
            s = sub["pred"].values
            y = sub["y_ret"].values
            ics.append(compute_spearman_ic(s, y))
            spreads.append(compute_decile_spread(s, y))
            ndcgs.append(compute_ndcg(s, y, k=50))
        ic = float(np.mean(ics)) if ics else 0.0
        spread = float(np.mean(spreads)) if spreads else 0.0
        ndcg = float(np.mean(ndcgs)) if ndcgs else 0.0
        logger.info(f"  {name}: IC={ic:.4f}  spread={spread:.5f}  NDCG@50={ndcg:.4f}  ({len(ics)} dates)")
        return {"ic": ic, "spread": spread, "ndcg": ndcg, "n_dates": len(ics)}

    X_train = train_df[feature_cols].fillna(0).values
    y_train = train_df["y_ret"].values
    finite_mask = np.isfinite(y_train)
    X_train = X_train[finite_mask]
    y_train = y_train[finite_mask]

    if len(X_train) < 10:
        logger.warning("  Too few training samples for ridge baseline")
        return

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)

    results = {"model": "ridge", "alpha": 1.0}
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        if split_df.empty:
            continue
        X = split_df[feature_cols].fillna(0).values
        split_df = split_df.copy()
        split_df["pred"] = ridge.predict(X)
        metrics = _per_date_metrics(split_df, split_name)
        results[f"{split_name}_ic"] = metrics["ic"]
        results[f"{split_name}_spread"] = metrics["spread"]
        results[f"{split_name}_ndcg"] = metrics["ndcg"]

    # Save baseline metrics
    baseline_path = out_dir / "ridge_baseline.json"
    with open(baseline_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"  Ridge baseline saved: {baseline_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════════════════

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load priors frame (partitioned parquet)
    priors_path = Path(args.priors_frame)
    logger.info(f"Loading priors frame: {priors_path}")
    if priors_path.is_dir():
        parts = sorted(priors_path.glob("date=*/part.parquet"))
        if not parts:
            logger.error("No partitions found"); sys.exit(1)
        dfs = [pd.read_parquet(p) for p in parts]
        # Extract date from partition path
        for p, df in zip(parts, dfs):
            date_str = p.parent.name.replace("date=", "")
            df["date"] = pd.to_datetime(date_str)
        df_all = pd.concat(dfs, ignore_index=True)
    else:
        df_all = pd.read_parquet(priors_path)

    logger.info(f"Total rows: {len(df_all)}, dates: {df_all['date'].nunique()}")

    # Time split
    train_df, val_df, test_df = make_time_split(df_all, args.train_end, args.val_end)
    logger.info(f"Split: train={len(train_df)} ({train_df['date'].nunique()}d), "
                f"val={len(val_df)} ({val_df['date'].nunique()}d), "
                f"test={len(test_df)} ({test_df['date'].nunique()}d)")

    if len(train_df) == 0:
        logger.error("Empty training set"); sys.exit(1)

    # Feature columns
    feature_cols = list(MODEL_FEATURE_COLS)
    # Filter to columns that actually exist
    available = set(df_all.columns)
    feature_cols = [c for c in feature_cols if c in available]
    if not feature_cols:
        logger.error("No feature columns found"); sys.exit(1)
    logger.info(f"Feature dim: {len(feature_cols)}")

    # Ridge baseline (runs before RankTransformer train)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_baseline:
        run_ridge_baseline(train_df, val_df, test_df, feature_cols, out_dir)

    # Datasets
    train_ds = SelectorDataset(train_df, feature_cols=feature_cols)
    val_ds = SelectorDataset(val_df, feature_cols=feature_cols)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=selector_collate_fn, num_workers=0, pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=selector_collate_fn, num_workers=0, pin_memory=False,
    )

    # Model
    from algaie.models.ranker.rank_transformer import RankTransformer
    model = RankTransformer(
        d_input=len(feature_cols),
        d_model=args.d_model,
        n_head=args.n_head,
        n_layers=args.n_layers,
        dropout=args.dropout,
    ).to(device)
    logger.info(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Cosine LR with warmup
    total_steps = len(train_loader) * args.epochs
    warmup_steps = min(args.warmup_steps, total_steps // 5)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # AMP
    use_amp = args.amp and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp) if use_amp else None

    # Output dir (already created above for ridge baseline)
    metrics_path = out_dir / "metrics.jsonl"
    metrics_fh = open(metrics_path, "w")

    # Training state
    best_ic = -float("inf")
    ic_ema = 0.0
    ema_alpha = 0.3
    patience = args.patience
    patience_counter = 0
    alpha = args.alpha  # ranking vs regression balance

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_rank_loss = 0.0
        epoch_reg_loss = 0.0
        steps = 0
        t0 = time.time()

        for batch in train_loader:
            X = batch["X"].to(device)
            y_ret = batch["y_ret"].to(device)
            mask = batch["mask"].to(device)

            optimizer.zero_grad()

            with autocast(device_type=device.type, enabled=use_amp):
                out = model(X, mask)
                scores = out["score"].squeeze(-1)  # [B, N]

                # Ranking loss
                rank_loss = pairwise_ranking_loss(
                    scores, y_ret, mask,
                    top_bottom_q=args.top_bottom_q,
                    max_pairs=args.max_pairs,
                    pairwise_mode=args.pairwise_mode,
                    max_pairs_per_symbol=args.max_pairs_per_symbol,
                    stratify_top_q=args.stratify_top_q,
                    stratify_bot_q=args.stratify_bot_q,
                )

                # Regression loss
                reg_loss = F.smooth_l1_loss(
                    scores[mask], y_ret[mask], reduction="mean"
                )

                loss = alpha * rank_loss + (1 - alpha) * reg_loss

                # Optional risk head
                if args.risk_head and "risk" in out:
                    y_vol = batch["y_vol"].to(device)
                    risk_scores = out["risk"].squeeze(-1)
                    risk_loss = F.smooth_l1_loss(
                        risk_scores[mask], y_vol[mask], reduction="mean"
                    )
                    loss = loss + args.beta * risk_loss

            if use_amp and scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

            scheduler.step()
            epoch_loss += loss.item()
            epoch_rank_loss += rank_loss.item()
            epoch_reg_loss += reg_loss.item()
            steps += 1

        avg_loss = epoch_loss / max(1, steps)
        avg_rank = epoch_rank_loss / max(1, steps)
        avg_reg = epoch_reg_loss / max(1, steps)
        elapsed = time.time() - t0

        # Validate
        val_metrics = evaluate_epoch(model, val_loader, device, ndcg_k=50, topk=50)

        # EMA of IC for early stopping
        ic_ema = ema_alpha * val_metrics["ic"] + (1 - ema_alpha) * ic_ema

        # Log
        lr_now = optimizer.param_groups[0]["lr"]
        log_entry = {
            "epoch": epoch,
            "train_loss": round(avg_loss, 6),
            "rank_loss": round(avg_rank, 6),
            "reg_loss": round(avg_reg, 6),
            "val_ic": round(val_metrics["ic"], 4),
            "val_ic_ema": round(ic_ema, 4),
            "val_ndcg": round(val_metrics["ndcg"], 4),
            "val_decile_spread": round(val_metrics["decile_spread"], 6),
            "val_topk_return": round(val_metrics["topk_return"], 6),
            "val_n_dates": val_metrics["n_dates"],
            "lr": round(lr_now, 8),
            "elapsed_s": round(elapsed, 1),
        }
        metrics_fh.write(json.dumps(log_entry) + "\n")
        metrics_fh.flush()

        logger.info(
            f"Epoch {epoch}/{args.epochs} — loss={avg_loss:.5f} "
            f"(rank={avg_rank:.5f} reg={avg_reg:.5f}) | "
            f"val IC={val_metrics['ic']:.4f} IC_ema={ic_ema:.4f} "
            f"NDCG@50={val_metrics['ndcg']:.4f} "
            f"spread={val_metrics['decile_spread']:.5f} | "
            f"{elapsed:.1f}s"
        )

        # Checkpointing
        if val_metrics["ic"] > best_ic:
            best_ic = val_metrics["ic"]
            patience_counter = 0
            ckpt_path = out_dir / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_ic": best_ic,
                "feature_cols": feature_cols,
                "config": {
                    "d_input": len(feature_cols),
                    "d_model": args.d_model,
                    "n_head": args.n_head,
                    "n_layers": args.n_layers,
                    "dropout": args.dropout,
                    "alpha": args.alpha,
                },
            }, ckpt_path)
            logger.info(f"  ★ New best IC={best_ic:.4f}, saved {ckpt_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"  Early stopping at epoch {epoch} (patience={patience})")
                break

    # Final checkpoint
    final_path = out_dir / "final_model.pt"
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "val_ic": val_metrics["ic"],
        "feature_cols": feature_cols,
    }, final_path)

    metrics_fh.close()

    # Test set evaluation
    if len(test_df) > 0:
        test_ds = SelectorDataset(test_df, feature_cols=feature_cols)
        test_loader = DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False,
            collate_fn=selector_collate_fn, num_workers=0, pin_memory=False,
        )
        # Load best model
        best_ckpt = torch.load(out_dir / "best_model.pt", map_location=device,
                               weights_only=False)
        model.load_state_dict(best_ckpt["model_state_dict"])
        test_metrics = evaluate_epoch(model, test_loader, device)
        logger.info(
            f"Test: IC={test_metrics['ic']:.4f} NDCG@50={test_metrics['ndcg']:.4f} "
            f"spread={test_metrics['decile_spread']:.5f}"
        )
        test_entry = {"split": "test", **test_metrics}
        with open(metrics_path, "a") as f:
            f.write(json.dumps(test_entry) + "\n")

    logger.info(f"Done. Best val IC={best_ic:.4f}. Outputs: {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train selector model")
    parser.add_argument("--priors-frame", required=True, help="Path to priors_frame dir or parquet")
    parser.add_argument("--train-end", default="2021-12-31")
    parser.add_argument("--val-end", default="2023-12-31")
    parser.add_argument("--target-horizon", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=0.7, help="Ranking vs regression balance")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--risk-head", action="store_true", default=False)
    parser.add_argument("--beta", type=float, default=0.1, help="Risk head loss weight")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--skip-baseline", action="store_true", default=False,
                        help="Skip ridge regression baseline")
    # Pairwise loss params
    parser.add_argument("--pairwise-mode", choices=["uniform", "stratified"],
                        default="uniform", help="Pair sampling strategy")
    parser.add_argument("--max-pairs", type=int, default=10_000,
                        help="Max pairs per date (default 10000)")
    parser.add_argument("--top-bottom-q", type=float, default=0.2,
                        help="Fraction defining top/bottom quantile sets")
    parser.add_argument("--max-pairs-per-symbol", type=int, default=0,
                        help="Per-symbol appearance cap (0=disabled)")
    parser.add_argument("--stratify-top-q", type=float, default=0.05,
                        help="Extreme top fraction for stratified mode")
    parser.add_argument("--stratify-bot-q", type=float, default=0.05,
                        help="Extreme bottom fraction for stratified mode")
    args = parser.parse_args()

    if args.output_dir is None:
        ts = time.strftime("%Y%m%d-%H%M%S")
        args.output_dir = str(ROOT / "backend" / "data" / "selector" / "runs" / f"SEL-{ts}")

    train(args)


if __name__ == "__main__":
    main()
