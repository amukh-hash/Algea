"""
Walk-forward evaluation for the selector pipeline.

Trains MLPSelector (or RankTransformer) on rolling windows and evaluates
on successive test periods, producing a fold_summary.csv that shows
generalization stability across regimes.

Usage
-----
::

    python backend/scripts/walkforward_eval.py \\
        --priors-frame backend/data/selector/priors_frame \\
        --output-dir backend/data/selector/walkforward \\
        --train-months 24 --val-months 3 --test-months 3 --step-months 1 \\
        --max-epochs 10 --patience 5
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dateutil.relativedelta import relativedelta
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from algaie.data.priors.selector_schema import MODEL_FEATURE_COLS
from algaie.eval.selector_metrics import (
    compute_bucketed_metrics,
    compute_priors_baseline_score,
    per_date_metrics,
)
from algaie.training.selector_dataset import (
    SelectorDataset,
    selector_collate_fn,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Walk-forward window generation
# ═══════════════════════════════════════════════════════════════════════════

def generate_folds(
    min_date: pd.Timestamp,
    max_date: pd.Timestamp,
    train_months: int = 24,
    val_months: int = 3,
    test_months: int = 3,
    step_months: int = 1,
) -> List[dict]:
    """Generate rolling train/val/test windows."""
    folds = []
    fold_id = 0
    cursor = min_date

    while True:
        train_start = cursor
        train_end = cursor + relativedelta(months=train_months) - pd.Timedelta(days=1)
        val_start = train_end + pd.Timedelta(days=1)
        val_end = val_start + relativedelta(months=val_months) - pd.Timedelta(days=1)
        test_start = val_end + pd.Timedelta(days=1)
        test_end = test_start + relativedelta(months=test_months) - pd.Timedelta(days=1)

        if test_end > max_date:
            break

        folds.append({
            "fold_id": fold_id,
            "train_start": train_start,
            "train_end": train_end,
            "val_start": val_start,
            "val_end": val_end,
            "test_start": test_start,
            "test_end": test_end,
        })
        fold_id += 1
        cursor += relativedelta(months=step_months)

    return folds


# ═══════════════════════════════════════════════════════════════════════════
# Single-fold training  (lightweight, no checkpointing to disk)
# ═══════════════════════════════════════════════════════════════════════════

def _evaluate_split(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """Evaluate model on a dataloader, return IC/NDCG/spread."""
    from scipy.stats import spearmanr

    model.eval()
    all_ic, all_ndcg, all_spread = [], [], []

    with torch.no_grad():
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

                # IC
                finite = np.isfinite(s) & np.isfinite(y)
                if finite.sum() >= 3 and np.std(s[finite]) > 1e-12:
                    ic, _ = spearmanr(s[finite], y[finite])
                    all_ic.append(float(ic) if np.isfinite(ic) else 0.0)

                # NDCG@50
                n = len(s)
                k = min(50, n)
                rel = y - y.min() + 1e-8
                pred_order = np.argsort(-s)
                dcg = sum(rel[pred_order[i]] / np.log2(i + 2) for i in range(k))
                ideal_order = np.argsort(-rel)
                idcg = sum(rel[ideal_order[i]] / np.log2(i + 2) for i in range(k))
                all_ndcg.append(dcg / max(idcg, 1e-12))

                # Spread
                dk = max(1, n // 10)
                order = np.argsort(-s)
                all_spread.append(float(y[order[:dk]].mean() - y[order[-dk:]].mean()))

    return {
        "ic": float(np.mean(all_ic)) if all_ic else 0.0,
        "ndcg": float(np.mean(all_ndcg)) if all_ndcg else 0.0,
        "spread": float(np.mean(all_spread)) if all_spread else 0.0,
        "n_dates": len(all_ic),
    }


def train_fold(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    args,
    device: torch.device,
) -> dict:
    """Train a single fold and return val/test metrics."""
    from algaie.models.ranker.mlp_selector import MLPSelector

    # Import pairwise_ranking_loss from the training script
    sys.path.insert(0, str(ROOT / "backend" / "scripts"))
    from train_selector import pairwise_ranking_loss

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

    model = MLPSelector(
        d_input=len(feature_cols),
        hidden=args.mlp_hidden,
        depth=args.mlp_depth,
        dropout=args.mlp_dropout,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total_steps = len(train_loader) * args.max_epochs
    warmup = min(50, total_steps // 5)

    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    use_amp = args.amp and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp) if use_amp else None

    best_ic = -float("inf")
    best_state = None
    patience_counter = 0
    alpha = args.alpha

    for epoch in range(args.max_epochs):
        model.train()
        for batch in train_loader:
            X = batch["X"].to(device)
            y_ret = batch["y_ret"].to(device)
            mask = batch["mask"].to(device)
            optimizer.zero_grad()

            with autocast(device_type=device.type, enabled=use_amp):
                out = model(X, mask)
                scores = out["score"].squeeze(-1)
                rank_loss = pairwise_ranking_loss(
                    scores, y_ret, mask,
                    max_pairs=args.max_pairs,
                    pairwise_mode=args.pairwise_mode,
                    max_pairs_per_symbol=args.max_pairs_per_symbol,
                )
                reg_loss = F.smooth_l1_loss(
                    scores[mask.bool()], y_ret[mask.bool()], reduction="mean"
                )
                loss = alpha * rank_loss + (1 - alpha) * reg_loss

            if use_amp and scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

        # Validation
        val_metrics = _evaluate_split(model, val_loader, device)
        if val_metrics["ic"] > best_ic:
            best_ic = val_metrics["ic"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                break

    # Load best and evaluate test
    if best_state is not None:
        model.load_state_dict(best_state)

    val_final = _evaluate_split(model, val_loader, device)

    if len(test_df) > 0:
        test_ds = SelectorDataset(test_df, feature_cols=feature_cols)
        test_loader = DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False,
            collate_fn=selector_collate_fn, num_workers=0, pin_memory=False,
        )
        test_final = _evaluate_split(model, test_loader, device)
    else:
        test_final = {"ic": 0.0, "ndcg": 0.0, "spread": 0.0, "n_dates": 0}

    return {
        "val_ic": val_final["ic"],
        "val_ndcg": val_final["ndcg"],
        "val_spread": val_final["spread"],
        "val_n_dates": val_final["n_dates"],
        "test_ic": test_final["ic"],
        "test_ndcg": test_final["ndcg"],
        "test_spread": test_final["spread"],
        "test_n_dates": test_final["n_dates"],
        "best_epoch": epoch + 1 - patience_counter,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Walk-forward evaluation")
    parser.add_argument("--priors-frame", required=True)
    parser.add_argument("--output-dir", default=None)
    # Window config
    parser.add_argument("--train-months", type=int, default=24)
    parser.add_argument("--val-months", type=int, default=3)
    parser.add_argument("--test-months", type=int, default=3)
    parser.add_argument("--step-months", type=int, default=1)
    # Training config
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--mlp-hidden", type=int, default=128)
    parser.add_argument("--mlp-depth", type=int, default=3)
    parser.add_argument("--mlp-dropout", type=float, default=0.10)
    # Pairwise
    parser.add_argument("--pairwise-mode", default="stratified")
    parser.add_argument("--max-pairs", type=int, default=10_000)
    parser.add_argument("--max-pairs-per-symbol", type=int, default=50)
    args = parser.parse_args()

    if args.output_dir is None:
        ts = time.strftime("%Y%m%d-%H%M%S")
        args.output_dir = str(ROOT / "backend" / "data" / "selector" / "walkforward" / f"WF-{ts}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from algaie.core.device import get_device
    device = get_device()

    # Load data
    priors_path = Path(args.priors_frame)
    logger.info(f"Loading priors frame: {priors_path}")
    if priors_path.is_dir():
        parts = sorted(priors_path.glob("date=*/part.parquet"))
        dfs = [pd.read_parquet(p) for p in parts]
        for p, df in zip(parts, dfs):
            date_str = p.parent.name.replace("date=", "")
            df["date"] = pd.to_datetime(date_str)
        df_all = pd.concat(dfs, ignore_index=True)
    else:
        df_all = pd.read_parquet(priors_path)

    df_all["date"] = pd.to_datetime(df_all["date"])
    logger.info(f"Total rows: {len(df_all)}, dates: {df_all['date'].nunique()}")

    # Feature columns
    feature_cols = [c for c in MODEL_FEATURE_COLS if c in df_all.columns]
    logger.info(f"Feature dim: {len(feature_cols)}")

    min_date = df_all["date"].min()
    max_date = df_all["date"].max()

    folds = generate_folds(
        min_date, max_date,
        args.train_months, args.val_months, args.test_months, args.step_months,
    )
    logger.info(f"Generated {len(folds)} folds from {min_date.date()} to {max_date.date()}")

    if not folds:
        logger.error("No folds generated — date range too short")
        sys.exit(1)

    results = []
    for fold in folds:
        fid = fold["fold_id"]
        logger.info(
            f"Fold {fid}: train={fold['train_start'].date()}..{fold['train_end'].date()}, "
            f"val={fold['val_start'].date()}..{fold['val_end'].date()}, "
            f"test={fold['test_start'].date()}..{fold['test_end'].date()}"
        )

        train_mask = (df_all["date"] >= fold["train_start"]) & (df_all["date"] <= fold["train_end"])
        val_mask = (df_all["date"] >= fold["val_start"]) & (df_all["date"] <= fold["val_end"])
        test_mask = (df_all["date"] >= fold["test_start"]) & (df_all["date"] <= fold["test_end"])

        train_df = df_all[train_mask].copy()
        val_df = df_all[val_mask].copy()
        test_df = df_all[test_mask].copy()

        if len(train_df) == 0 or len(val_df) == 0:
            logger.warning(f"Fold {fid}: empty split, skipping")
            continue

        t0 = time.time()
        fold_metrics = train_fold(train_df, val_df, test_df, feature_cols, args, device)
        elapsed = time.time() - t0

        row = {
            "fold_id": fid,
            "train_start": str(fold["train_start"].date()),
            "train_end": str(fold["train_end"].date()),
            "val_start": str(fold["val_start"].date()),
            "val_end": str(fold["val_end"].date()),
            "test_start": str(fold["test_start"].date()),
            "test_end": str(fold["test_end"].date()),
            "train_rows": len(train_df),
            "elapsed_s": round(elapsed, 1),
            **fold_metrics,
        }
        results.append(row)
        logger.info(
            f"  Fold {fid} done in {elapsed:.1f}s: "
            f"val_IC={fold_metrics['val_ic']:.4f} test_IC={fold_metrics['test_ic']:.4f} "
            f"test_spread={fold_metrics['test_spread']:.5f}"
        )

    # Save results
    summary = pd.DataFrame(results)
    summary_path = out_dir / "fold_summary.csv"
    summary.to_csv(summary_path, index=False)
    logger.info(f"\nFold summary saved: {summary_path}")

    # Print aggregate stats
    if len(summary) > 0:
        logger.info(f"\n{'='*60}")
        logger.info(f"Walk-forward summary ({len(summary)} folds):")
        logger.info(f"  Mean test IC:     {summary['test_ic'].mean():.4f} ± {summary['test_ic'].std():.4f}")
        logger.info(f"  Mean test NDCG:   {summary['test_ndcg'].mean():.4f}")
        logger.info(f"  Mean test spread: {summary['test_spread'].mean():.5f}")
        logger.info(f"  % folds with positive test IC: "
                     f"{(summary['test_ic'] > 0).mean()*100:.0f}%")
        logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
