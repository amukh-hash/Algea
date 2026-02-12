"""
Selector training script (MLPSelector default, RankTransformer optional).

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
from algaie.eval.selector_metrics import (
    apply_risk_adjustment,
    compute_bucketed_metrics,
    diagnose_target_alignment,
    diagnose_zscore_universe,
    per_date_metrics,
)
from algaie.eval.portfolio import (
    build_equity_curve,
    build_portfolio,
    compute_portfolio_metrics,
    compute_portfolio_returns,
    compute_regime_breakdown,
)
from algaie.portfolio.portfolio_rules import PortfolioConfig, construct_portfolio as construct_portfolio_v2
from algaie.portfolio.cost_model import CostConfig, apply_costs as apply_costs_v2, compute_turnover_and_cost
from algaie.portfolio.vol_scaling import VolTargetConfig, compute_leverage, apply_leverage
from algaie.models.ranker.baseline_scorer import (
    blend_scores,
    compute_baseline_score,
    compute_gate_input,
    compute_gate_weights,
    detect_dead_features,
    sanity_check_gate_monotonicity,
    tune_gate_params,
)
from algaie.data.priors.feature_utils import (
    add_date_regime_features,
    compute_date_cross_sectional_stats,
    diagnose_cross_sectional_variance,
    fit_time_zscore_scaler,
    make_cs_feature_spec,
    recompute_regime_risk,
)
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

        # Per-symbol appearance cap (vectorized — no Python loops)
        if max_pairs_per_symbol > 0 and len(pair_i) > 0:
            cap = max_pairs_per_symbol
            # Count appearances on each side
            counts_i = torch.bincount(pair_i, minlength=N)
            counts_j = torch.bincount(pair_j, minlength=N)
            combined = counts_i + counts_j  # total appearances per symbol

            over_cap_mask = combined > cap
            if over_cap_mask.any():
                # For each pair, compute keep probability based on
                # how over-represented its symbols are
                ci_count = combined[pair_i].float()
                cj_count = combined[pair_j].float()
                worst = torch.maximum(ci_count, cj_count)
                # keep_prob = cap / actual_count (clamped to [0,1])
                keep_prob = (cap / worst).clamp(max=1.0)
                keep = torch.bernoulli(keep_prob).bool()
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

    # Horizon validation
    target_horizon = getattr(args, "target_horizon", 10)
    logger.info(f"Target horizon: {target_horizon}d")
    if "horizon_ret" in df_all.columns:
        frame_horizons = df_all["horizon_ret"].dropna().unique()
        if len(frame_horizons) == 1 and int(frame_horizons[0]) != target_horizon:
            logger.warning(
                f"⚠ Horizon mismatch: priors_frame has horizon_ret={int(frame_horizons[0])}, "
                f"but --target-horizon={target_horizon}. "
                f"Rebuild priors frame with --target-horizon {target_horizon} for correct targets."
            )
        elif len(frame_horizons) > 1:
            logger.warning(f"⚠ Mixed horizons in priors_frame: {sorted(frame_horizons)}")

    # Recompute regime risk with robust fallback (fix degenerate z_iqr_30)
    df_all = recompute_regime_risk(df_all)

    # Time split (BEFORE fitting temporal scaler to avoid leakage)
    train_df, val_df, test_df = make_time_split(df_all, args.train_end, args.val_end)
    logger.info(f"Split: train={len(train_df)} ({train_df['date'].nunique()}d), "
                f"val={len(val_df)} ({val_df['date'].nunique()}d), "
                f"test={len(test_df)} ({test_df['date'].nunique()}d)")

    # Fit temporal scaler on TRAIN dates only, then apply to all splits
    train_date_stats = compute_date_cross_sectional_stats(train_df)
    cs_scaler = fit_time_zscore_scaler(train_date_stats, "cs_tail_30_std")
    logger.info(f"CS scaler fitted on train: mu={cs_scaler['mu']:.6f}, sigma={cs_scaler['sigma']:.6f}")

    train_df, _ = add_date_regime_features(train_df, scaler=cs_scaler)
    val_df, _ = add_date_regime_features(val_df, scaler=cs_scaler)
    test_df, _ = add_date_regime_features(test_df, scaler=cs_scaler)

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

    # Datasets (Part 2A: decision-date sampling)
    decision_freq = getattr(args, 'decision_frequency', 1)
    train_ds = SelectorDataset(train_df, feature_cols=feature_cols,
                               decision_frequency=decision_freq)
    val_ds = SelectorDataset(val_df, feature_cols=feature_cols)
    if decision_freq > 1:
        logger.info(f"Decision-date sampling: using every {decision_freq}th date "
                    f"({len(train_ds)} training dates)")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=selector_collate_fn, num_workers=0, pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=selector_collate_fn, num_workers=0, pin_memory=False,
    )

    # Model
    model_type = getattr(args, 'model_type', 'mlp')
    if model_type == 'transformer':
        from algaie.models.ranker.rank_transformer import RankTransformer
        model = RankTransformer(
            d_input=len(feature_cols),
            d_model=args.d_model,
            n_head=args.n_head,
            n_layers=args.n_layers,
            dropout=args.dropout,
        ).to(device)
    else:
        from algaie.models.ranker.mlp_selector import MLPSelector
        model = MLPSelector(
            d_input=len(feature_cols),
            hidden=args.mlp_hidden,
            depth=args.mlp_depth,
            dropout=args.mlp_dropout,
            use_risk_head=args.risk_head,
        ).to(device)
    logger.info(f"Model: {model_type}, params: {sum(p.numel() for p in model.parameters()):,}")

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

    # Part 4: score smoothness — cached previous batch scores for cross-batch penalty
    prev_batch_scores_cache = {}  # symbol -> score from prev batch
    smooth_penalty_weight = getattr(args, 'score_smoothness_penalty', 0.0)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_rank_loss = 0.0
        epoch_reg_loss = 0.0
        steps = 0
        t0 = time.time()
        step_times = []

        for batch in train_loader:
            X = batch["X"].to(device)
            y_ret = batch["y_ret"].to(device)
            mask = batch["mask"].to(device)
            step_t0 = time.time()

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

                # Part 4: score smoothness penalty
                if smooth_penalty_weight > 0 and prev_batch_scores_cache:
                    symbols = batch.get("symbols", [[]])
                    if symbols and len(symbols) > 0:
                        smooth_loss = torch.tensor(0.0, device=device)
                        n_smooth = 0
                        for bi in range(scores.shape[0]):
                            batch_syms = symbols[bi] if bi < len(symbols) else []
                            for si, sym in enumerate(batch_syms):
                                sym_str = str(sym)
                                if sym_str in prev_batch_scores_cache and mask[bi, si]:
                                    prev_s = prev_batch_scores_cache[sym_str]
                                    smooth_loss = smooth_loss + (scores[bi, si] - prev_s) ** 2
                                    n_smooth += 1
                        if n_smooth > 0:
                            loss = loss + smooth_penalty_weight * (smooth_loss / n_smooth)

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
            step_times.append(time.time() - step_t0)

            # Part 4: update score cache for smoothness penalty
            if smooth_penalty_weight > 0:
                with torch.no_grad():
                    symbols = batch.get("symbols", [[]])
                    for bi in range(scores.shape[0]):
                        batch_syms = symbols[bi] if bi < len(symbols) else []
                        for si, sym in enumerate(batch_syms):
                            if mask[bi, si]:
                                prev_batch_scores_cache[str(sym)] = scores[bi, si].item()

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

        # Part 2B: portfolio-based early stopping metric
        early_metric_name = getattr(args, 'early_stop_metric', 'ic')
        if early_metric_name == "portfolio":
            selection_metric = _fast_val_portfolio(
                model, val_df, feature_cols, device, args,
            )
        else:
            selection_metric = val_metrics["ic"]

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
        if early_metric_name == "portfolio":
            log_entry["val_portfolio_sharpe"] = round(selection_metric, 4)
        metrics_fh.write(json.dumps(log_entry) + "\n")
        metrics_fh.flush()

        extra_tag = ""
        if early_metric_name == "portfolio":
            extra_tag = f" port_sharpe={selection_metric:+.4f}"
        logger.info(
            f"Epoch {epoch}/{args.epochs} — loss={avg_loss:.5f} "
            f"(rank={avg_rank:.5f} reg={avg_reg:.5f}) | "
            f"val IC={val_metrics['ic']:.4f} IC_ema={ic_ema:.4f} "
            f"NDCG@50={val_metrics['ndcg']:.4f} "
            f"spread={val_metrics['decile_spread']:.5f}{extra_tag} | "
            f"{elapsed:.1f}s | "
            f"avg_step={np.mean(step_times)*1000:.0f}ms"
        )

        # Checkpointing (uses selection_metric — IC or portfolio Sharpe)
        if selection_metric > best_ic:
            best_ic = selection_metric
            patience_counter = 0
            ckpt_path = out_dir / "best_model.pt"
            ckpt_config = {
                "model_type": model_type,
                "d_input": len(feature_cols),
                "alpha": args.alpha,
            }
            if model_type == 'transformer':
                ckpt_config.update({
                    "d_model": args.d_model,
                    "n_head": args.n_head,
                    "n_layers": args.n_layers,
                    "dropout": args.dropout,
                })
            else:
                ckpt_config.update({
                    "hidden": args.mlp_hidden,
                    "depth": args.mlp_depth,
                    "dropout": args.mlp_dropout,
                    "use_risk_head": args.risk_head,
                })
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_ic": val_metrics["ic"],
                "selection_metric": best_ic,
                "selection_metric_name": early_metric_name,
                "feature_cols": feature_cols,
                "config": ckpt_config,
            }, ckpt_path)
            logger.info(f"  ★ New best {early_metric_name}={best_ic:.4f}, saved {ckpt_path}")
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

    # ━━━ Post-training evaluation ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    _run_post_training_eval(
        model, val_df, test_df, feature_cols, device, out_dir, args,
        cs_scaler=cs_scaler,
    )

    logger.info(f"Done. Best val IC={best_ic:.4f}. Outputs: {out_dir}")


def _fast_val_portfolio(
    model: nn.Module,
    val_df: pd.DataFrame,
    feature_cols: List[str],
    device: torch.device,
    args,
) -> float:
    """Run a fast portfolio backtest on val data and return net Sharpe.

    Used for portfolio-based early stopping (Part 2B). Scores val split with
    current model weights, then runs a lightweight backtest with default
    portfolio parameters and returns the net Sharpe ratio.
    """
    from backend.scripts.run_selector_portfolio import run_backtest

    # Score val split quickly
    scored = _score_split_with_model(model, val_df, feature_cols, device,
                                      batch_size=getattr(args, 'batch_size', 8))

    # Apply baseline scorer to get score_final
    if "score" in scored.columns and "score_final" not in scored.columns:
        scored["score_final"] = scored["score"]

    if "score_final" not in scored.columns or scored.empty:
        return -999.0

    # Quick backtest with default params
    port_cfg = PortfolioConfig(
        top_k=getattr(args, "portfolio_k", 50),
        rebalance_horizon_days=getattr(args, "rebalance_period", 10),
        buffer_entry_rank=getattr(args, "buffer_entry_rank", 40),
        buffer_exit_rank=getattr(args, "buffer_exit_rank", 70),
        max_replacements=getattr(args, "max_replacements", 10),
        hold_bonus=getattr(args, "hold_bonus", 0.0),
    )
    cost_cfg = CostConfig(cost_bps=getattr(args, "cost_bps", 10.0))
    target_vol = getattr(args, "target_vol", 0.0)
    vol_cfg = VolTargetConfig(target_vol_ann=target_vol) if target_vol > 0 else None

    target_col = "y_ret"
    if "y_ret" not in scored.columns:
        return -999.0

    _, summary = run_backtest(
        scored, port_cfg, cost_cfg, vol_cfg,
        score_col="score_final", target_col=target_col,
    )

    net_sharpe = summary.get("net_sharpe", -999.0) if summary else -999.0
    return net_sharpe


def _score_split_with_model(
    model: nn.Module,
    split_df: pd.DataFrame,
    feature_cols: List[str],
    device: torch.device,
    batch_size: int = 8,
) -> pd.DataFrame:
    """Score a split DataFrame using the model, returning df with 'score' column."""
    ds = SelectorDataset(split_df, feature_cols=feature_cols)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        collate_fn=selector_collate_fn, num_workers=0, pin_memory=False,
    )
    model.eval()
    all_scores = []
    all_symbols = []
    all_dates = []

    # Process date by date through the loader
    dates_in_order = sorted(split_df["date"].unique())
    date_groups = split_df.groupby("date")

    with torch.no_grad():
        for batch in loader:
            X = batch["X"].to(device)
            mask = batch["mask"].to(device)
            out = model(X, mask)
            scores_batch = out["score"].squeeze(-1).cpu()
            mask_cpu = batch["mask"]

            for b in range(scores_batch.shape[0]):
                m = mask_cpu[b].numpy().astype(bool)
                s = scores_batch[b].numpy()[m]
                all_scores.extend(s.tolist())

    # Match scores back to DataFrame rows
    result = split_df.copy()
    result = result.sort_values(["date", "symbol"]).reset_index(drop=True)
    if len(all_scores) == len(result):
        result["score"] = all_scores
    else:
        logger.warning(f"Score count mismatch: {len(all_scores)} vs {len(result)} rows")
        result["score"] = 0.0
    return result


def _run_post_training_eval(
    model: nn.Module,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    device: torch.device,
    out_dir: Path,
    args,
    cs_scaler: dict | None = None,
):
    """Post-training: 3-variant evaluation (baseline / model / blended)."""
    logger.info("\n" + "═" * 60)
    logger.info("Post-training evaluation (baseline / model / blended)")
    logger.info("═" * 60)

    target_col = "y_ret"
    blend_mode = getattr(args, "blend_mode", "sigmoid")
    g0 = getattr(args, "gate_g0", 0.0)
    g1 = getattr(args, "gate_g1", 1.0)
    baseline_a = getattr(args, "baseline_a", 1.0)
    baseline_lam = getattr(args, "baseline_lambda", 0.5)
    baseline_mu = getattr(args, "baseline_mu", 0.5)

    # ── Dead feature report (per-split) ─────────────────────────────────
    for sname, sdf in [("val", val_df), ("test", test_df)]:
        dead = detect_dead_features(sdf, feature_cols)
        if dead:
            logger.warning(
                f"⚠ Dead features in {sname} split (std≈0): {dead}\n"
                f"    → Consider adding to DEAD_ZSCORE_COLS in selector_schema.py\n"
                f"    → Or pass --drop-dead-features to auto-prune (NOT recommended without review)"
            )

    # Score val first for gate tuning
    scored_splits = {}
    for split_name, split_df in [("val", val_df), ("test", test_df)]:
        if split_df.empty:
            continue
        scored = _score_split_with_model(model, split_df, feature_cols, device, args.batch_size)
        scored.rename(columns={"score": "score_model"}, inplace=True)
        scored["score_baseline"] = compute_baseline_score(
            scored, a=baseline_a, lam=baseline_lam, mu=baseline_mu,
        )
        scored_splits[split_name] = scored

    # Gate tuning on validation (lightweight grid search)
    tuned_g0, tuned_g1 = g0, g1
    tuned_gamma_cs = getattr(args, "gate_gamma_cs", 1.0)
    tuned_cs_sign = getattr(args, "gate_cs_sign", -1.0)
    if "val" in scored_splits:
        try:
            tune_result = tune_gate_params(
                scored_splits["val"], target_col,
                "score_model", "score_baseline",
            )
            tuned_g0 = tune_result["best_g0"]
            tuned_g1 = tune_result["best_g1"]
            tuned_gamma_cs = tune_result["best_gamma_cs"]
            tuned_cs_sign = tune_result["best_cs_sign"]
            logger.info(
                f"  Gate tuning: best g0={tuned_g0}, g1={tuned_g1}, "
                f"gamma_cs={tuned_gamma_cs}, cs_sign={tuned_cs_sign}, "
                f"val IC={tune_result['best_ic']:.4f}"
            )
            # Save tuning results + scaler (manifest format per spec)
            gate_gamma_unc = getattr(args, "gate_gamma", 0.0)
            gate_use_unc = gate_gamma_unc != 0.0
            manifest = {
                "blend_mode": blend_mode,
                "gate_g0": tuned_g0,
                "gate_g1": tuned_g1,
                "gate_gamma_cs": tuned_gamma_cs,
                "gate_cs_sign": tuned_cs_sign,
                "gate_gamma_unc": gate_gamma_unc,
                "gate_use_uncertainty": gate_use_unc,
                "baseline_a": baseline_a,
                "baseline_lambda": baseline_lam,
                "baseline_mu": baseline_mu,
                "cs_feature": "cs_tail_30_std",
                "cs_feature_spec": make_cs_feature_spec(),
                "cs_scaler": cs_scaler,
                "target_horizon": getattr(args, "target_horizon", 10),
                "cost_model": {
                    "cost_bps": getattr(args, "cost_bps", 10),
                    "slippage_multiplier": getattr(args, "slippage_multiplier", 1.0),
                },
                "volatility_scaling": {
                    "enabled": getattr(args, "target_vol", 0) > 0,
                    "target_vol": getattr(args, "target_vol", 0.15),
                },
                "portfolio": {
                    "k": getattr(args, "portfolio_k", 50),
                    "weighting": getattr(args, "portfolio_weighting", "equal"),
                    "market_neutral": getattr(args, "market_neutral", False),
                },
                "tuning": {
                    "objective": "mean_ic",
                    "split": "val",
                    "grid": {
                        "g0": tune_result["grid_results"][0]["g0"] if tune_result["grid_results"] else None,
                        "g1_values": sorted(set(r["g1"] for r in tune_result["grid_results"])),
                        "gamma_cs_values": sorted(set(r["gamma_cs"] for r in tune_result["grid_results"])),
                        "cs_sign_values": sorted(set(r["cs_sign"] for r in tune_result["grid_results"])),
                    },
                    "best_ic": tune_result["best_ic"],
                },
                "grid_results": tune_result["grid_results"],
            }
            with open(out_dir / "blend_manifest.json", "w") as f:
                json.dump(manifest, f, indent=2)
        except Exception as e:
            logger.warning(f"  Gate tuning failed: {e}")

    # Evaluate all splits with 3 variants
    highlight_months = ["2025-08", "2025-09"]  # months of interest
    summary_rows = []

    for split_name, scored in scored_splits.items():
        logger.info(f"\n--- {split_name.upper()} split: baseline / model / blended ---")

        # Compute blended score
        scored["score_final"] = blend_scores(
            scored, "score_model", "score_baseline",
            blend_mode=blend_mode, g0=tuned_g0, g1=tuned_g1,
            gate_gamma_cs=tuned_gamma_cs,
            gate_cs_sign=tuned_cs_sign,
        )

        variants = {
            "baseline": "score_baseline",
            "model": "score_model",
            "blended": "score_final",
        }

        # Overall metrics per variant
        for vname, vcol in variants.items():
            dm = per_date_metrics(scored, vcol, target_col)
            overall_ic = dm["ic"].mean()
            overall_ndcg = dm["ndcg"].mean()
            overall_spread = dm["spread"].mean()
            logger.info(
                f"  {vname:10s}: IC={overall_ic:+.4f}  NDCG={overall_ndcg:.4f}  "
                f"spread={overall_spread:+.5f}"
            )
            summary_rows.append({
                "split": split_name, "variant": vname,
                "ic": overall_ic, "ndcg": overall_ndcg, "spread": overall_spread,
            })

        # Monthly bucketed metrics per variant
        for vname, vcol in variants.items():
            try:
                bm = compute_bucketed_metrics(scored, vcol, target_col, bucket="month")
                bm_path = out_dir / f"bucket_metrics_month_{vname}_{split_name}.csv"
                bm.to_csv(bm_path, index=False)
            except Exception as e:
                logger.warning(f"  Bucketed ({vname}) failed: {e}")

        # Compact monthly table + sign diagnostics (test split)
        if split_name == "test":
            # Compute gate weights and gate_input for reporting
            gate_gamma = getattr(args, "gate_gamma", 0.0)
            scored["_gate_w"] = compute_gate_weights(
                scored, blend_mode, g0=tuned_g0, g1=tuned_g1,
                gate_gamma=gate_gamma,
                gate_gamma_cs=tuned_gamma_cs,
                gate_cs_sign=tuned_cs_sign,
            )
            scored["_gate_input"] = compute_gate_input(
                scored, gate_gamma=gate_gamma,
                gate_gamma_cs=tuned_gamma_cs,
                gate_cs_sign=tuned_cs_sign,
            )

            # --- Deliverable C: monotonicity guardrail ---
            sanity_check_gate_monotonicity(
                scored["_gate_w"].values,
                scored["_gate_input"].values,
                tuned_g1,
                label="test split",
            )

            # --- Deliverable B: effective CS direction ---
            eff_cs_dir = np.sign(tuned_gamma_cs * tuned_cs_sign)
            cs_spec = make_cs_feature_spec()
            meaning = cs_spec["meaning"]
            if eff_cs_dir > 0 and meaning == "stress_up":
                interp = "stress_up -> gate_input UP -> w DOWN -> more baseline (CORRECT)"
            elif eff_cs_dir < 0 and meaning == "stress_up":
                interp = "stress_up -> gate_input DOWN -> w UP -> more model (INVERTED)"
            elif eff_cs_dir == 0:
                interp = "CS contribution is ZERO (gamma_cs * cs_sign = 0)"
            else:
                interp = f"eff_sign={eff_cs_dir:+.0f} with meaning={meaning}"
            logger.info(
                f"\n  Gate CS semantics: gamma_cs={tuned_gamma_cs}, cs_sign={tuned_cs_sign}, "
                f"effective={eff_cs_dir:+.0f}"
            )
            logger.info(f"  Interpretation: {interp}")

            # --- Deliverable B: row-level correlations ---
            w_arr = scored["_gate_w"].values
            gi_arr = scored["_gate_input"].values
            cs_col = "z_cs_tail_30_std"
            cs_arr = scored[cs_col].fillna(0).values if cs_col in scored.columns else np.zeros(len(scored))
            cs_term_arr = tuned_gamma_cs * tuned_cs_sign * cs_arr  # effective cs term

            rho_w_gi = np.corrcoef(w_arr, gi_arr)[0, 1]
            rho_w_csterm = np.corrcoef(w_arr, cs_term_arr)[0, 1]
            rho_gi_cs = np.corrcoef(gi_arr, cs_arr)[0, 1]
            rho_w_cs_raw = np.corrcoef(w_arr, cs_arr)[0, 1]

            logger.info(f"\n  Row-level correlations (test split):")
            logger.info(f"    corr(w, gate_input)           = {rho_w_gi:+.4f}  (expected <0 when g1>0)")
            logger.info(f"    corr(w, cs_term)              = {rho_w_csterm:+.4f}  (expected <0 when stress_up & gamma_cs>0)")
            logger.info(f"    corr(gate_input, z_cs)        = {rho_gi_cs:+.4f}  (expected sign={np.sign(tuned_cs_sign):+.0f})")
            logger.info(f"    corr(w, z_cs_raw)             = {rho_w_cs_raw:+.4f}  (expected sign={-np.sign(tuned_cs_sign):+.0f} when g1>0)")

            # --- Deliverable B: monthly table ---
            logger.info(f"\n  Monthly test comparison (IC + gate weight w):")
            logger.info(
                f"  {'Month':8s}  {'Baseline':>9s}  {'Model':>9s}  {'Blended':>9s}"
                f"  {'mean_w':>7s} {'p10_w':>6s} {'p90_w':>6s} {'z_cs':>7s}"
            )
            logger.info(f"  {'---':8s}  {'---':>9s}  {'---':>9s}  {'---':>9s}  {'---':>7s} {'---':>6s} {'---':>6s} {'---':>7s}")
            try:
                bm_b = compute_bucketed_metrics(scored, "score_baseline", target_col, bucket="month")
                bm_m = compute_bucketed_metrics(scored, "score_model", target_col, bucket="month")
                bm_f = compute_bucketed_metrics(scored, "score_final", target_col, bucket="month")

                scored["_date_ts"] = pd.to_datetime(scored["date"])
                scored["_month"] = scored["_date_ts"].dt.to_period("M").astype(str)
                gate_stats = scored.groupby("_month")["_gate_w"].agg(
                    ["mean", lambda x: x.quantile(0.1), lambda x: x.quantile(0.9)]
                )
                gate_stats.columns = ["mean_w", "p10_w", "p90_w"]

                if cs_col in scored.columns:
                    cs_stats = scored.groupby("_month")[cs_col].mean()
                else:
                    cs_stats = pd.Series(dtype=float)

                for i in range(len(bm_b)):
                    bkt = bm_b.iloc[i]["bucket"]
                    ic_b = bm_b.iloc[i]["ic_by_date"]
                    ic_m = bm_m.iloc[i]["ic_by_date"]
                    ic_f = bm_f.iloc[i]["ic_by_date"]
                    gs = gate_stats.loc[bkt] if bkt in gate_stats.index else None
                    mw = gs["mean_w"] if gs is not None else float("nan")
                    p10 = gs["p10_w"] if gs is not None else float("nan")
                    p90 = gs["p90_w"] if gs is not None else float("nan")
                    csm = cs_stats.get(bkt, float("nan"))
                    marker = " <<" if bkt in highlight_months else ""
                    logger.info(
                        f"  {bkt:8s}  {ic_b:+9.4f}  {ic_m:+9.4f}  {ic_f:+9.4f}"
                        f"  {mw:7.3f} {p10:6.3f} {p90:6.3f} {csm:+7.3f}{marker}"
                    )

                # --- Deliverable B: top/bottom 5 dates by z_cs ---
                if cs_col in scored.columns:
                    per_date = scored.groupby("date").agg(
                        z_cs=(cs_col, "first"),
                        mean_w=("_gate_w", "mean"),
                    ).reset_index()
                    per_date = per_date.sort_values("z_cs")
                    logger.info(f"\n  Top 5 stress dates (highest z_cs):")
                    logger.info(f"    {'date':12s}  {'z_cs':>7s}  {'mean_w':>7s}")
                    for _, r in per_date.tail(5).iloc[::-1].iterrows():
                        logger.info(f"    {str(r['date']):12s}  {r['z_cs']:+7.3f}  {r['mean_w']:7.3f}")
                    logger.info(f"  Bottom 5 calm dates (lowest z_cs):")
                    for _, r in per_date.head(5).iterrows():
                        logger.info(f"    {str(r['date']):12s}  {r['z_cs']:+7.3f}  {r['mean_w']:7.3f}")

                scored.drop(columns=["_date_ts", "_month", "_gate_w", "_gate_input"],
                            inplace=True, errors="ignore")

            except Exception as e:
                logger.warning(f"  Monthly comparison failed: {e}")

        # Save scored data
        save_cols = ["date", "symbol", target_col,
                     "score_model", "score_baseline", "score_final"]
        save_cols = [c for c in save_cols if c in scored.columns]
        scored[save_cols].to_parquet(
            out_dir / f"scored_{split_name}.parquet", index=False,
        )

    # Overall summary
    if summary_rows:
        logger.info(f"\n  {'='*55}")
        logger.info(f"  Overall summary:")
        for r in summary_rows:
            logger.info(
                f"    {r['split']:5s} {r['variant']:10s}: IC={r['ic']:+.4f}  "
                f"NDCG={r['ndcg']:.4f}  spread={r['spread']:+.5f}"
            )
        logger.info(f"  {'='*55}")

    # (E) Diagnostics
    if getattr(args, "diagnose", False):
        logger.info("\n--- Diagnostics ---")
        zdiag = diagnose_zscore_universe(combined, n_dates=5)
        logger.info(f"  Z-score universe check: {zdiag['n_dates_checked']} dates checked")
        for w in zdiag.get("warnings", []):
            logger.warning(f"  ⚠ {w}")
        if not zdiag.get("warnings"):
            logger.info("  ✓ Z-scores look healthy (mean≈0, std≈1)")

        ticker_dir = ROOT / "backend" / "data" / "canonical" / "per_ticker"
        diag_horizon = getattr(args, "target_horizon", 10)
        if ticker_dir.exists():
            tdiag = diagnose_target_alignment(
                combined, ticker_dir, target_col, horizon=diag_horizon, n_samples=20,
            )
            logger.info(
                f"  Target alignment ({diag_horizon}d): {tdiag['n_checked']} checked, "
                f"{tdiag['n_mismatches']} mismatches (max err={tdiag['max_abs_error']:.6f})"
            )
        else:
            logger.info(f"  Target alignment: skipped (no ticker data at {ticker_dir})")

    # ── Portfolio evaluation ──────────────────────────────────────────────
    _run_portfolio_eval(scored_splits, target_col, out_dir, args)



def _run_portfolio_eval(scored_splits, target_col, out_dir, args):
    """Run portfolio construction, cost modeling, and vol scaling evaluation."""
    logger.info("\n" + "═" * 60)
    logger.info("Portfolio evaluation (construction / cost / vol-scaling)")
    logger.info("═" * 60)

    cost_bps = getattr(args, "cost_bps", 10)
    slippage_mult = getattr(args, "slippage_multiplier", 1.0)
    port_k = getattr(args, "portfolio_k", 50)
    port_weighting = getattr(args, "portfolio_weighting", "equal")
    market_neutral = getattr(args, "market_neutral", False)
    target_vol = getattr(args, "target_vol", 0.0)
    rebalance_period = getattr(args, "rebalance_period", 10)
    periods_per_year = int(252 / rebalance_period)

    logger.info(f"  Rebalance: every {rebalance_period}d ({periods_per_year} periods/year)")
    logger.info(f"  Cost: {cost_bps} bps, slippage: {slippage_mult}x")
    if target_vol > 0:
        logger.info(f"  Vol scaling: target={target_vol:.0%}")

    full_report = {"signal": {}, "portfolio": {}, "regime": {}}

    for split_name, scored in scored_splits.items():
        if scored.empty or "score_final" not in scored.columns:
            continue

        logger.info(f"\n  ── {split_name.upper()} portfolio ──")

        # Signal layer metrics
        from algaie.eval.selector_metrics import per_date_metrics as pdm
        dm = pdm(scored, "score_final", target_col)
        sig = {
            "ic": round(float(dm["ic"].mean()), 4),
            "ndcg": round(float(dm["ndcg"].mean()), 4),
            "spread": round(float(dm["spread"].mean()), 5),
            "n_dates": len(dm),
        }
        full_report["signal"][split_name] = sig
        logger.info(f"    Signal: IC={sig['ic']:+.4f}  NDCG={sig['ndcg']:.4f}  spread={sig['spread']:+.5f}")

        # Build portfolio
        sector_col = "sector" if "sector" in scored.columns else None
        port = build_portfolio(
            scored, score_col="score_final", target_col=target_col,
            k=port_k, weighting=port_weighting,
            max_weight=0.05, max_sector_weight=0.25,
            sector_col=sector_col, market_neutral=market_neutral,
            rebalance_period=rebalance_period,
        )
        if port.empty:
            logger.warning(f"    Portfolio empty for {split_name} (not enough dates with k={port_k} stocks)")
            continue

        # Returns with cost model
        returns = compute_portfolio_returns(
            port, cost_bps=cost_bps, slippage_multiplier=slippage_mult,
        )

        # Metrics (with optional vol scaling)
        tv = target_vol if target_vol > 0 else None
        metrics = compute_portfolio_metrics(returns, target_vol=tv, periods_per_year=periods_per_year)
        full_report["portfolio"][split_name] = metrics

        logger.info(
            f"    Gross Sharpe: {metrics['gross_sharpe']:.4f}  "
            f"Net Sharpe: {metrics['net_sharpe']:.4f}  "
            f"CAGR(net): {metrics['cagr_net']:.4f}"
        )
        logger.info(
            f"    Max DD: {metrics['max_drawdown']:.4f}  "
            f"Turnover: {metrics['avg_turnover']:.4f} (ann: {metrics['ann_turnover']:.1f})  "
            f"Avg hold: {metrics['avg_holding_period']:.1f}d"
        )
        if "vol_scaled_sharpe" in metrics:
            logger.info(
                f"    Vol-scaled Sharpe: {metrics['vol_scaled_sharpe']:.4f}  "
                f"(effect: {metrics['vol_scaling_effect']:+.4f})"
            )

        # Regime breakdown
        regime = compute_regime_breakdown(scored, returns, cs_col="z_cs_tail_30_std", periods_per_year=periods_per_year)
        if regime:
            full_report["regime"][split_name] = regime
            logger.info(f"    Regime breakdown:")
            for bucket, stats in regime.items():
                logger.info(
                    f"      {bucket:20s}: n={stats['n_dates']:3d}  "
                    f"gross={stats['mean_gross_ret']:+.5f}  "
                    f"net={stats['mean_net_ret']:+.5f}  "
                    f"sharpe={stats['ann_sharpe']:+.4f}"
                )

        # Equity curve (save for test split)
        if split_name == "test":
            eq = build_equity_curve(returns)
            eq_path = out_dir / "portfolio_equity_curve.csv"
            eq.to_csv(eq_path, index=False)
            logger.info(f"    Equity curve saved: {eq_path}")

    # Persist full report
    report_path = out_dir / "selector_full_report.json"
    with open(report_path, "w") as f:
        json.dump(full_report, f, indent=2, default=str)
    logger.info(f"\n  Full report saved: {report_path}")

    # Persist portfolio metrics separately
    if "test" in full_report["portfolio"]:
        pm_path = out_dir / "portfolio_metrics.json"
        with open(pm_path, "w") as f:
            json.dump(full_report["portfolio"]["test"], f, indent=2)
        logger.info(f"  Portfolio metrics saved: {pm_path}")

    # Final summary line
    if "test" in full_report["signal"] and "test" in full_report["portfolio"]:
        s = full_report["signal"]["test"]
        p = full_report["portfolio"]["test"]
        logger.info(f"\n  ╔══════════════════════════════════════╗")
        logger.info(f"  ║  Selector {getattr(args, 'target_horizon', 10)}d IC:      {s['ic']:+.4f}       ║")
        logger.info(f"  ║  Portfolio gross Sharpe: {p['gross_sharpe']:+.4f}       ║")
        logger.info(f"  ║  Portfolio net Sharpe:   {p['net_sharpe']:+.4f}       ║")
        logger.info(f"  ║  Turnover (ann):         {p['ann_turnover']:7.1f}       ║")
        if "vol_scaled_sharpe" in p:
            logger.info(f"  ║  Vol scaling effect:    {p['vol_scaling_effect']:+.4f}       ║")
        logger.info(f"  ╚══════════════════════════════════════╝")

    # ── New portfolio engine (turnover-controlled) ────────────────────
    _run_portfolio_v2(scored_splits, target_col, out_dir, args)


def _run_portfolio_v2(scored_splits, target_col, out_dir, args):
    """Run turnover-controlled portfolio backtest (v2 engine)."""
    from backend.scripts.run_selector_portfolio import run_backtest, log_summary, save_artifacts

    rebalance_period = getattr(args, "rebalance_period", 10)
    cost_bps = getattr(args, "cost_bps", 10.0)
    target_vol = getattr(args, "target_vol", 0.0)

    port_cfg = PortfolioConfig(
        top_k=getattr(args, "portfolio_k", 50),
        rebalance_horizon_days=rebalance_period,
        buffer_entry_rank=getattr(args, "buffer_entry_rank", 40),
        buffer_exit_rank=getattr(args, "buffer_exit_rank", 70),
        max_replacements=getattr(args, "max_replacements", 10),
        hold_bonus=getattr(args, "hold_bonus", 0.0),
    )
    cost_cfg = CostConfig(cost_bps=cost_bps)
    vol_cfg = VolTargetConfig(target_vol_ann=target_vol) if target_vol > 0 else None

    logger.info("\n" + "─" * 60)
    logger.info("Turnover-controlled portfolio (v2 engine)")
    logger.info("─" * 60)

    for split_name in ["val", "test"]:
        scored = scored_splits.get(split_name)
        if scored is None or scored.empty:
            continue
        if "score_final" not in scored.columns:
            continue

        logger.info(f"\n  ── {split_name.upper()} (v2) ──")
        returns_df, summary = run_backtest(
            scored, port_cfg, cost_cfg, vol_cfg,
            score_col="score_final", target_col=target_col,
        )
        if summary:
            log_summary(summary, f"{split_name} v2")

            # Also run baseline (full-refresh)
            baseline_cfg = PortfolioConfig(
                top_k=port_cfg.top_k,
                rebalance_horizon_days=rebalance_period,
                buffer_entry_rank=port_cfg.top_k,
                buffer_exit_rank=9999,
                max_replacements=None,
                hold_bonus=0.0,
            )
            _, base_sum = run_backtest(
                scored, baseline_cfg, cost_cfg, vol_cfg,
                score_col="score_final", target_col=target_col,
            )
            if base_sum:
                delta_sharpe = summary.get("net_sharpe",0) - base_sum.get("net_sharpe",0)
                delta_to = summary.get("avg_turnover_1way",0) - base_sum.get("avg_turnover_1way",0)
                logger.info(f"    vs baseline: Sharpe Δ={delta_sharpe:+.4f}  Turnover Δ={delta_to:+.4f}")

        if split_name == "test" and not returns_df.empty:
            save_artifacts(
                out_dir / "portfolio_v2",
                returns_df, summary, port_cfg, cost_cfg, vol_cfg,
                label="test_v2",
            )


def main():
    parser = argparse.ArgumentParser(description="Train selector model")
    parser.add_argument("--priors-frame", required=True, help="Path to priors_frame dir or parquet")
    parser.add_argument("--train-end", default="2021-12-31")
    parser.add_argument("--val-end", default="2023-12-31")
    parser.add_argument("--target-horizon", type=int, default=10,
                        help="Forward return horizon in trading days (default: 10)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of epochs (default: 15 for mlp, 30 for transformer)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=0.7, help="Ranking vs regression balance")
    # Model selection
    parser.add_argument("--model-type", choices=["mlp", "transformer"],
                        default="mlp", help="Model architecture (default: mlp)")
    # Transformer-specific
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    # MLP-specific
    parser.add_argument("--mlp-hidden", type=int, default=128,
                        help="MLP hidden dimension")
    parser.add_argument("--mlp-depth", type=int, default=3,
                        help="Number of MLP hidden layers")
    parser.add_argument("--mlp-dropout", type=float, default=0.10,
                        help="MLP dropout rate")
    # Common
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
    # Blend / gate strategy
    parser.add_argument("--blend-mode", choices=["sigmoid", "piecewise"],
                        default="sigmoid", help="Blend strategy (default: sigmoid)")
    parser.add_argument("--gate-g0", type=float, default=0.0,
                        help="Sigmoid gate bias (default 0.0)")
    parser.add_argument("--gate-g1", type=float, default=1.0,
                        help="Sigmoid gate slope (default 1.0)")
    parser.add_argument("--gate-threshold", type=float, default=0.0,
                        help="Piecewise gate threshold")
    parser.add_argument("--gate-k", type=float, default=1.0,
                        help="Piecewise gate slope")
    parser.add_argument("--gate-gamma-cs", type=float, default=1.0,
                        help="Weight for date-level regime stress in gate (default 1.0)")
    parser.add_argument("--gate-cs-sign", type=float, default=-1.0,
                        help="Sign multiplier for date-level signal (default -1.0)")
    # Baseline scorer
    parser.add_argument("--baseline-a", type=float, default=1.0,
                        help="Baseline drift weight")
    parser.add_argument("--baseline-lambda", type=float, default=0.5,
                        help="Baseline regime risk weight")
    parser.add_argument("--baseline-mu", type=float, default=0.5,
                        help="Baseline tail risk weight")
    # Risk-adjusted scoring (legacy)
    parser.add_argument("--risk-adjust", action="store_true", default=False,
                        help="Enable risk-adjusted score (post-processing)")
    parser.add_argument("--lambda-risk", type=float, default=0.5,
                        help="Regime risk penalty weight for risk-adjusted score")
    parser.add_argument("--mu-tail", type=float, default=0.5,
                        help="Tail risk penalty weight for risk-adjusted score")
    # Diagnostics
    parser.add_argument("--diagnose", action="store_true", default=False,
                        help="Run target alignment & z-score diagnostics")
    # Production mode (swing-trading optimised defaults)
    parser.add_argument("--production-mode", action="store_true", default=False,
                        help="Apply swing-trading optimised defaults "
                             "(epochs=15, batch=8, stratified, alpha=0.7, etc.)")
    parser.add_argument("--drop-dead-features", action="store_true", default=False,
                        help="Auto-prune dead features (NOT recommended without review)")
    # Portfolio construction
    parser.add_argument("--portfolio-k", type=int, default=50,
                        help="Number of stocks in portfolio (default 50)")
    parser.add_argument("--portfolio-weighting", choices=["equal", "score_proportional", "softmax"],
                        default="equal", help="Portfolio weighting mode")
    parser.add_argument("--market-neutral", action="store_true", default=False,
                        help="Long-short dollar-neutral portfolio")
    # Cost model
    parser.add_argument("--cost-bps", type=float, default=10.0,
                        help="Round-trip cost in basis points (default 10)")
    parser.add_argument("--slippage-multiplier", type=float, default=1.0,
                        help="Slippage multiplier on turnover (default 1.0)")
    # Volatility scaling
    parser.add_argument("--target-vol", type=float, default=0.0,
                        help="Target annualized vol for scaling (0=disabled, e.g. 0.15)")
    # Rebalance
    parser.add_argument("--rebalance-period", type=int, default=10,
                        help="Rebalance every N days (default 10, matches target horizon)")
    # ── Hardening flags (Parts 2A, 2B, 4) ────────────────────────────
    parser.add_argument("--decision-frequency", type=int, default=1,
                        help="Train on every Nth trading day (Part 2A, default 1=all)")
    parser.add_argument("--early-stop-metric", choices=["ic", "portfolio"],
                        default="ic",
                        help="Early stopping metric: IC or portfolio net Sharpe (Part 2B)")
    parser.add_argument("--score-smoothness-penalty", type=float, default=0.0,
                        help="Smoothness penalty weight for score stability (Part 4, 0=disabled)")
    args = parser.parse_args()

    # --production-mode: override defaults unless user explicitly set them
    if args.production_mode:
        _defaults = {
            "epochs": 15,
            "batch_size": 8 if args.model_type == "mlp" else 1,
            "pairwise_mode": "stratified",
            "max_pairs": 10_000,
            "alpha": 0.7,
            "skip_baseline": False,
            "blend_mode": "sigmoid",
            "patience": 10,
            "target_horizon": 10,
            "rebalance_period": 10,
            "cost_bps": 10.0,
            "slippage_multiplier": 1.0,
            "target_vol": 0.15,
            # Part 8: hardening defaults
            "early_stop_metric": "portfolio",
            "decision_frequency": 10,
            "score_smoothness_penalty": 0.0,
        }
        for k, v in _defaults.items():
            # Only override if the user didn't explicitly set the flag
            if getattr(args, k, None) == parser.get_default(k):
                setattr(args, k, v)
        logger.info("Production mode: swing-trading defaults applied (with hardening)")

    # Default epochs based on model type
    if args.epochs is None:
        args.epochs = 15 if args.model_type == 'mlp' else 30

    if args.output_dir is None:
        ts = time.strftime("%Y%m%d-%H%M%S")
        args.output_dir = str(ROOT / "backend" / "data" / "selector" / "runs" / f"SEL-{ts}")

    # Part 8: save production manifest
    if args.production_mode:
        manifest = {k: v for k, v in vars(args).items()}
        manifest_path = Path(args.output_dir)
        manifest_path.mkdir(parents=True, exist_ok=True)
        with open(manifest_path / "production_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2, default=str)
        logger.info(f"Production manifest: {manifest_path / 'production_manifest.json'}")

    train(args)


if __name__ == "__main__":
    main()
