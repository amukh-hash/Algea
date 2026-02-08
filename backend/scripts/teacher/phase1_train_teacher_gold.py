#!/usr/bin/env python3
"""
Phase 1 (GOLD): Optional Chronos-2 calibration on daily adjusted equity bars + covariates.
Uses 'backend.app.models.chronos2_teacher' for model wrapper.
This aligns with the Training Protocol (daily OHLCV + regime covariates).
"""

from __future__ import annotations

import os
import sys
import argparse
import time
import json
import uuid
import datetime
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Any
from collections import OrderedDict
import threading
import gc
import psutil

import math
import numpy as np
from chronos import Chronos2Pipeline

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader, Subset
    from transformers import get_linear_schedule_with_warmup
except Exception:
    print("ERROR: torch/transformers is required.", file=sys.stderr)
    raise

try:
    import polars as pl
except Exception:
    print("ERROR: polars is required.", file=sys.stderr)
    raise

# Imports from backend
sys.path.append(os.getcwd())
try:
    from backend.app.models.chronos2_teacher import load_chronos_adapter, Chronos2NativeWrapper
    from backend.app.ops import run_recorder
except ImportError as e:
    print(f"ERROR: Backend imports failed: {e}", file=sys.stderr)
    sys.exit(1)


@dataclass(frozen=False)
class Phase1Config:
    run_id: str
    seed: int
    model_id: str
    
    use_qlora: bool
    lora_config: dict
    
    gold_parquet_dir: Path
    gold_glob: str
    required_cols: Tuple[str, ...]
    
    context: int
    pred: int
    stride_rows: int
    max_files: int
    max_windows: int
    cache_size: int
    
    batch_size: int
    grad_accum: int
    lr: float
    max_steps: int
    warmup_steps: int
    checkpoint_every: int
    log_every: int
    val_split_pct: float
    
    out_dir: Path
    probe_only: bool
    allow_fallback_codec: bool
    target_col: str
    covariate_cols: Tuple[str, ...]
    future_covariate_cols: Tuple[str, ...]
    
    max_files_limit: int = 1000
    unfreeze_layernorms: bool = False
    unfreeze_last_blocks: int = 0
    
    # Sigma Penalty Config
    sigma_band_low: float = 0.5
    sigma_band_high: float = 2.0
    sigma_reg_weight: float = 1e-4
    
    # Quantile Head Flags (bypass until encoder extraction fixed)
    enable_q10d_head: bool = False
    calibration_mode: str = "sigma_only"  # or "full" when quantiles enabled


def env_path(name: str, default: str) -> Path:
    return Path(os.getenv(name, default)).expanduser().resolve()


def load_config(args) -> Phase1Config:
    seed = int(os.getenv("SEED", "42"))
    model_id = os.getenv("CHRONOS2_MODEL_ID", "amazon/chronos-2")
    gold_dir = env_path("GOLD_DAILY_PARQUET_DIR", "backend/data_canonical/daily_parquet")
    gold_glob = os.getenv("GOLD_EXAMPLE_GLOB", "*.parquet")
    required_cols = tuple(c.strip() for c in os.getenv(
        "GOLD_REQUIRED_COLS",
        "date,ret_1d,ret_3d,ret_5d,ret_10d,volume,"
        "spy_ret_1d,qqq_ret_1d,iwm_ret_1d,vix_level,rate_proxy,market_breadth_ad"
    ).split(","))

    lora_config = {
        "rank": int(os.getenv("LORA_RANK", "16")),
        "alpha": int(os.getenv("LORA_ALPHA", "32")),
        "dropout": float(os.getenv("LORA_DROPOUT", "0.05")),
        "target_modules": os.getenv("LORA_TARGET_MODULES", None) # "all", "q,v", or None (auto)
    }

    target_col = os.getenv("CHRONOS2_TARGET_COL", "ret_1d")
    covariate_cols = tuple(c.strip() for c in os.getenv(
        "CHRONOS2_COVARIATE_COLS",
        "volume,spy_ret_1d,qqq_ret_1d,iwm_ret_1d,vix_level,rate_proxy,market_breadth_ad"
    ).split(",") if c.strip())
    future_covariate_cols = tuple(c.strip() for c in os.getenv(
        "CHRONOS2_FUTURE_COVARIATE_COLS",
        "spy_ret_1d,qqq_ret_1d,iwm_ret_1d,vix_level,rate_proxy,market_breadth_ad"
    ).split(",") if c.strip())

    return Phase1Config(
        run_id=str(uuid.uuid4())[:8],
        seed=seed,
        model_id=model_id,
        use_qlora=os.getenv("USE_QLORA", "0") == "1",
        lora_config=lora_config,
        gold_parquet_dir=gold_dir,
        gold_glob=gold_glob,
        required_cols=required_cols,
        context=int(os.getenv("GOLD_CONTEXT", "180")),
        pred=int(os.getenv("GOLD_PRED", "10")),
        stride_rows=int(os.getenv("GOLD_STRIDE", "120")),
        max_files=int(os.getenv("GOLD_MAX_FILES", "50")),
        max_windows=int(os.getenv("GOLD_MAX_WINDOWS_PER_FILE", "500")),
        cache_size=int(os.getenv("GOLD_FILE_CACHE", "64")),
        
        batch_size=int(os.getenv("GOLD_BATCH_SIZE", "16")),
        grad_accum=int(os.getenv("GOLD_GRAD_ACCUM", "2")),
        lr=float(os.getenv("GOLD_LR", "1e-4")),
        max_steps=int(os.getenv("GOLD_MAX_STEPS", "3000")),
        warmup_steps=int(os.getenv("GOLD_WARMUP", "100")),
        checkpoint_every=int(os.getenv("GOLD_CHECKPOINT_EVERY", "250")),
        log_every=int(os.getenv("GOLD_LOG_EVERY", "25")),
        val_split_pct=float(os.getenv("GOLD_VAL_SPLIT", "0.10")),
        
        out_dir=env_path("TEACHER_E_GOLD_OUTDIR", "backend/models/teacher_e/gold"),
        probe_only=args.probe_only,
        allow_fallback_codec=args.allow_fallback,
        target_col=target_col,
        covariate_cols=covariate_cols,
        future_covariate_cols=future_covariate_cols,
        max_files_limit=int(os.getenv("GOLD_MAX_FILES_LIMIT", "1000")),
        unfreeze_layernorms=os.getenv("UNFREEZE_LAYERNORMS", "0") == "1",
        unfreeze_last_blocks=int(os.getenv("UNFREEZE_LAST_BLOCKS", "0")),
        
        # Sigma Penalty Config
        sigma_band_low=float(os.getenv("SIGMA_BAND_LOW", "0.5")),
        sigma_band_high=float(os.getenv("SIGMA_BAND_HIGH", "2.0")),
        sigma_reg_weight=float(os.getenv("SIGMA_REG_WEIGHT", "1e-4"))
    )



class GoldFuturesWindowDataset(Dataset):
    def __init__(self, files: List[Path], required_cols: Tuple[str, ...],
                 context: int, pred: int, stride_rows: int, 
                 max_windows: int, seed: int, cache_size: int,
                 target_col: str = "close_adj"):
        self.files = files
        self.required_cols = required_cols
        self.col_map = {name: i for i, name in enumerate(required_cols)}
        self.target_col = target_col
        
        self.context = context
        self.pred = pred
        self.stride_rows = stride_rows
        self.max_windows = max_windows
        self.rng = np.random.RandomState(seed)
        
        # EAGER PREFETCH: Load all parquet files into RAM at init
        # This eliminates disk I/O bottleneck during training
        print(f"[Dataset] Prefetching {len(files)} parquet files into RAM...")
        self.data = {}  # {file_idx: np.ndarray}
        
        self.index = [] # List[(file_idx, start_row, timestamp_val)]
        
        for fi, fp in enumerate(self.files):
            try:
                # Load file schema and data
                schema = pl.scan_parquet(fp).collect_schema()
                exprs = []
                for col in self.required_cols:
                    if col in schema:
                        exprs.append(pl.col(col).cast(pl.Float32))
                    else:
                        exprs.append(pl.lit(0.0, dtype=pl.Float32).alias(col))
                
                df = pl.scan_parquet(fp).select(exprs).collect()
                arr = df.to_numpy()  # float32
                self.data[fi] = arr  # Store in RAM
                del df
                
                n = len(arr)
                max_start = n - (context + pred)
                if max_start <= 0:
                    continue
                
                starts = list(range(0, max_start, stride_rows))
                
                # Filter caps
                if len(starts) > max_windows:
                     starts = self.rng.choice(starts, size=max_windows, replace=False).tolist()
                
                for s in starts:
                    # Use row index as t_val for stable sorting
                    self.index.append((fi, int(s), int(s)))
                    
            except Exception as e:
                print(f"ERR loading {fp}: {e}")
        
        gc.collect()
        print(f"[Dataset] Prefetch complete. {len(self.data)} files loaded, {len(self.index)} windows indexed.")
        
        # Sort by time for temporal split
        try:
             self.index.sort(key=lambda x: x[2])
        except:
             print("WARN: Could not sort index by timestamp (mixed types?). Shuffling instead.")
             self.rng.shuffle(self.index)

    def split_validation(self, split_pct: float) -> Tuple['Subset', 'Subset']:
        total = len(self.index)
        val_size = int(total * split_pct)
        train_size = total - val_size
        
        # Temporal split: Train first, Val last
        # Enforce chronological integrity (metadata sort key is approximate if cross-file)
        train_idx = list(range(train_size))
        val_idx = list(range(train_size, total))
        
        return (
            torch.utils.data.Subset(self, train_idx),
            torch.utils.data.Subset(self, val_idx)
        )

    def get_quick_val_subset(self, val_ds: 'Subset', size: int = 1024, output_dir: Path = None) -> 'Subset':
        """
        Returns a deterministic, fixed-order subset of the validation set.
        Indices are saved/loaded from output_dir to ensure consistency across runs.
        """
        indices_path = output_dir / "val_quick_indices.json" if output_dir else None
        
        dataset_len = len(val_ds)
        actual_size = min(size, dataset_len)
        
        if indices_path and indices_path.exists():
            print(f"[Data] Loading deterministic quick-val indices from {indices_path}")
            import json
            with open(indices_path, 'r') as f:
                indices = json.load(f)
            # Integrity check
            if max(indices) >= dataset_len:
                print("[Data] WARN: Saved indices out of bounds. Regenerating.")
                indices = None
        else:
            indices = None

        if indices is None:
            # Deterministic selection: evenly spaced indices to cover the whole period
            # Or fixed random seed selection
            rng = np.random.RandomState(42) # Fixed seed
            indices = rng.choice(dataset_len, size=actual_size, replace=False).tolist()
            indices.sort() # Fixed order
            
            if indices_path:
                print(f"[Data] Saving quick-val indices to {indices_path}")
                import json
                with open(indices_path, 'w') as f:
                    json.dump(indices, f)
        
        return torch.utils.data.Subset(val_ds, indices)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        # Handle Subset wrappng index mapping?
        # Subset deals with it. 'idx' here is into self.index
        
        fi, s, _ = self.index[idx]
        
        # Direct RAM access (Prefetched)
        arr = self.data[fi]
        
        # Slicing
        ts_idx = self.col_map.get("timestamp")
        if ts_idx is None:
            ts_idx = self.col_map.get("date")
        start_row = s
        end_row = s + self.context + self.pred
        
        # Safety clip
        if end_row > len(arr):
             end_row = len(arr)
             
        subset = arr[start_row:end_row]
        
        if ts_idx is not None:
             # Exclude timestamp
             feat_indices = [i for i in range(arr.shape[1]) if i != ts_idx]
             feats = subset[:, feat_indices].astype(np.float32)
             
             # Locate target column index in filtered features
             # Robust Math:
             raw_target_idx = self.col_map.get(self.target_col)
             if raw_target_idx is not None:
                 if raw_target_idx < ts_idx:
                     target_feat_idx = raw_target_idx
                 elif raw_target_idx > ts_idx:
                     target_feat_idx = raw_target_idx - 1
                 else:
                     target_feat_idx = -1 
             else:
                 target_feat_idx = -1
        else:
             feats = subset.astype(np.float32)
             raw_target_idx = self.col_map.get(self.target_col)
             target_feat_idx = raw_target_idx if raw_target_idx is not None else -1
        
        # Pad if short
        if len(feats) < (self.context + self.pred):
             pad_len = (self.context + self.pred) - len(feats)
             feats = np.pad(feats, ((0, pad_len), (0, 0)))

        # Phase 2: Log-Return Transform
        if target_feat_idx >= 0:
            target_series = feats[:, target_feat_idx]
            
            if not self.target_col.startswith("ret"):
                 # Log-Return Transform for Prices
                 ref_idx = self.context - 1
                 if ref_idx < len(target_series):
                     ref_val = target_series[ref_idx]
                     if ref_val > 1e-6:
                          feats[:, target_feat_idx] = np.log(target_series / ref_val)
                     else:
                          feats[:, target_feat_idx] = 0.0
            else:
                 # Standardize Returns (Percent -> Decimal)
                 # Data is in percent (e.g. 1.0 = 1%), we want decimal (0.01)
                 feats[:, target_feat_idx] *= 0.01
                 
                 # Outlier Clipping (+/- 50%)
                 np.clip(feats[:, target_feat_idx], -0.5, 0.5, out=feats[:, target_feat_idx])
        else:
             pass

        # Sanitize NaNs/Infs final check
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

        future_target_1d = None
        target_10d = None
        if target_feat_idx >= 0:
            future_target_1d = feats[self.context:self.context + self.pred, target_feat_idx].astype(np.float32)
            if self.target_col.startswith("ret"):
                if np.any(future_target_1d[:10] < -0.5) or np.any(future_target_1d[:10] > 0.5):
                    raise ValueError("future_target_1d must be decimal and clipped before compounding.")
                if np.any(1 + future_target_1d[:10] <= 0):
                    raise ValueError("Invalid return encountered: 1 + r must be > 0.")
                target_10d = np.prod(1 + future_target_1d[:10]) - 1

        return {
            "x_float": feats[:self.context],
            "y_float": feats[self.context:],
            "future_target_1d": future_target_1d,
            "target_10d": target_10d,
            "group_id": fi,
        }

def _build_mask(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 3:
        mask = torch.isfinite(tensor).all(dim=-1)
    else:
        mask = torch.isfinite(tensor)
    return mask.to(dtype=torch.long)


class CollateFn:
    """Picklable collate function for multiprocessing DataLoader."""
    def __init__(self, feature_names: List[str], target_col: str,
                 covariate_cols: List[str], future_covariate_cols: List[str]):
        if target_col not in feature_names:
            raise ValueError(f"Target column '{target_col}' missing from features.")
        self.target_idx = feature_names.index(target_col)
        self.covariate_indices = [feature_names.index(c) for c in covariate_cols if c in feature_names]
        self.future_covariate_indices = [feature_names.index(c) for c in future_covariate_cols if c in feature_names]
    
    def __call__(self, batch):
        xs = np.stack([b["x_float"] for b in batch]) # [B, T, F]
        ys = np.stack([b["y_float"] for b in batch]) # [B, Pred, F]
        
        x_pt = torch.from_numpy(xs).float()
        y_pt = torch.from_numpy(ys).float()

        context_target = x_pt[:, :, self.target_idx]  # [B, T]
        future_target = y_pt[:, :, self.target_idx]    # [B, Pred]
        future_target_1d = future_target.clone()
        target_10d = None
        if future_target_1d.shape[1] >= 10:
            if torch.any(1 + future_target_1d[:, :10] <= 0):
                raise ValueError("Invalid return encountered: 1 + r must be > 0.")
            target_10d = (1 + future_target_1d[:, :10]).prod(dim=1) - 1
        past_covariates = x_pt[:, :, self.covariate_indices] if self.covariate_indices else None
        future_covariates = y_pt[:, :, self.future_covariate_indices] if self.future_covariate_indices else None

        context_mask = _build_mask(context_target)
        future_mask = _build_mask(future_target)

        if not torch.isfinite(context_target).all():
            raise ValueError("Non-finite values in context target.")
        if not torch.isfinite(future_target).all():
            raise ValueError("Non-finite values in future target.")
        if context_target.std() <= 1e-12 or torch.all(context_target == context_target.flatten()[0]):
            raise ValueError("Degenerate context target detected.")
        if future_target.std() <= 1e-12 or torch.all(future_target == future_target.flatten()[0]):
            raise ValueError("Degenerate future target detected.")
        
        # Build future_covariates_mask
        future_covariates_mask = None
        if future_covariates is not None:
            future_covariates_mask = _build_mask(future_covariates)

        # Build group_ids from file index
        group_ids = torch.tensor([b["group_id"] for b in batch], dtype=torch.long)

        return {
            "context": context_target,
            "context_mask": context_mask,
            "future_target": future_target,
            "future_target_mask": future_mask,
            "future_target_1d": future_target_1d,
            "target_10d": target_10d,
            "past_covariates": past_covariates,
            "future_covariates": future_covariates,
            "future_covariates_mask": future_covariates_mask,
            "group_ids": group_ids,
        }

def get_collate_fn(feature_names: List[str], target_col: str,
                   covariate_cols: List[str], future_covariate_cols: List[str]):
    return CollateFn(feature_names, target_col, covariate_cols, future_covariate_cols)

def compute_pinball_loss(y_true, quantile_preds, quantiles, quantile_weights=None):
    """
    y_true: [B] or [B, H]
    quantile_preds: [B, Q] or [B, Q, H'] where H' may be >= H (Chronos 2 outputs 32 steps)
    quantiles: List[float]
    quantile_weights: Optional[Tensor] [Q] - per-quantile weights (for curriculum)
    Returns: scalar mean loss
    """
    # Ensure correct shapes
    if y_true.ndim == 1:
        y_true = y_true.unsqueeze(1)
    if y_true.ndim == 2:
        y_true = y_true.unsqueeze(1) # [B, 1, H]
    if quantile_preds.ndim == 2:
        quantile_preds = quantile_preds.unsqueeze(-1)
    
    horizon = y_true.shape[-1] # Target horizon
    
    # Truncate predictions to match target horizon
    if quantile_preds.shape[-1] > horizon:
        quantile_preds = quantile_preds[..., :horizon]
    
    device = y_true.device
    q_tensor = torch.tensor(quantiles, device=device).view(1, -1, 1) # [1, Q, 1]
    
    # Loss = (y - y_hat) * q if y >= y_hat else (y_hat - y) * (1-q)
    #      = max( (y-y_hat)*q, (y_hat-y)*(1-q) )
    errors = y_true - quantile_preds
    loss_per_quantile = torch.max(errors * q_tensor, -errors * (1 - q_tensor)) # [B, Q, H]
    
    # Apply per-quantile weights if provided (Phase 3: Curriculum)
    if quantile_weights is not None:
        # quantile_weights: [Q]
        weights = quantile_weights.view(1, -1, 1).to(device) # [1, Q, 1]
        loss_per_quantile = loss_per_quantile * weights
    
    return loss_per_quantile.mean()


def _wilson_interval(p: float, n: int, z: float = 1.96) -> tuple:
    """Wilson score interval for binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    lo = max(0.0, centre - margin)
    hi = min(1.0, centre + margin)
    return (lo, hi)


def create_calibration_indices(val_dataset, size: int, output_dir: Path, seed: int = 137) -> list:
    """
    Create or load a persistent calibration index set for stable coverage metrics.
    Saves to output_dir/calib_indices.json for reproducibility across runs.
    """
    indices_path = output_dir / "calib_indices.json"
    dataset_len = len(val_dataset)
    requested_size = size

    # Try loading existing
    if indices_path.exists():
        print(f"[Calib] Loading calibration indices from {indices_path}")
        with open(indices_path, 'r') as f:
            indices = json.load(f)
        
        # Integrity check
        if indices:
            max_idx = max(indices)
            if max_idx >= dataset_len:
                print(f"[Calib] WARN: Saved indices out of bounds (max={max_idx} >= len={dataset_len}). Regenerating.")
            else:
                print(f"[Calib] Loaded {len(indices)} calibration indices (valid)")
                return indices
        else:
             print("[Calib] WARN: Empty indices file. Regenerating.")

    # Create new
    actual_size = min(requested_size, dataset_len)
    if actual_size < requested_size:
        print(f"[Calib] WARN: Dataset size ({dataset_len}) smaller than requested calibration slice ({requested_size}). Using all available.")

    rng = np.random.RandomState(seed)
    indices = rng.choice(dataset_len, size=actual_size, replace=False).tolist()
    indices.sort()
    
    # Final Validation
    if not indices:
         print("[Calib] ERROR: Generated empty indices list!")
         return []
    if max(indices) >= dataset_len:
         raise ValueError(f"[Calib] Generated indices out of bounds: max={max(indices)}, len={dataset_len}")

    indices_path.parent.mkdir(parents=True, exist_ok=True)
    with open(indices_path, 'w') as f:
        json.dump(indices, f)
    print(f"[Calib] Created {len(indices)} calibration indices -> {indices_path}")
    return indices


def compute_calib_stats(
    val_dataset, calib_indices: list, collate_fn, output_dir: Path,
    batch_size: int = 256
) -> dict:
    """
    Compute sigma0/mu0 from fixed calibration slice and persist to calib_stats.json.
    On resume: loads from file if present.
    Returns: {"mu0": float, "sigma0": float, "N_total": int, "N_used": int}
    """
    stats_path = output_dir / "calib_stats.json"

    # Try loading existing
    if stats_path.exists():
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        sigma0 = stats.get("sigma0", 0)
        mu0 = stats.get("mu0", 0)
        N_used = stats.get("N_used", stats.get("N", 0)) # Backwards compat
        
        if sigma0 > 0 and N_used > 0:
            print(f"[CalibStats] Loaded from {stats_path}: mu0={mu0:.6f}, sigma0={sigma0:.6f}, N_used={N_used}")
            # Ensure new fields (if missing) are populated mostly for consistency
            if "N_total" not in stats: stats["N_total"] = len(calib_indices)
            if "N_used" not in stats: stats["N_used"] = N_used
            return stats
        print("[CalibStats] WARN: Invalid saved stats, recomputing.")

    # Compute from scratch
    calib_subset = Subset(val_dataset, calib_indices)
    calib_dl = DataLoader(calib_subset, batch_size=batch_size, shuffle=False,
                          collate_fn=collate_fn, num_workers=0)

    all_targets = []
    for batch in calib_dl:
        t10d = batch.get("target_10d")
        if t10d is not None:
            # Filter NaNs/Infs
            valid_mask = torch.isfinite(t10d)
            if valid_mask.any():
                all_targets.append(t10d[valid_mask])
    
    N_total = len(calib_indices)
    
    if not all_targets:
        print("[CalibStats] WARN: No valid target_10d in calib slice, using fallback sigma0=0.032")
        stats = {"mu0": -0.00088, "sigma0": 0.032, "N_total": N_total, "N_used": 0}
    else:
        all_t = torch.cat(all_targets).float()
        mu0 = all_t.mean().item()
        sigma0 = all_t.std().item()  # population std (PyTorch default is sample std, close enough)
        N_used = int(all_t.numel())
        stats = {"mu0": mu0, "sigma0": sigma0, "N_total": N_total, "N_used": N_used}

    # Validation
    if stats["N_used"] < 0.8 * stats["N_total"]:
        print(f"[CalibStats] WARN: Low valid target count: {stats['N_used']}/{stats['N_total']} ({(stats['N_used']/stats['N_total']):.1%})")
    if stats["N_used"] < 512 and stats["N_total"] >= 512:
        print(f"[CalibStats] WARN: Very few valid samples ({stats['N_used']}) for calibration!")

    # Sanity check
    if stats["sigma0"] <= 0 or stats["sigma0"] > 0.5:
        print(f"[CalibStats] WARN: sigma0={stats['sigma0']:.6f} looks suspicious (expected 0.01-0.1). "
              f"Falling back to 0.032.")
        stats["sigma0"] = 0.032

    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"[CalibStats] Computed and saved: mu0={stats['mu0']:.6f}, sigma0={stats['sigma0']:.6f}, N_used={stats['N_used']}/{stats['N_total']}")
    return stats


def compute_calibration_metrics(
    model, val_dataset, calib_indices: list, quantiles: list, device: torch.device,
    collate_fn, batch_size: int = 64
) -> dict:
    """
    Compute stable calibration metrics over a fixed calibration slice.

    Returns dict with:
        - coverage_q01..q99: empirical coverage for key quantile levels
        - wilson_q01..q99_lo/hi: Wilson confidence intervals + half_width
        - avg_pinball: average pinball loss on calib slice
        - direction_acc: fraction where sign(mu) == sign(target)
        - pred_up_rate: fraction of predictions with mu > 0
        - true_up_rate: fraction of targets > 0
        - n_samples: total samples evaluated
    """
    model.eval()
    calib_subset = Subset(val_dataset, calib_indices)
    calib_dl = DataLoader(calib_subset, batch_size=batch_size, shuffle=False,
                          collate_fn=collate_fn, num_workers=0)

    # Key quantile indices
    key_quantiles = {'q01': 0.01, 'q10': 0.1, 'q50': 0.5, 'q90': 0.9, 'q99': 0.99}
    key_indices = {}
    for name, q_val in key_quantiles.items():
        try:
            key_indices[name] = quantiles.index(q_val)
        except ValueError:
            pass

    # Accumulators
    coverage_counts = {name: 0 for name in key_indices}
    total_samples = 0
    total_pinball = 0.0
    direction_correct = 0
    pred_up_count = 0
    true_up_count = 0
    n_batches = 0

    with torch.no_grad():
        for batch in calib_dl:
            context = batch["context"].to(device)
            context_mask = batch["context_mask"].to(device)
            target_10d = batch["target_10d"]
            if target_10d is None:
                continue
            target_10d = target_10d.to(device)
            past_covariates = batch.get("past_covariates")
            future_covariates = batch.get("future_covariates")
            if past_covariates is not None:
                past_covariates = past_covariates.to(device)
            if future_covariates is not None:
                future_covariates = future_covariates.to(device)

            B = context.shape[0]

            q_10d = model.predict_quantiles_10d(
                context,
                context_mask=context_mask,
                past_covariates=past_covariates,
                future_covariates=future_covariates,
            )

            # Coverage counts
            for name, idx in key_indices.items():
                coverage_counts[name] += (target_10d < q_10d[:, idx]).sum().item()

            # Pinball loss
            pinball = compute_pinball_loss(target_10d, q_10d, quantiles)
            total_pinball += pinball.item() * B

            # Direction accuracy using mu (from head internals)
            # Extract mu via head encoder -> mu_head
            outputs = None
            try:
                outputs = model(context, context_mask=context_mask,
                                past_covariates=past_covariates,
                                future_covariates=future_covariates)
            except Exception:
                pass
            rep = model._pool_context_representation(context, outputs)
            h = model.quantile_head.encoder(rep)
            mu = model.quantile_head.mu_head(h).squeeze(-1)  # [B]

            pred_up = (mu > 0)
            true_up = (target_10d > 0)
            direction_correct += (pred_up == true_up).sum().item()
            pred_up_count += pred_up.sum().item()
            true_up_count += true_up.sum().item()

            total_samples += B
            n_batches += 1

    model.train()

    if total_samples == 0:
        return {"n_samples": 0}

    result = {"n_samples": total_samples}

    # Coverages + Wilson CIs + half-widths
    for name, count in coverage_counts.items():
        p = count / total_samples
        result[f"coverage_{name}"] = p * 100.0
        lo, hi = _wilson_interval(p, total_samples)
        result[f"wilson_{name}_lo"] = lo * 100.0
        result[f"wilson_{name}_hi"] = hi * 100.0
        result[f"wilson_{name}_halfwidth"] = (hi - lo) * 50.0  # half of full CI width, in %

    result["avg_pinball"] = total_pinball / total_samples
    result["direction_acc"] = direction_correct / total_samples
    result["pred_up_rate"] = pred_up_count / total_samples
    result["true_up_rate"] = true_up_count / total_samples

    return result


def run_validation(model, dl, device, quantiles, desc="Val", pipeline=None):
    """
    Compute validation loss using **Weighted Pinball Loss** (Canonical).
    Internal model loss is computed for diagnostics only.
    We use UNWEIGHTED (Uniform) Pinball Loss for validation to ensure metric stability 
    across training (independent of curriculum stage).
    
    Args:
        pipeline: Chronos2Pipeline instance for generating quantile predictions
    """
    model.eval()
    total_pinball_loss = 0.0
    total_internal_loss = 0.0
    total_s90 = 0.0
    total_s99 = 0.0
    steps = 0
    
    # Validation uses uniform weights (1.0) for stability
    quantile_weights = torch.ones(len(quantiles), device=device)
    
    # Trackers for 10-Day Metrics
    total_10d_pinball = 0.0
    total_10d_direction_acc = 0.0
    total_10d_q10_hit = 0.0
    total_samples = 0
    
    # Standard metrics
    total_pinball_loss = 0.0
    total_internal_loss = 0.0
    steps = 0
    
    with torch.no_grad():
        for batch in dl:
            context = batch["context"].to(device)
            context_mask = batch["context_mask"].to(device)
            future_target = batch["future_target"].to(device) # [B, Pred] (Decimal Returns)
            future_target_mask = batch["future_target_mask"].to(device)
            target_10d = batch["target_10d"]
            if target_10d is not None:
                target_10d = target_10d.to(device)
            past_covariates = batch["past_covariates"]
            future_covariates = batch["future_covariates"]
            future_covariates_mask = batch.get("future_covariates_mask")
            group_ids = batch.get("group_ids")
            if past_covariates is not None: past_covariates = past_covariates.to(device)
            if future_covariates is not None: future_covariates = future_covariates.to(device)
            if future_covariates_mask is not None: future_covariates_mask = future_covariates_mask.to(device)
            if group_ids is not None: group_ids = group_ids.to(device)
            
            pred_len = future_target.shape[1]
            batch_size = context.shape[0]
            
            # --- 1. Standard Quantile Prediction (Daily) ---
            if pipeline is not None:
                context_np = context.cpu().unsqueeze(1).numpy()
                predictions = pipeline.predict(inputs=context_np, prediction_length=pred_len) 
                qp = torch.stack([pred.squeeze(0) for pred in predictions], dim=0).to(device) # [B, Q, Pred]
                
                # Fix #2: Prediction fingerprint (first batch only)
                if steps == 0:
                    print(f"[Fingerprint] Pred batch0: mean={qp.mean().item():.6f} std={qp.std().item():.6f}")
                
                # Daily Pinball
                p_loss = compute_pinball_loss(
                    future_target.squeeze(-1) if future_target.ndim == 3 else future_target,
                    qp, quantiles, quantile_weights=quantile_weights
                )
                total_pinball_loss += p_loss.item()

            if hasattr(model, "predict_quantiles_10d") and target_10d is not None:
                q_10d = model.predict_quantiles_10d(
                    context,
                    context_mask=context_mask,
                    past_covariates=past_covariates,
                    future_covariates=future_covariates,
                )
                
            # Check B: Fingerprint 10-day predictions (first batch only)
                if steps == 0:
                    # Test 1: Also fingerprint daily predictions from same batch for correlation
                    daily_preds_same_batch = qp if 'qp' in locals() else None
                    print(f"[Fingerprint 10d] q_10d batch0: mean={q_10d.mean().item():.6f} std={q_10d.std().item():.6f} shape={tuple(q_10d.shape)}")
                    if daily_preds_same_batch is not None:
                        print(f"[Test 1 Correlate] Daily preds (same batch): mean={daily_preds_same_batch.mean().item():.6f} std={daily_preds_same_batch.std().item():.6f}")
                    else:
                        print(f"[Test 1 Correlate] Daily preds not available in this batch (pipeline may not have run yet)")
                    
                    # SCALE CHECK: target_10d distribution vs q_10d distribution
                    print(f"\n[Scale Check] target_10d: mean={target_10d.mean().item():.6f} std={target_10d.std().item():.6f} min={target_10d.min().item():.6f} max={target_10d.max().item():.6f}")
                    print(f"[Scale Check] target_10d samples: {[round(v.item(), 6) for v in target_10d[:5]]}")
                    idx_50_check = quantiles.index(0.5)
                    q50_vals = q_10d[:, idx_50_check]
                    print(f"[Scale Check] q50 (10d): mean={q50_vals.mean().item():.6f} std={q50_vals.std().item():.6f} min={q50_vals.min().item():.6f} max={q50_vals.max().item():.6f}")
                    print(f"[Scale Check] Ratio q50/target: {(q50_vals.mean().item() / (target_10d.mean().item() + 1e-8)):.1f}x")
                    
                    # GUARDRAIL: Extract σ and μ statistics from quantile head
                    # Compute intermediate values from the head
                    with torch.no_grad():
                        # Get pooled representation (same as what quantile head uses)
                        pooled_repr = model._pool_context_representation(
                            context, 
                            model(context, context_mask=context_mask, past_covariates=past_covariates, future_covariates=future_covariates)
                        )
                        # Pass through quantile head encoder
                        h = model.quantile_head.encoder(pooled_repr)
                        # Extract μ and σ
                        mu = model.quantile_head.mu_head(h).squeeze(-1)
                        raw_sigma = model.quantile_head.sigma_head(h).squeeze(-1)
                        sigma = torch.nn.functional.softplus(raw_sigma) + model.quantile_head.sigma_floor
                        
                        # Log σ statistics
                        print(f"\\n[Guardrail σ] mean={sigma.mean().item():.6f} std={sigma.std().item():.6f} min={sigma.min().item():.6f} max={sigma.max().item():.6f}")
                        
                        # Log μ variability (should not be ~0)
                        print(f"[Guardrail μ] mean={mu.mean().item():.6f} std={mu.std().item():.6f} (μ_std should be non-zero = head using features)")
                        
                        # Log spread (q90 - q10)
                        idx_10 = quantiles.index(0.1)
                        idx_90 = quantiles.index(0.9)
                        spread = (q_10d[:, idx_90] - q_10d[:, idx_10]).mean().item()
                        print(f"[Guardrail spread] q90-q10: {spread:.6f} (should be ~0.05-0.15 for your data)")

                    # Diagnostic 1: Print sample row showing key quantile values
                    try:
                        idx_01 = quantiles.index(0.01)
                        idx_10 = quantiles.index(0.1)
                        idx_50 = quantiles.index(0.5)
                        idx_90 = quantiles.index(0.9)
                        idx_99 = quantiles.index(0.99)
                        sample_row = q_10d[0]  # First example
                        print(f"\n[Diagnostic 1] Sample quantiles (row 0):")
                        print(f"  q01={sample_row[idx_01].item():.6f}, q10={sample_row[idx_10].item():.6f}, q50={sample_row[idx_50].item():.6f}, q90={sample_row[idx_90].item():.6f}, q99={sample_row[idx_99].item():.6f}")
                        print(f"  Quantile indices: 0.01@{idx_01}, 0.10@{idx_10}, 0.50@{idx_50}, 0.90@{idx_90}, 0.99@{idx_99}")
                    except (ValueError, IndexError) as e:
                        print(f"[Diagnostic 1] Could not extract key quantiles: {e}")
                    
                    # Diagnostic 2: Check monotonicity violations
                    monotonic_rows = 0
                    for i in range(q_10d.shape[0]):
                        row = q_10d[i]
                        is_monotonic = all(row[j] <= row[j+1] for j in range(len(row)-1))
                        if is_monotonic:
                            monotonic_rows += 1
                    monotonic_pct = (monotonic_rows / q_10d.shape[0]) * 100
                    print(f"\n[Diagnostic 2] Monotonicity: {monotonic_pct:.1f}% of rows have non-decreasing quantiles")
                    
                    # Diagnostic 3: Empirical coverage for key quantiles
                    try:
                        coverages = {}
                        for q_name, q_val, idx in [("q01", 0.01, idx_01), ("q10", 0.1, idx_10), 
                                                     ("q50", 0.5, idx_50), ("q90", 0.9, idx_90), 
                                                     ("q99", 0.99, idx_99)]:
                            q_pred = q_10d[:, idx]
                            below = (target_10d < q_pred).float().mean().item()
                            coverages[q_name] = below * 100
                        print(f"\n[Diagnostic 3] Empirical coverage (% actual < predicted):")
                        for q_name in ["q01", "q10", "q50", "q90", "q99"]:
                            expected = float(q_name[1:])
                            actual = coverages[q_name]
                            print(f"  {q_name}: {actual:.1f}% (expected {expected:.0f}%)")
                    except Exception as e:
                        print(f"[Diagnostic 3] Could not compute coverages: {e}")
                    print()
                
                loss_primary = compute_pinball_loss(target_10d, q_10d, quantiles)
                total_10d_pinball += loss_primary.item()

                idx_50 = quantiles.index(0.5)
                idx_10 = quantiles.index(0.1)
                q50_10d = q_10d[:, idx_50]
                dir_correct = ((q50_10d > 0) == (target_10d > 0)).float().sum().item()
                total_10d_direction_acc += dir_correct
                q10_10d = q_10d[:, idx_10]
                total_10d_q10_hit += (target_10d < q10_10d).float().sum().item()
                total_samples += batch_size
                
                # Monitor Spreads (Day 10 Marginal)
                if qp.shape[2] >= 10:
                    day_10_preds = qp[:, :, 9] 
                    if len(quantiles) >= 21:
                         try:
                             i90 = quantiles.index(0.9)
                             i10 = quantiles.index(0.1)
                             s90 = (day_10_preds[:, i90] - day_10_preds[:, i10]).mean().item()
                             total_s90 += s90
                             
                             i99 = quantiles.index(0.99)
                             i01 = quantiles.index(0.01)
                             s99 = (day_10_preds[:, i99] - day_10_preds[:, i01]).mean().item()
                             total_s99 += s99
                         except ValueError:
                             pass

            # 3. Internal Loss
            out = model(
                context, 
                context_mask=context_mask, 
                future_target=future_target,
                future_target_mask=future_target_mask,
                past_covariates=past_covariates,
                future_covariates=future_covariates,
                group_ids=group_ids,
                num_output_patches=None
            )
            if hasattr(out, "loss") and out.loss is not None:
                total_internal_loss += out.loss.item()
            
            steps += 1
            
    model.train()
    
    avg_pinball_loss = total_pinball_loss / steps if steps > 0 else 0.0
    avg_internal_loss = total_internal_loss / steps if steps > 0 else 0.0
    
    # 10-Day Metrics Aggregation
    avg_10d_pinball = total_10d_pinball / steps if steps > 0 else 0.0
    avg_10d_direction = total_10d_direction_acc / total_samples if total_samples > 0 else 0.0
    avg_10d_q10 = total_10d_q10_hit / total_samples if total_samples > 0 else 0.0
    
    # Spreads
    avg_s90 = total_s90 / steps if steps > 0 else 0.0
    avg_s99 = total_s99 / steps if steps > 0 else 0.0
    
    # Standardize output keys (Safe Return)
    return {
        "pinball_loss": avg_pinball_loss, 
        #"internal_loss": avg_internal_loss, # Removed to avoid confusion
        "val_10d_pinball": avg_10d_pinball,
        "direction_10d": avg_10d_direction,
        "q10_10d": avg_10d_q10,
        "s90": avg_s90,
        "s99": avg_s99
    }

def main():
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--probe_only", action="store_true")
        parser.add_argument("--allow_fallback", action="store_true")
        parser.add_argument("--resume_adapter", type=str, default=None, 
                            help="Path to existing adapter checkpoint to resume from")
        args = parser.parse_args()

        cfg = load_config(args)
        run_id = run_recorder.init_run(
            pipeline_type="teacher_gold",
            trigger="manual",
            config={k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()},
            data_versions={"gold": "unknown", "silver": "unknown", "macro": "unknown", "universe": "unknown"},
            tags=["training", "teacher_gold"],
        )
        run_dir = run_recorder.run_paths.get_run_dir(run_id)
        outputs_dir = run_dir / "outputs"
        checkpoints_dir = run_dir / "checkpoints"
        outputs_dir.mkdir(parents=True, exist_ok=True)
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        cfg.out_dir = checkpoints_dir
        run_recorder.set_status(run_id, "RUNNING", stage="train", step="init")
        print(f"[Debug] Output Directory: {cfg.out_dir}")
        cfg.out_dir.mkdir(parents=True, exist_ok=True)
        
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        
        # 1. Model
        # Setup Logging
        log_path = outputs_dir / "training.log"
        import logging
        
        # Configure root logger to write to file and stdout
        class APIStreamLogger:
            def __init__(self, logger, level):
                self.logger = logger
                self.level = level
            def write(self, msg):
                if msg.strip(): self.logger.log(self.level, msg.strip())
            def flush(self): pass

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.FileHandler(log_path, mode='a', encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        print(f"[Logging] writing to {log_path}")
        
        class Tee(object):
            def __init__(self, *files):
                self.files = files
            def write(self, obj):
                for f in self.files:
                    try:
                        f.write(obj)
                        f.flush()
                    except Exception:
                        pass # Ignore encoding errors in tee
            def flush(self):
                for f in self.files:
                    try:
                        f.flush()
                    except Exception:
                        pass
        
        f = open(log_path, 'a', encoding='utf-8')
        sys.stdout = Tee(sys.stdout, f)
        sys.stderr = Tee(sys.stderr, f)
        
        print(f"[Run ID] {cfg.run_id}")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_wrapper, info = load_chronos_adapter(
            model_id=cfg.model_id,
            use_qlora=cfg.use_qlora,
            device=device,
            adapter_path=None,
            lora_config=cfg.lora_config
        )

        if not isinstance(model_wrapper, Chronos2NativeWrapper):
            print("ERROR: Loaded model does not expose Chronos native API.")
            return 1
        
        # Debug: Check trainable parameters
        trainable = [(n, p.shape) for n, p in model_wrapper.model.named_parameters() if p.requires_grad]
        frozen = [(n, p.shape) for n, p in model_wrapper.model.named_parameters() if not p.requires_grad]
        print(f"[Debug] Trainable params: {len(trainable)}")
        print(f"[Debug] Frozen params: {len(frozen)}")
        if trainable:
            print(f"[Debug] First 5 trainable: {[n for n, _ in trainable[:5]]}")
        else:
            print("[Debug] ⚠️ NO TRAINABLE PARAMETERS!")

        if cfg.probe_only:
            print("[Probe] Success.")
            return 0
        files_to_use = sorted(cfg.gold_parquet_dir.glob(cfg.gold_glob))
        if cfg.max_files_limit:
            files_to_use = files_to_use[:cfg.max_files_limit]
            
        # 2. Dataset
        dataset = GoldFuturesWindowDataset(
            files=files_to_use,
            required_cols=cfg.required_cols,
            context=cfg.context,
            pred=cfg.pred,
            stride_rows=cfg.stride_rows, 
            max_windows=1000, # Per file cap
            seed=42,
            cache_size=cfg.cache_size,
            target_col=cfg.target_col
        )
        
        train_ds, val_ds = dataset.split_validation(cfg.val_split_pct)
        
        # Feature names for collate_fn
        feature_names = [c for c in cfg.required_cols if c not in ("timestamp", "date")]
        
        train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, 
                              collate_fn=get_collate_fn(feature_names, cfg.target_col, cfg.covariate_cols, cfg.future_covariate_cols), 
                              num_workers=4, pin_memory=True, persistent_workers=True)
        
        val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            collate_fn=get_collate_fn(feature_names, cfg.target_col, cfg.covariate_cols, cfg.future_covariate_cols), 
                            num_workers=4, pin_memory=True, persistent_workers=True)

        # 3. Model
        quantiles = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
        
        model_wrapper, model_info = load_chronos_adapter(
            model_id="amazon/chronos-2",
            use_qlora=cfg.use_qlora,
            device=device,
            lora_config=cfg.lora_config
        )
        model_wrapper.to(device)
        
        # Resume from existing adapter checkpoint if specified
        if args.resume_adapter:
            resume_path = Path(args.resume_adapter)
            if resume_path.exists():
                print(f"[Resume] Loading adapter weights from: {resume_path}")
                # Try safetensors first, then fall back to .bin
                adapter_path_st = resume_path / "adapter_model.safetensors"
                model_path_st = resume_path / "model.safetensors"  # Full model checkpoint
                adapter_path_bin = resume_path / "adapter_model.bin"
                pytorch_model = resume_path / "pytorch_model.bin"
                
                state_dict = None
                if adapter_path_st.exists():
                    from safetensors.torch import load_file
                    state_dict = load_file(str(adapter_path_st))
                    print(f"[Resume] Using safetensors format (adapter_model)")
                elif model_path_st.exists():
                    from safetensors.torch import load_file
                    state_dict = load_file(str(model_path_st))
                    print(f"[Resume] Using safetensors format (model.safetensors)")
                elif adapter_path_bin.exists():
                    state_dict = torch.load(str(adapter_path_bin), map_location=device)
                    print(f"[Resume] Using bin format (adapter_model.bin)")
                elif pytorch_model.exists():
                    state_dict = torch.load(str(pytorch_model), map_location=device)
                    print(f"[Resume] Using bin format (pytorch_model.bin)")

                    
                if state_dict:
                    # Filter and load only LoRA weights
                    model_state = model_wrapper.model.state_dict()
                    loaded_keys = []
                    for k, v in state_dict.items():
                        if k in model_state:
                            model_state[k] = v
                            loaded_keys.append(k)
                    model_wrapper.model.load_state_dict(model_state, strict=False)
                    print(f"[Resume] Loaded {len(loaded_keys)} adapter weights")
                else:
                    print(f"[Resume] WARN: No adapter checkpoint found in {resume_path}")
            else:
                print(f"[Resume] WARN: Path {resume_path} does not exist, starting fresh")

        
        print(f"[Info] Model Loaded on {device}")
        
        try:
             # Load pipeline with base config
             pipeline = Chronos2Pipeline.from_pretrained(cfg.model_id)
             # INJECT TRAINED MODEL (Adapter)
             # This ensures validation uses the actual trained weights!
             pipeline.model = model_wrapper.model
             pipeline.model.to(device)
             print(f"[Validation] Pipeline loaded and injected with trained model.")
        except Exception as e:
             print(f"[Validation] WARN: Could not load pipeline: {e}")
             pipeline = None

        # 4. Training Setup
        
        # --- Partial Unfreeze Logic ---
        if cfg.unfreeze_layernorms:
            print("[Unfreeze] Unfreezing LayerNorm parameters...")
            count_ln = 0
            for name, param in model_wrapper.model.named_parameters():
                if "layer_norm" in name or "layernorm" in name:
                     param.requires_grad = True
                     count_ln += 1
            print(f"[Unfreeze] Unfroze {count_ln} LayerNorm params.")

        if cfg.unfreeze_last_blocks > 0:
            print(f"[Unfreeze] Unfreezing last {cfg.unfreeze_last_blocks} transformer blocks...")
            # Detect model type and block structure
            # For T5: encoder.block.N
            
            # Find max block index
            max_block_idx = -1
            for name, _ in model_wrapper.model.named_parameters():
                # T5 pattern: encoder.block.11.layer...
                parts = name.split('.')
                if "block" in parts:
                    try:
                        idx = int(parts[parts.index("block") + 1])
                        max_block_idx = max(max_block_idx, idx)
                    except (ValueError, IndexError):
                        pass
            
            if max_block_idx >= 0:
                start_unsqueezed_idx = max_block_idx - cfg.unfreeze_last_blocks + 1
                if start_unsqueezed_idx < 0: start_unsqueezed_idx = 0
                
                print(f"[Unfreeze] Targeting blocks {start_unsqueezed_idx} to {max_block_idx}")
                
                count_blk = 0
                for name, param in model_wrapper.model.named_parameters():
                    parts = name.split('.')
                    if "block" in parts:
                        try:
                            idx = int(parts[parts.index("block") + 1])
                            if idx >= start_unsqueezed_idx:
                                param.requires_grad = True
                                count_blk += 1
                        except:
                            pass
                print(f"[Unfreeze] Unfroze {count_blk} params in last {cfg.unfreeze_last_blocks} blocks.")
            else:
                print("[Unfreeze] WARN: Could not detect block structure (no 'block' in param names).")

        trainable_params = [p for p in model_wrapper.parameters() if p.requires_grad]
        
        # Track Unfrozen Base Params (Non-LoRA)
        unfrozen_param_names = []
        for name, param in model_wrapper.model.named_parameters():
            if param.requires_grad and "lora" not in name.lower():
                unfrozen_param_names.append(name)
        print(f"[Checkpoint] Tracking {len(unfrozen_param_names)} base parameters for 'base_delta.pt'")

        # Separate param groups: LoRA, head (μ/σ), shape (z_τ), temperature, base
        lora_params = [p for n, p in model_wrapper.model.named_parameters() if p.requires_grad and "lora" in n.lower()]
        base_params = [p for n, p in model_wrapper.model.named_parameters() if p.requires_grad and "lora" not in n.lower()]

        # Quantile head: separate μ/σ encoder+heads from shape and temperature
        head_mu_sigma_params = []
        shape_params = []
        temperature_params = []
        for n, p in model_wrapper.named_parameters():
            if not p.requires_grad or "quantile_head" not in n:
                continue
            if "shape." in n:
                shape_params.append(p)
            elif "logT" in n:
                temperature_params.append(p)
            else:
                head_mu_sigma_params.append(p)

        lr_lora = cfg.lr
        lr_head = cfg.lr
        lr_shape = cfg.lr * 0.1
        lr_temp = cfg.lr * 0.05

        param_groups = [
            {"params": lora_params, "lr": lr_lora, "name": "lora"},
            {"params": head_mu_sigma_params, "lr": lr_head, "name": "head_mu_sigma"},
        ]
        if shape_params:
            param_groups.append({"params": shape_params, "lr": lr_shape, "name": "shape_z_tau"})
        if temperature_params:
            param_groups.append({"params": temperature_params, "lr": lr_temp, "name": "temperature"})
        if base_params:
            base_lr = cfg.lr / 20.0
            param_groups.append({"params": base_params, "lr": base_lr, "weight_decay": 0.01, "name": "base_unfrozen"})

        print(f"[Optimizer] Param groups:")
        print(f"  LoRA: {len(lora_params)} tensors @ lr={lr_lora}")
        print(f"  Head (μ/σ): {len(head_mu_sigma_params)} tensors @ lr={lr_head}")
        print(f"  Shape (z_τ): {len(shape_params)} tensors @ lr={lr_shape}")
        print(f"  Temperature: {len(temperature_params)} tensors @ lr={lr_temp}")
        if base_params:
            print(f"  Base: {len(base_params)} tensors @ lr={base_lr}")
        
        opt = torch.optim.AdamW(param_groups)
        scheduler = get_linear_schedule_with_warmup(
            opt, num_warmup_steps=cfg.warmup_steps, num_training_steps=cfg.max_steps
        )
        
        # Verification: Check quantile_head params are initialized and in optimizer
        print("\n[Verify] Checking quantile_head initialization:")
        for name, param in model_wrapper.named_parameters():
            if "quantile_head" in name.lower():
                # Check if in optimizer
                in_opt = any(param is p for group in opt.param_groups for p in group['params'])
                print(f"  {name}: shape={tuple(param.shape)}, requires_grad={param.requires_grad}, in_optimizer={in_opt}")
        print()
        
        # Helper: Robust Checkpointer (saves LoRA + base delta + head + shape + temperature + optimizer)
        def save_checkpoint_extended(target_dir: Path, step_meta: dict):
            target_dir.mkdir(parents=True, exist_ok=True)

            # 1. Save LoRA Adapter (Standard)
            model_wrapper.model.save_pretrained(target_dir)

            # 2. Save Unfrozen Base Weights (Base Delta)
            if unfrozen_param_names:
                base_delta = {}
                state_dict = model_wrapper.model.state_dict()
                for name in unfrozen_param_names:
                    if name in state_dict:
                        base_delta[name] = state_dict[name].cpu()
                torch.save(base_delta, target_dir / "base_delta.pt")

            # 3. Save quantile head state dict (includes shape + temperature)
            head_state = model_wrapper.quantile_head.state_dict()
            torch.save({k: v.cpu() for k, v in head_state.items()}, target_dir / "quantile_head.pt")

            # 4. Save optimizer state for resume
            torch.save(opt.state_dict(), target_dir / "optimizer.pt")

            # 5. Save run_start_step for resume-safe shape reg decay
            with open(target_dir / "run_start_step.json", "w") as f:
                json.dump({"run_start_step": run_start_step, "saved_at_step": step_meta.get("step")}, f)

            # 6. Save Manifest
            with open(target_dir / "manifest.json", "w") as f:
                json.dump(step_meta, f, indent=2)
            run_recorder.register_checkpoint(
                run_id,
                checkpoint_path=str(target_dir),
                step=int(step_meta.get("step", 0)),
                epoch=float(step_meta.get("epoch", 0.0) or 0.0),
                component="chronos_lora",
                meta=step_meta,
            )

        print(f"Starting Training: {cfg.run_id}")
        
        # --- RESUME BASE DELTA + HEAD + OPTIMIZER IF EXISTS (Post-Unfreeze) ---
        if args.resume_adapter:
            resume_path = Path(args.resume_adapter)

            # Base delta
            base_delta_path = resume_path / "base_delta.pt"
            if base_delta_path.exists():
                print(f"[Resume] Loading Base Delta from {base_delta_path}...")
                base_delta = torch.load(base_delta_path, map_location=device)
                model_wrapper.model.load_state_dict(base_delta, strict=False)
                print(f"[Resume] Applied {len(base_delta)} base param updates.")
            elif len(unfrozen_param_names) > 0:
                print(f"[Resume] WARN: Unfrozen params detected but no 'base_delta.pt' found in {resume_path}!")

            # Quantile head (includes shape + temperature)
            head_path = resume_path / "quantile_head.pt"
            if head_path.exists():
                print(f"[Resume] Loading quantile head from {head_path}...")
                head_state = torch.load(head_path, map_location=device)
                model_wrapper.quantile_head.load_state_dict(head_state)
                print(f"[Resume] Loaded quantile head ({len(head_state)} tensors)")
            else:
                print(f"[Resume] No quantile_head.pt found, using fresh head init.")

            # Optimizer state
            opt_path = resume_path / "optimizer.pt"
            if opt_path.exists():
                print(f"[Resume] Loading optimizer state from {opt_path}...")
                try:
                    opt.load_state_dict(torch.load(opt_path, map_location=device))
                    print(f"[Resume] Optimizer state restored.")
                except Exception as e:
                    print(f"[Resume] WARN: Could not restore optimizer state: {e}")

        # Prepare Validation Subsets
        val_quick_ds = dataset.get_quick_val_subset(val_ds, size=50, output_dir=cfg.out_dir)
        val_quick_dl = DataLoader(val_quick_ds, batch_size=cfg.batch_size, shuffle=False,
                                  collate_fn=get_collate_fn(feature_names, cfg.target_col, cfg.covariate_cols, cfg.future_covariate_cols),
                                  num_workers=0)

        # Create persistent calibration slice (N>=2048) for stable coverage metrics
        calib_size = int(os.getenv("CALIB_SLICE_SIZE", "2048"))
        calib_indices = create_calibration_indices(val_ds, size=calib_size, output_dir=cfg.out_dir)
        calib_collate = get_collate_fn(feature_names, cfg.target_col, cfg.covariate_cols, cfg.future_covariate_cols)

        # --- Part A: Compute data-driven sigma0/mu0 from calib slice ---
        calib_stats = compute_calib_stats(val_ds, calib_indices, calib_collate, output_dir=cfg.out_dir)
        sigma0 = calib_stats["sigma0"]
        mu0 = calib_stats["mu0"]
        print(f"[CalibStats] Using sigma0={sigma0:.6f}, mu0={mu0:.6f}")

        # --- Part D: Separate warmup for shape vs temperature ---
        shape_warmup_steps = int(os.getenv("SHAPE_WARMUP_STEPS", "500"))
        temp_warmup_steps = int(os.getenv("TEMP_WARMUP_STEPS", "2000"))
        shape_z_frozen = shape_warmup_steps > 0
        temp_frozen = temp_warmup_steps > 0

        # Freeze shape initially
        if shape_z_frozen:
            for n, p in model_wrapper.named_parameters():
                if "quantile_head.shape." in n:
                    p.requires_grad = False
            print(f"[Warmup] Shape (z_τ) frozen for first {shape_warmup_steps} global steps")

        # Freeze temperature initially (separate from shape)
        if temp_frozen:
            for n, p in model_wrapper.named_parameters():
                if "quantile_head.logT" in n:
                    p.requires_grad = False
            print(f"[Warmup] Temperature frozen for first {temp_warmup_steps} global steps")

        # --- Part B: σ band penalty config ---
        sigma_reg_weight = cfg.sigma_reg_weight
        sigma_band_low = cfg.sigma_band_low
        sigma_band_high = cfg.sigma_band_high

        # Shape regularization config
        shape_reg_weight_init = float(os.getenv("SHAPE_REG_WEIGHT", "1e-3"))
        shape_reg_decay_steps = int(os.getenv("SHAPE_REG_DECAY_STEPS", "5000"))
        lambda_10d = float(os.getenv("LAMBDA_10D", "0.5"))

        # --- Part E: run_start_step for resume-safe shape reg decay ---
        run_start_step = 0
        if args.resume_adapter:
            resume_path = Path(args.resume_adapter)
            run_start_path = resume_path / "run_start_step.json"
            if run_start_path.exists():
                with open(run_start_path, 'r') as f:
                    run_start_step = json.load(f).get("run_start_step", 0)
                print(f"[Resume] Loaded run_start_step={run_start_step}")
            else:
                # Best effort: use the step from the manifest if available
                manifest_path = resume_path / "manifest.json"
                if manifest_path.exists():
                    with open(manifest_path, 'r') as f:
                        run_start_step = json.load(f).get("step", 0)
                    print(f"[Resume] Inferred run_start_step={run_start_step} from manifest")

        step = 0
        running_loss = 0.0

        # Infinite loop pattern for step-based training
        train_iter = iter(train_dl)

        # Phase 1: Valid Tracking
        best_val_loss = float("inf")
        quick_val_ema = None
        stall_count = 0
        min_delta = 5e-4 # User specified: 0.0005
        ema_alpha = 0.3
        early_stop_patience = 8 if cfg.checkpoint_every < 1000 else 3 # Adaptive patience

        
        while step < cfg.max_steps * cfg.grad_accum:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_dl)
                batch = next(train_iter)
            
            context = batch["context"].to(device)
            context_mask = batch["context_mask"].to(device)
            future_target = batch["future_target"].to(device)
            future_target_mask = batch["future_target_mask"].to(device)
            target_10d = batch["target_10d"]
            if target_10d is not None:
                target_10d = target_10d.to(device)
            past_covariates = batch["past_covariates"]
            future_covariates = batch["future_covariates"]
            future_covariates_mask = batch.get("future_covariates_mask")
            group_ids = batch.get("group_ids")
            if past_covariates is not None: past_covariates = past_covariates.to(device)
            if future_covariates is not None: future_covariates = future_covariates.to(device)
            if future_covariates_mask is not None: future_covariates_mask = future_covariates_mask.to(device)
            if group_ids is not None: group_ids = group_ids.to(device)
            
            # Forward
            if step == 0:
                 print(f"[Debug] Input device: {context.device}")
                 print(f"[Debug] future_covariates shape: {future_covariates.shape if future_covariates is not None else None}")
                 print(f"[Debug] group_ids shape: {group_ids.shape if group_ids is not None else None}")
            
            out = model_wrapper(
                context,
                context_mask=context_mask,
                future_target=future_target,
                future_target_mask=future_target_mask,
                past_covariates=past_covariates,
                future_covariates=future_covariates,
                group_ids=group_ids,
            )
            
            # Phase 3: Quantile Curriculum (Weighted Pinball Loss)
            # Chronos 2 outputs quantile predictions [B, Q, H]
            # We manually compute weighted Pinball Loss
            
            # Define quantile weights based on curriculum stage
            num_quantiles = len(quantiles)
            quantile_weights = torch.ones(num_quantiles, device=device)
            
            # Stage A: Down-weight tails (first 30% of training)
            # Tails = P01 (idx 0), P05 (idx 1), P95 (idx 19), P99 (idx 20)
            # Assuming quantiles = [0.01, 0.05, 0.1, ..., 0.95, 0.99]
            curr_global_step_est = (step + 1) // cfg.grad_accum
            curriculum_threshold = int(cfg.max_steps * 0.3)  # 30% of total steps
            
            if curr_global_step_est < curriculum_threshold:
                # Stage A: Core quantiles weight=1.0, Tails weight=0.25
                tail_weight = 0.25
                # Identify tail indices (P01, P05, P95, P99)
                # quantiles = [0.01, 0.05, 0.1, 0.15, ..., 0.85, 0.9, 0.95, 0.99]
                # Tails: indices where q < 0.10 or q > 0.90
                for i, q in enumerate(quantiles):
                    if q < 0.10 or q > 0.90:
                        quantile_weights[i] = tail_weight
            # Stage B (>= threshold): Uniform weights (already initialized to 1.0)
            
            loss_1d = None
            if hasattr(out, 'quantile_preds') and out.quantile_preds is not None:
                loss_1d = compute_pinball_loss(
                    future_target.squeeze(-1) if future_target.ndim == 3 else future_target,
                    out.quantile_preds,
                    quantiles,
                    quantile_weights=quantile_weights
                )

            loss_10d = None
            # CRITICAL: Call 10-day head during training to supervise it
            # BYPASS: Skip if quantile head disabled (encoder extraction broken)
            if cfg.enable_q10d_head and hasattr(model_wrapper, "predict_quantiles_10d") and target_10d is not None:
                q_10d_train = model_wrapper.predict_quantiles_10d(
                    context,
                    context_mask=context_mask,
                    past_covariates=past_covariates,
                    future_covariates=future_covariates,
                )
                loss_10d = compute_pinball_loss(
                    target_10d,
                    q_10d_train,
                    quantiles,
                    quantile_weights=quantile_weights
                )

                # --- Part B: σ soft clamp band penalty (data-driven anchor sigma0) ---
                # Compute σ through the head (WITH gradients for proper backprop)
                pooled_rep = model_wrapper._pool_context_representation(context, out)
                h_reg = model_wrapper.quantile_head.encoder(pooled_rep)
                raw_sigma_reg = model_wrapper.quantile_head.sigma_head(h_reg).squeeze(-1)
                sigma_reg = F.softplus(raw_sigma_reg) + model_wrapper.quantile_head.sigma_floor
                sigma_mean = sigma_reg.mean()

                low_bound = sigma0 * sigma_band_low
                high_bound = sigma0 * sigma_band_high
                log_sigma_mean = torch.log(sigma_mean + 1e-8)
                log_low = math.log(low_bound)
                log_high = math.log(high_bound)

                if sigma_mean.item() < low_bound:
                    sigma_penalty = (log_low - log_sigma_mean) ** 2
                elif sigma_mean.item() > high_bound:
                    sigma_penalty = (log_sigma_mean - log_high) ** 2
                else:
                    sigma_penalty = torch.tensor(0.0, device=device)
                reg_sigma = sigma_reg_weight * sigma_penalty
                loss_10d = loss_10d + reg_sigma

                # --- Part E: Shape regularizer with resume-safe elapsed decay ---
                if not shape_z_frozen and hasattr(model_wrapper.quantile_head, 'shape'):
                    z_current = model_wrapper.quantile_head.shape()
                    z_init_norm = model_wrapper.quantile_head.shape.z_init_normalized
                    reg_z = ((z_current - z_init_norm) ** 2).mean()
                    # Decay using elapsed steps since run_start_step (resume-safe)
                    elapsed = curr_global_step_est - run_start_step
                    # Clamp elapsed to >= 0
                    if elapsed < 0: elapsed = 0
                    decay_frac = max(0.0, 1.0 - elapsed / max(shape_reg_decay_steps, 1))
                    loss_10d = loss_10d + shape_reg_weight_init * decay_frac * reg_z

            # Fallback: check if model wrapper directly returned q_10d
            elif hasattr(out, "q_10d") and out.q_10d is not None and target_10d is not None:
                loss_10d = compute_pinball_loss(
                    target_10d,
                    out.q_10d,
                    quantiles,
                    quantile_weights=quantile_weights
                )

            # DEBUG: Log which loss branch is taken (first 3 steps only)
            if step < 3:
                has_qp = hasattr(out, 'quantile_preds') and out.quantile_preds is not None
                has_q10 = hasattr(out, 'q_10d') and out.q_10d is not None
                has_loss = hasattr(out, 'loss') and out.loss is not None
                out_type = type(out).__name__
                out_attrs = [a for a in dir(out) if not a.startswith('_')]
                print(f"[Debug Loss] step={step} out_type={out_type} has_qp={has_qp} has_q10={has_q10} has_loss={has_loss}")
                print(f"[Debug Loss] out attrs: {out_attrs[:15]}")
                if has_loss:
                    print(f"[Debug Loss] out.loss = {out.loss.item():.6f}")

            if loss_10d is not None and loss_1d is not None:
                loss = loss_1d + lambda_10d * loss_10d
            elif loss_10d is not None:
                loss = loss_10d
            elif loss_1d is not None:
                loss = loss_1d
            else:
                loss = out.loss if hasattr(out, "loss") else None
                if loss is None:
                    print("[Warn] Model returned no loss, skipping batch.")
                    loss = torch.tensor(0.0, device=device, requires_grad=True)

            loss_10d_value = loss_10d.item() if loss_10d is not None else None
            loss_1d_value = loss_1d.item() if loss_1d is not None else None
            
            # Accumulate
            loss_scaled = loss / cfg.grad_accum
            loss_scaled.backward()
            
            running_loss += loss.item()
            
            if (step + 1) % cfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model_wrapper.parameters(), 1.0)
                opt.step()
                scheduler.step()
                opt.zero_grad()

                curr_global_step = (step + 1) // cfg.grad_accum

                # Warmup: unfreeze shape and temperature at their respective steps
                if shape_z_frozen and curr_global_step >= shape_warmup_steps:
                    print(f"[Warmup] Unfreezing shape (z_τ) at step {curr_global_step}")
                    for n, p in model_wrapper.named_parameters():
                        if "quantile_head.shape." in n:
                            p.requires_grad = True
                    shape_z_frozen = False

                if temp_frozen and curr_global_step >= temp_warmup_steps:
                    print(f"[Warmup] Unfreezing temperature at step {curr_global_step}")
                    for n, p in model_wrapper.named_parameters():
                        if "quantile_head.logT" in n:
                            p.requires_grad = True
                    temp_frozen = False

                if curr_global_step % cfg.log_every == 0:
                    avg_loss = running_loss / cfg.grad_accum / cfg.log_every
                    # Specifically for first steps where log_every=1
                    if cfg.log_every == 1: avg_loss = running_loss / cfg.grad_accum
                    
                    current_lr = scheduler.get_last_lr()[0]
                    
                    # Log extras removed to reduce noise (training batch spreads are volatile)
                    print(f"Step {curr_global_step} | Loss: {avg_loss:.4f} | LR: {current_lr:.2e}")
                    run_recorder.emit_metric(
                        run_id,
                        step=curr_global_step,
                        epoch=None,
                        metrics={"train_loss": avg_loss, "lr": current_lr},
                    )
                    run_recorder.set_status(
                        run_id,
                        "RUNNING",
                        stage="train",
                        step=f"step_{curr_global_step}",
                    )
                    running_loss = 0.0
                    
                # Quick Val (Every 500 steps)
                if curr_global_step % 500 == 0: 
                    print(f"--- QuickVal (Step {curr_global_step}) ---")
                    qv_metrics = run_validation(model_wrapper, val_quick_dl, device, quantiles, desc="QuickVal", pipeline=pipeline)
                    
                    # Safe Unpack
                    qv_loss = qv_metrics.get('pinball_loss', float('inf'))
                    qv_10d = qv_metrics.get('val_10d_pinball', 0.0)
                    qv_s90 = qv_metrics.get('s90', 0.0)
                    qv_s99 = qv_metrics.get('s99', 0.0)

                    # Get current head diagnostics for QuickVal log
                    with torch.no_grad():
                        T_qv = model_wrapper.quantile_head.temperature_value

                    print(f"QuickVal Pinball: {qv_loss:.4f} | 10d Pinball: {qv_10d:.4f} | S90: {qv_s90:.4f} | T={T_qv:.3f}")
                    run_recorder.emit_metric(
                        run_id,
                        step=curr_global_step,
                        epoch=None,
                        metrics={
                            "quickval_pinball": qv_loss,
                            "quickval_10d_pinball": qv_10d,
                            "quickval_s90": qv_s90,
                            "quickval_s99": qv_s99,
                            "temperature_T": T_qv,
                        },
                    )
                    
                    # Update EMA
                    if quick_val_ema is None:
                        quick_val_ema = qv_loss
                    else:
                        quick_val_ema = ema_alpha * qv_loss + (1 - ema_alpha) * quick_val_ema
                    print(f"QuickVal EMA: {quick_val_ema:.4f}")

                    # Save LAST checkpoint
                    save_path = cfg.out_dir / f"checkpoint-{curr_global_step}" 
                    save_checkpoint_extended(save_path, {
                        "step": curr_global_step,
                        "type": "checkpoint",
                        "metrics": qv_metrics
                    })
                    
                    # Also update 'last' pointer
                    save_checkpoint_extended(cfg.out_dir / "last", {
                        "step": curr_global_step,
                        "type": "last",
                        "metrics": qv_metrics
                    })

                # Full Validation (Every 1000 steps initially as per Plan Phase 5)
                if curr_global_step > 0 and curr_global_step % 1000 == 0:
                     print(f"--- FullVal (Step {curr_global_step}) ---")

                     # Re-inject trained model into pipeline
                     if pipeline is not None:
                         pipeline.model = model_wrapper.model

                     # Parameter fingerprint
                     for pname, pparam in model_wrapper.model.named_parameters():
                         if "lora_A" in pname and pparam.requires_grad:
                             print(f"[Fingerprint] {pname}: mean={pparam.data.mean().item():.6f} std={pparam.data.std().item():.6f}")
                             break

                     fv_metrics = run_validation(model_wrapper, val_dl, device, quantiles, pipeline=pipeline)

                     # Extract Metrics (Safe)
                     val_loss = fv_metrics.get('pinball_loss', float('inf'))
                     val_10d_pinball = fv_metrics.get('val_10d_pinball', 0.0)
                     val_10d_dir = fv_metrics.get('direction_10d', 0.0)
                     val_10d_q10 = fv_metrics.get('q10_10d', 0.0)
                     val_s90 = fv_metrics.get('s90', 0.0)
                     val_s99 = fv_metrics.get('s99', 0.0)

                     # --- Stable Calibration Metrics (N>=2048 slice) ---
                     print(f"\n--- Calibration Metrics (N={len(calib_indices)}) ---")
                     calib_metrics = compute_calibration_metrics(
                         model_wrapper, val_ds, calib_indices, quantiles, device,
                         collate_fn=calib_collate, batch_size=64
                     )
                     n_calib = calib_metrics.get("n_samples", 0)
                     if n_calib > 0:
                         for qname in ["q01", "q10", "q50", "q90", "q99"]:
                             cov = calib_metrics.get(f"coverage_{qname}", 0)
                             lo = calib_metrics.get(f"wilson_{qname}_lo", 0)
                             hi = calib_metrics.get(f"wilson_{qname}_hi", 0)
                             hw = calib_metrics.get(f"wilson_{qname}_halfwidth", 0)
                             expected = float(qname[1:])
                             print(f"  {qname}: {cov:.1f}% (expected {expected:.0f}%) "
                                   f"[95% CI: {lo:.1f}%-{hi:.1f}%, ±{hw:.1f}%]")
                         calib_pinball = calib_metrics.get("avg_pinball", 0)
                         calib_dir = calib_metrics.get("direction_acc", 0)
                         pred_up = calib_metrics.get("pred_up_rate", 0)
                         true_up = calib_metrics.get("true_up_rate", 0)
                         print(f"  Calib Pinball: {calib_pinball:.6f} | Direction Acc: {calib_dir:.2%}")
                         print(f"  Pred Up Rate: {pred_up:.2%} | True Up Rate: {true_up:.2%}")

                         # Calibration sanity check (after warmup)
                         if curr_global_step > shape_warmup_steps:
                             cov_q50 = calib_metrics.get("coverage_q50", 50.0)
                             cov_q10 = calib_metrics.get("coverage_q10", 10.0)
                             if cov_q50 < 40.0 or cov_q50 > 60.0:
                                 print(f"  [WARN] q50 coverage {cov_q50:.1f}% outside [40%, 60%] — calibration may be off")
                             if cov_q10 < 5.0 or cov_q10 > 15.0:
                                 print(f"  [WARN] q10 coverage {cov_q10:.1f}% outside [5%, 15%] — calibration may be off")

                     # --- Guardrails: z_τ shape + σ + T + regularizers ---
                     with torch.no_grad():
                         z_current = model_wrapper.quantile_head.shape()
                         z_std_val = z_current.std().item()
                         z_diffs = z_current[1:] - z_current[:-1]
                         z_monotonic = (z_diffs >= 0).all().item()
                         T_val = model_wrapper.quantile_head.temperature_value

                         # Compute current sigma_mean for reporting
                         # Use a sample from calib to get representative sigma
                         _calib_sub = Subset(val_ds, calib_indices[:min(64, len(calib_indices))])
                         _calib_dl_guard = DataLoader(_calib_sub, batch_size=64, shuffle=False,
                                                     collate_fn=calib_collate, num_workers=0)
                         sigma_vals = []
                         for _gb in _calib_dl_guard:
                             _ctx = _gb["context"].to(device)
                             _mask = _gb["context_mask"].to(device)
                             _pc = _gb.get("past_covariates")
                             _fc = _gb.get("future_covariates")
                             if _pc is not None: _pc = _pc.to(device)
                             if _fc is not None: _fc = _fc.to(device)
                             _out = model_wrapper(_ctx, context_mask=_mask,
                                                past_covariates=_pc, future_covariates=_fc)
                             _rep = model_wrapper._pool_context_representation(_ctx, _out)
                             _h = model_wrapper.quantile_head.encoder(_rep)
                             _raw_s = model_wrapper.quantile_head.sigma_head(_h).squeeze(-1)
                             _s = F.softplus(_raw_s) + model_wrapper.quantile_head.sigma_floor
                             sigma_vals.append(_s.cpu())
                         if sigma_vals:
                             _all_sigma = torch.cat(sigma_vals)
                             sigma_mean_val = _all_sigma.mean().item()
                         else:
                             sigma_mean_val = 0.0

                         # Compute current sigma penalty (for reporting only)
                         _low_b = sigma0 * sigma_band_low
                         _high_b = sigma0 * sigma_band_high
                         if sigma_mean_val < _low_b:
                             _sp = (math.log(_low_b) - math.log(sigma_mean_val + 1e-8)) ** 2
                         elif sigma_mean_val > _high_b:
                             _sp = (math.log(sigma_mean_val + 1e-8) - math.log(_high_b)) ** 2
                         else:
                             _sp = 0.0
                         sigma_penalty_val = sigma_reg_weight * _sp

                         # Shape reg value (current)
                         if hasattr(model_wrapper.quantile_head, 'shape'):
                             z_init_norm = model_wrapper.quantile_head.shape.z_init_normalized
                             shape_reg_val = ((z_current - z_init_norm) ** 2).mean().item()
                             elapsed_g = curr_global_step - run_start_step
                             decay_frac_g = max(0.0, 1.0 - elapsed_g / max(shape_reg_decay_steps, 1))
                             shape_reg_effective = shape_reg_weight_init * decay_frac_g * shape_reg_val
                         else:
                             shape_reg_val = 0.0
                             shape_reg_effective = 0.0

                         print(f"\n[Guardrail z_τ] std={z_std_val:.4f} (expect ~1.0) | monotonic={z_monotonic} | T={T_val:.4f}")
                         print(f"[Guardrail σ]  sigma_mean={sigma_mean_val:.6f} | sigma0={sigma0:.6f} | "
                               f"band=[{sigma0*sigma_band_low:.6f}, {sigma0*sigma_band_high:.6f}] | "
                               f"penalty={sigma_penalty_val:.6f}")
                         print(f"[Guardrail shape_reg] value={shape_reg_val:.6f} | decay_frac={decay_frac_g:.3f} | "
                               f"effective={shape_reg_effective:.6f}")
                         if not z_monotonic:
                             print(f"  [ERROR] z_τ monotonicity violated!")
                         if abs(z_std_val - 1.0) > 0.3:
                             print(f"  [WARN] z_τ std deviated from 1.0 by {abs(z_std_val - 1.0):.3f}")

                     print(f"\n[Check A] Current 10d_pinball: {val_10d_pinball:.6f} | Best 10d_pinball: {best_val_loss:.6f}")

                     print(f"Full Val Pinball (Daily): {val_loss:.6f}")
                     print(f"Full Val 10-Day (Native Head):")
                     print(f"  Pinball: {val_10d_pinball:.6f} (Primary)")
                     print(f"  Direction Acc: {val_10d_dir:.2%}")
                     print(f"  Q10 Calibration: {val_10d_q10:.2%} (Target 10%)")
                     run_recorder.emit_metric(
                         run_id,
                         step=curr_global_step,
                         epoch=None,
                         metrics={
                             "val_pinball": val_loss,
                             "val_10d_pinball": val_10d_pinball,
                             "val_10d_direction": val_10d_dir,
                             "val_10d_q10": val_10d_q10,
                             "val_s90": val_s90,
                             "val_s99": val_s99,
                             "sigma_mean": sigma_mean_val,
                             "sigma_penalty": sigma_penalty_val,
                             "shape_reg_value": shape_reg_val,
                             "shape_reg_effective": shape_reg_effective,
                             **({"calib_pinball": calib_metrics.get("avg_pinball"),
                                 "calib_coverage_q10": calib_metrics.get("coverage_q10"),
                                 "calib_coverage_q50": calib_metrics.get("coverage_q50"),
                                 "calib_coverage_q90": calib_metrics.get("coverage_q90"),
                                 "calib_direction": calib_metrics.get("direction_acc"),
                                 "calib_pred_up_rate": calib_metrics.get("pred_up_rate"),
                                 "calib_true_up_rate": calib_metrics.get("true_up_rate"),
                                 "z_tau_std": z_std_val,
                                 "temperature_T": T_val,
                                 } if n_calib > 0 else {}),
                         },
                     )

                     # Primary metric: use calib pinball if available, else val
                     primary_metric = calib_metrics.get("avg_pinball", val_10d_pinball) if n_calib > 0 else val_10d_pinball

                     # Best Checkpoint Logic
                     if primary_metric < (best_val_loss - min_delta):
                         print(f"[Best] Improvement found: {best_val_loss:.6f} -> {primary_metric:.6f}")
                         best_val_loss = primary_metric
                         stall_count = 0

                         save_checkpoint_extended(cfg.out_dir / "best", {
                                 "step": curr_global_step,
                                 "metric_primary": "calib_pinball" if n_calib > 0 else "val_10d_pinball",
                                 "val_10d_pinball": val_10d_pinball,
                                 "val_daily_pinball": val_loss,
                                 "val_10d_direction": val_10d_dir,
                                 "val_10d_q10": val_10d_q10,
                                 "s90": val_s90,
                                 "s99": val_s99,
                                 "calib_metrics": calib_metrics if n_calib > 0 else None,
                                 "z_tau_std": z_std_val,
                                 "temperature_T": T_val,
                                 "config": str(cfg),
                                 "timestamp": str(datetime.datetime.now())
                         })
                     else:
                         stall_count += 1
                         print(f"[Stall] No improvement (Best: {best_val_loss:.6f}). Stall count: {stall_count}/{early_stop_patience}")

                     # Early Stopping Logic (Phase 1 Gate)
                     if stall_count >= early_stop_patience:
                         print(f"[Stop] Early stopping triggered. Stalled for {stall_count} checks.")
                         break

        
            step += 1
                    
        # Final save
        save_checkpoint_extended(cfg.out_dir / "last", {
            "step": step,
            "type": "final",
            "metrics": {"last_loss": avg_loss}
        })
        print("Training Complete.")
        run_recorder.register_artifact(
            run_id,
            name="training_log",
            type="log",
            path=str(log_path),
            tags=["training"],
        )
        run_recorder.finalize_run(run_id, "PASSED")
        return 0
    except KeyboardInterrupt:
        print("Training interrupted.")
        run_recorder.set_status(
            run_id,
            "ABORTED",
            stage="train",
            step="interrupt",
            error={"type": "KeyboardInterrupt", "message": "Training interrupted.", "traceback": ""},
        )
        run_recorder.finalize_run(run_id, "ABORTED")
        return 0
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        run_recorder.set_status(
            run_id,
            "FAILED",
            stage="train",
            step="error",
            error={"type": type(e).__name__, "message": str(e), "traceback": traceback.format_exc()},
        )
        run_recorder.finalize_run(run_id, "FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())
