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

import numpy as np
from chronos import Chronos2Pipeline

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
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
except ImportError as e:
    print(f"ERROR: Backend imports failed: {e}", file=sys.stderr)
    sys.exit(1)


@dataclass(frozen=True)
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


def env_path(name: str, default: str) -> Path:
    return Path(os.getenv(name, default)).expanduser().resolve()


def load_config(args) -> Phase1Config:
    seed = int(os.getenv("SEED", "42"))
    model_id = os.getenv("CHRONOS2_MODEL_ID", "amazon/chronos-2")
    gold_dir = env_path("GOLD_DAILY_PARQUET_DIR", "backend/data_canonical/daily_parquet")
    gold_glob = os.getenv("GOLD_EXAMPLE_GLOB", "*.parquet")
    required_cols = tuple(c.strip() for c in os.getenv(
        "GOLD_REQUIRED_COLS",
        "date,open_adj,high_adj,low_adj,close_adj,volume,"
        "ret_1d,ret_5d,ret_20d,ewma_vol_20d,parkinson_vol_20d,range_pct_1d,volume_z_20d,"
        "spy_ret_1d,qqq_ret_1d,iwm_ret_1d,vix_level,rate_proxy,market_breadth_ad,rel_ret_1d,beta_spy_20d"
    ).split(","))

    lora_config = {
        "rank": int(os.getenv("LORA_RANK", "16")),
        "alpha": int(os.getenv("LORA_ALPHA", "32")),
        "dropout": float(os.getenv("LORA_DROPOUT", "0.05"))
    }

    target_col = os.getenv("CHRONOS2_TARGET_COL", "close_adj")
    covariate_cols = tuple(c.strip() for c in os.getenv(
        "CHRONOS2_COVARIATE_COLS",
        "ret_1d,ret_5d,ret_20d,ewma_vol_20d,parkinson_vol_20d,range_pct_1d,volume_z_20d,"
        "spy_ret_1d,qqq_ret_1d,iwm_ret_1d,vix_level,rate_proxy,market_breadth_ad,rel_ret_1d,beta_spy_20d"
    ).split(",") if c.strip())
    future_covariate_cols = tuple(c.strip() for c in os.getenv(
        "CHRONOS2_FUTURE_COVARIATE_COLS",
        ""
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
        pred=int(os.getenv("GOLD_PRED", "20")),
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
        max_files_limit=int(os.getenv("GOLD_MAX_FILES", "1000"))
    )

@dataclass
class Phase1Config:
    run_id: str
    seed: int
    model_id: str
    use_qlora: bool
    lora_config: dict
    gold_parquet_dir: Path
    gold_glob: str
    required_cols: list
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
    out_dir: Path
    probe_only: bool
    allow_fallback_codec: bool
    val_split_pct: float
    target_col: str
    covariate_cols: tuple
    future_covariate_cols: tuple
    max_files_limit: int = 1000

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

    def get_quick_val_subset(self, val_ds: 'Subset', size: int = 50, output_dir: Path = None) -> 'Subset':
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
        
        # Direct access to prefetched data (already in RAM)
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
            ref_idx = self.context - 1
            if ref_idx < len(target_series):
                ref_val = target_series[ref_idx]
                if ref_val > 1e-6:
                     # r_t = log(p_t / p_ref)
                     feats[:, target_feat_idx] = np.log(target_series / ref_val)
                else:
                     feats[:, target_feat_idx] = 0.0
        else:
             pass

        return {
            "x_float": feats[:self.context],
            "y_float": feats[self.context:]
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
        past_covariates = x_pt[:, :, self.covariate_indices] if self.covariate_indices else None
        future_covariates = y_pt[:, :, self.future_covariate_indices] if self.future_covariate_indices else None

        context_mask = _build_mask(context_target)
        future_mask = _build_mask(future_target)

        context_target = torch.nan_to_num(context_target, nan=0.0, posinf=0.0, neginf=0.0)
        future_target = torch.nan_to_num(future_target, nan=0.0, posinf=0.0, neginf=0.0)
        if past_covariates is not None:
            past_covariates = torch.nan_to_num(past_covariates, nan=0.0, posinf=0.0, neginf=0.0)
        if future_covariates is not None:
            future_covariates = torch.nan_to_num(future_covariates, nan=0.0, posinf=0.0, neginf=0.0)
        
        return {
            "context": context_target,
            "context_mask": context_mask,
            "future_target": future_target,
            "future_target_mask": future_mask,
            "past_covariates": past_covariates,
            "future_covariates": future_covariates,
        }

def get_collate_fn(feature_names: List[str], target_col: str,
                   covariate_cols: List[str], future_covariate_cols: List[str]):
    return CollateFn(feature_names, target_col, covariate_cols, future_covariate_cols)

def compute_pinball_loss(y_true, quantile_preds, quantiles, quantile_weights=None):
    """
    y_true: [B, H]
    quantile_preds: [B, Q, H'] where H' may be >= H (Chronos 2 outputs 32 steps)
    quantiles: List[float]
    quantile_weights: Optional[Tensor] [Q] - per-quantile weights (for curriculum)
    Returns: scalar mean loss
    """
    # Ensure correct shapes
    if y_true.ndim == 2:
        y_true = y_true.unsqueeze(1) # [B, 1, H]
    
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
    
    with torch.no_grad():
        for batch in dl:
            context = batch["context"].to(device)
            context_mask = batch["context_mask"].to(device)
            future_target = batch["future_target"].to(device)
            future_target_mask = batch["future_target_mask"].to(device)
            past_covariates = batch["past_covariates"]
            future_covariates = batch["future_covariates"]
            if past_covariates is not None: past_covariates = past_covariates.to(device)
            if future_covariates is not None: future_covariates = future_covariates.to(device)
            
            pred_len = future_target.shape[1]
            
            # 1. Generate quantile predictions using Chronos2Pipeline
            if pipeline is not None:
                # Pipeline.predict expects [B, 1, context_len] and returns list of [1, num_quantiles, pred_len]
                batch_size = context.shape[0]
                context_np = context.cpu().unsqueeze(1).numpy()  # [B, 1, context_len]
                
                # Predict returns list of [1, Q, H] tensors
                predictions = pipeline.predict(
                    inputs=context_np,
                    prediction_length=pred_len
                )  # List of B elements, each [1, Q, H]
                
                # Stack predictions into [B, Q, H]
                qp = torch.stack([pred.squeeze(0) for pred in predictions], dim=0).to(device)
                
                # DEBUG: Check values
                if steps == 0:
                     print(f"[Val Debug] Target Mean: {future_target.mean().item():.6f} | Pred Mean: {qp.mean().item():.6f}")
                     print(f"[Val Debug] Target Range: [{future_target.min().item():.6f}, {future_target.max().item():.6f}]")
                     print(f"[Val Debug] Pred Range: [{qp.min().item():.6f}, {qp.max().item():.6f}]")

                # Compute pinball loss
                p_loss = compute_pinball_loss(
                    future_target.squeeze(-1) if future_target.ndim == 3 else future_target,
                    qp,
                    quantiles,
                    quantile_weights=quantile_weights
                )
                total_pinball_loss += p_loss.item()
                
                # 2. Monitor Spreads (Day 10 / Index 9)
                if qp.shape[2] >= 10:
                    day_10_preds = qp[:, :, 9] 
                    s90 = (day_10_preds[:, 18] - day_10_preds[:, 2]).mean().item()
                    s99 = (day_10_preds[:, 20] - day_10_preds[:, 0]).mean().item()
                    total_s90 += s90
                    total_s99 += s99
            
            # 3. Also compute internal loss for comparison (forward pass)
            out = model(
                context, 
                context_mask=context_mask, 
                future_target=future_target,
                future_target_mask=future_target_mask,
                past_covariates=past_covariates,
                future_covariates=future_covariates,
                num_output_patches=None
            )
            if hasattr(out, "loss") and out.loss is not None:
                total_internal_loss += out.loss.item()
            
            steps += 1
            
    model.train()
    avg_pinball_loss = total_pinball_loss / steps if steps > 0 else 0.0
    avg_internal_loss = total_internal_loss / steps if steps > 0 else 0.0
    avg_s90 = total_s90 / steps if steps > 0 else 0.0
    avg_s99 = total_s99 / steps if steps > 0 else 0.0
    
    return {
        "pinball_loss": avg_pinball_loss, 
        "internal_loss": avg_internal_loss,
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
        print(f"[Debug] Output Directory: {cfg.out_dir}")
        cfg.out_dir.mkdir(parents=True, exist_ok=True)
        
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        
        # 1. Model
        # Setup Logging
        log_path = cfg.out_dir / "training.log"
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
        
        train_ds, val_ds = dataset.split_validation(0.15)
        
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
            use_qlora=False, # cfg.use_qlora
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
        trainable_params = [p for p in model_wrapper.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainable_params, lr=cfg.lr)
        scheduler = get_linear_schedule_with_warmup(
            opt, num_warmup_steps=cfg.warmup_steps, num_training_steps=cfg.max_steps
        )
        
        print(f"Starting Training: {cfg.run_id}")
        
        # Prepare Validation Subsets
        val_quick_ds = dataset.get_quick_val_subset(val_ds, size=50, output_dir=cfg.out_dir)
        val_quick_dl = DataLoader(val_quick_ds, batch_size=cfg.batch_size, shuffle=False, 
                                  collate_fn=get_collate_fn(feature_names, cfg.target_col, cfg.covariate_cols, cfg.future_covariate_cols), 
                                  num_workers=0)
        
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
            past_covariates = batch["past_covariates"]
            future_covariates = batch["future_covariates"]
            if past_covariates is not None: past_covariates = past_covariates.to(device)
            if future_covariates is not None: future_covariates = future_covariates.to(device)
            
            # Forward
            if step == 0:
                 print(f"[Debug] Input device: {context.device}")
            
            out = model_wrapper(
                context,
                context_mask=context_mask,
                future_target=future_target,
                future_target_mask=future_target_mask,
                past_covariates=past_covariates,
                future_covariates=future_covariates,
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
            
            # Check if model output has quantile_preds
            if hasattr(out, 'quantile_preds') and out.quantile_preds is not None:
                # Manual Pinball Loss with weights
                loss = compute_pinball_loss(
                    future_target.squeeze(-1) if future_target.ndim == 3 else future_target,
                    out.quantile_preds,
                    quantiles,
                    quantile_weights=quantile_weights
                )
            else:
                # Fallback to model's internal loss (Phase 1: Canonical)
                loss = out.loss if hasattr(out, "loss") else None
                if loss is None:
                    print("[Warn] Model returned no loss, skipping batch.")
                    loss = torch.tensor(0.0, device=device, requires_grad=True)
            
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
                
                if curr_global_step % cfg.log_every == 0:
                    avg_loss = running_loss / cfg.grad_accum / cfg.log_every
                    # Specifically for first steps where log_every=1
                    if cfg.log_every == 1: avg_loss = running_loss / cfg.grad_accum
                    
                    current_lr = scheduler.get_last_lr()[0]
                    
                    # Log extras removed to reduce noise (training batch spreads are volatile)
                    print(f"Step {curr_global_step} | Loss: {avg_loss:.4f} | LR: {current_lr:.2e}")
                    running_loss = 0.0
                    
                # Quick Val (Every 500 steps)
                if curr_global_step % 500 == 0: 
                    print(f"--- QuickVal (Step {curr_global_step}) ---")
                    qv_metrics = run_validation(model_wrapper, val_quick_dl, device, quantiles, desc="QuickVal", pipeline=pipeline)
                    qv_loss = qv_metrics['pinball_loss']
                    qv_internal = qv_metrics['internal_loss']
                    qv_s90 = qv_metrics['s90']
                    qv_s99 = qv_metrics['s99']
                    
                    print(f"QuickVal Pinball: {qv_loss:.4f} | Internal: {qv_internal:.4f} | S90: {qv_s90:.4f} | S99: {qv_s99:.4f}")
                    
                    # Update EMA
                    if quick_val_ema is None:
                        quick_val_ema = qv_loss
                    else:
                        quick_val_ema = ema_alpha * qv_loss + (1 - ema_alpha) * quick_val_ema
                    print(f"QuickVal EMA: {quick_val_ema:.4f}")

                    # Save LAST checkpoint every 500 steps (for recovery)
                    save_path = cfg.out_dir / f"checkpoint-{curr_global_step}" # Optional: rolling logic to save space?
                    # Actually, let's just save "last" always to save space, and numbered only if needed.
                    # Plan says: "Save last every N steps (crash recovery)"
                    # We will overwrite 'last' frequently, and keep numbered checkpoints sparingly?
                    # The user request implies keeping history: "out_dir/checkpoint-{step}/"
                    save_path.mkdir(exist_ok=True)
                    model_wrapper.model.save_pretrained(save_path)
                    
                    # Also update 'last' pointer
                    last_path = cfg.out_dir / "last"
                    last_path.mkdir(exist_ok=True)
                    model_wrapper.model.save_pretrained(last_path)

                # Full Validation (Every 1000 steps initially as per Plan Phase 5)
                # Plan says: "Full-val: every 2500 steps OR every 1000" -> User said "keep full-val every 1000"
                if curr_global_step > 0 and curr_global_step % 1000 == 0:
                     print(f"--- FullVal (Step {curr_global_step}) ---")
                     fv_metrics = run_validation(model_wrapper, val_dl, device, quantiles, pipeline=pipeline)
                     
                     # 3. Canonical Selection: Pinball Loss
                     val_loss = fv_metrics['pinball_loss']
                     val_internal = fv_metrics['internal_loss']
                     val_s90 = fv_metrics['s90']
                     val_s99 = fv_metrics['s99']
                     
                     print(f"Full Val Pinball: {val_loss:.4f} | Internal: {val_internal:.4f} | S90: {val_s90:.4f} | S99: {val_s99:.4f}")
                     
                     # Best Checkpoint Logic
                     if val_loss < (best_val_loss - min_delta):
                         print(f"[Best] Improvement found: {best_val_loss:.4f} -> {val_loss:.4f}")
                         best_val_loss = val_loss
                         stall_count = 0
                         
                         best_path = cfg.out_dir / "best"
                         best_path.mkdir(exist_ok=True)
                         model_wrapper.model.save_pretrained(best_path)
                         
                         # Save manifest
                         with open(best_path / "best_manifest.json", "w") as f:
                             json.dump({
                                 "step": curr_global_step,
                                 "val_loss": val_loss,
                                 "val_internal_loss": val_internal,
                                 "s90": val_s90,
                                 "s99": val_s99,
                                 "config": str(cfg),
                                 "timestamp": str(datetime.datetime.now())
                             }, f, indent=2)
                     else:
                         stall_count += 1
                         print(f"[Stall] No improvement (Best: {best_val_loss:.4f}). Stall count: {stall_count}/{early_stop_patience}")
                         
                     # Early Stopping Logic (Phase 1 Gate)
                     if stall_count >= early_stop_patience:
                         print(f"[Stop] Early stopping triggered. Stalled for {stall_count} checks.")
                         break

        
            step += 1
                    
        # Final save
        model_wrapper.model.save_pretrained(cfg.out_dir / "last")
        print("Training Complete.")
        return 0
    except KeyboardInterrupt:
        print("Training interrupted.")
        return 0
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
        # Manifest
        manifest = {
            "run_id": cfg.run_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "final_step": step,
            "config": asdict(cfg, dict_factory=lambda x: {k: str(v) if isinstance(v, Path) else v for k, v in x.items()})
        }
        with open(cfg.out_dir / "run_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"Done. Saved to {cfg.out_dir}")
        return 0
    except Exception:
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
