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
        "spy_ret_1d,qqq_ret_1d,iwm_ret_1d,vix_level,rate_proxy,market_breadth_ad"
    ).split(","))

    lora_config = {
        "rank": int(os.getenv("LORA_RANK", "16")),
        "alpha": int(os.getenv("LORA_ALPHA", "32")),
        "dropout": float(os.getenv("LORA_DROPOUT", "0.05"))
    }

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
        stride_rows=int(os.getenv("GOLD_STRIDE", "60")),
        max_files=int(os.getenv("GOLD_MAX_FILES", "50")),
        max_windows=int(os.getenv("GOLD_MAX_WINDOWS_PER_FILE", "500")),
        cache_size=int(os.getenv("GOLD_FILE_CACHE", "4")),
        
        batch_size=int(os.getenv("GOLD_BATCH_SIZE", "4")),
        grad_accum=int(os.getenv("GOLD_GRAD_ACCUM", "8")),
        lr=float(os.getenv("GOLD_LR", "1e-4")),
        max_steps=int(os.getenv("GOLD_MAX_STEPS", "3000")),
        warmup_steps=int(os.getenv("GOLD_WARMUP", "100")),
        checkpoint_every=int(os.getenv("GOLD_CHECKPOINT_EVERY", "250")),
        log_every=int(os.getenv("GOLD_LOG_EVERY", "25")),
        val_split_pct=float(os.getenv("GOLD_VAL_SPLIT", "0.10")),
        
        out_dir=env_path("TEACHER_E_GOLD_OUTDIR", "backend/models/teacher_e/gold"),
        probe_only=args.probe_only,
        allow_fallback_codec=args.allow_fallback
    )

class ThreadSafeLRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()
        self.lock = threading.Lock()

    def get(self, key):
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
            return None

    def put(self, key, value):
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            self.cache[key] = value
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)

class GoldFuturesWindowDataset(Dataset):
    def __init__(self, files: List[Path], required_cols: Tuple[str, ...],
                 context: int, pred: int, stride_rows: int, 
                 max_windows: int, seed: int, cache_size: int):
        self.files = files
        self.required_cols = required_cols
        self.col_map = {name: i for i, name in enumerate(required_cols)}
        
        self.context = context
        self.pred = pred
        self.stride_rows = stride_rows
        self.max_windows = max_windows
        self.rng = np.random.RandomState(seed)
        
        self.cache = ThreadSafeLRUCache(cache_size)
        self.index = [] # List[(file_idx, start_row, timestamp_val)]
        
        print(f"[Dataset] Indexing {len(files)} files...")
        
        for fi, fp in enumerate(self.files):
            try:
                # Read timestamps only for indexing?
                # Or just read shape if we assume continuity checked in Preflight?
                # For robust training check:
                # Use scan to get height without loading data
                n = pl.scan_parquet(fp).select(pl.len()).collect().item()
                
                max_start = n - (context + pred)
                if max_start <= 0: continue
                
                starts = list(range(0, max_start, stride_rows))
                
                # Filter caps
                if len(starts) > max_windows:
                     starts = self.rng.choice(starts, size=max_windows, replace=False).tolist()
                
                for s in starts:
                    # Use row index as t_val for stable sorting if timestamp missing
                    self.index.append((fi, int(s), int(s)))
                    
            except Exception as e:
                print(f"ERR indexing {fp}: {e}")
        
        # Sort by time for temporal split
        # Only works if timestamp types comparable.
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
        train_idx = list(range(train_size))
        val_idx = list(range(train_size, total))
        
        return (
            torch.utils.data.Subset(self, train_idx),
            torch.utils.data.Subset(self, val_idx)
        )

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        # Handle Subset wrappng index mapping?
        # Subset deals with it. 'idx' here is into self.index
        
        fi, s, _ = self.index[idx]
        fp = self.files[fi]
        
        # Cache access
        arr = self.cache.get(fi)
        if arr is None:
            # Flexible read: Only read columns that exist
            schema = pl.scan_parquet(fp).schema
            actual_cols = [c for c in self.required_cols if c in schema]
            
            # Limit memory spike by using scan
            df = pl.scan_parquet(fp).select(actual_cols).collect()
            arr = df.to_numpy().astype(np.float32)
            del df
            gc.collect() 
            self.cache.put(fi, arr)
            
        # Slicing
        ts_idx = self.col_map.get("timestamp")
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
        else:
             feats = subset.astype(np.float32)

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


def get_collate_fn():
    def collate(batch):
        xs = np.stack([b["x_float"] for b in batch]) # [B, T, F]
        ys = np.stack([b["y_float"] for b in batch]) # [B, Pred, F]
        
        x_pt = torch.from_numpy(xs).float()
        y_pt = torch.from_numpy(ys).float()

        context_mask = _build_mask(x_pt)
        future_mask = _build_mask(y_pt)

        x_pt = torch.nan_to_num(x_pt, nan=0.0, posinf=0.0, neginf=0.0)
        y_pt = torch.nan_to_num(y_pt, nan=0.0, posinf=0.0, neginf=0.0)
        
        return {
            "context": x_pt,
            "context_mask": context_mask,
            "future_target": y_pt,
            "future_target_mask": future_mask
        }
    return collate

def validation_loop(model, dl, device):
    model.eval()
    total_loss = 0
    steps = 0
    
    with torch.no_grad():
        for batch in dl:
            context = batch["context"].to(device)
            context_mask = batch["context_mask"].to(device)
            future_target = batch["future_target"].to(device)
            future_target_mask = batch["future_target_mask"].to(device)
            
            out = model(
                context,
                context_mask=context_mask,
                future_target=future_target,
                future_target_mask=future_target_mask
            )
            
            if hasattr(out, "loss"):
                loss = out.loss
            else:
                 pred = getattr(out, "prediction", None) or getattr(out, "predictions", None)
                 if pred is None:
                     raise RuntimeError("Chronos native forward did not return loss or predictions.")
                 pred = torch.as_tensor(pred)
                 if pred.ndim == 2:
                     pred = pred.unsqueeze(1).unsqueeze(-1)
                 elif pred.ndim == 3:
                     pred = pred.unsqueeze(-1)
                 loss = torch.mean((pred - future_target) ** 2)
            
            total_loss += loss.item()
            steps += 1
            if steps > 50: break # Partial eval for speed
            
    model.train()
    return total_loss / steps if steps > 0 else 0.0

def main():
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--probe_only", action="store_true")
        parser.add_argument("--allow_fallback", action="store_true")
        args = parser.parse_args()

        cfg = load_config(args)
        print(f"[Debug] Output Directory: {cfg.out_dir}")
        cfg.out_dir.mkdir(parents=True, exist_ok=True)
        
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        
        # 1. Model
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

        # 3. Data
        files = sorted(cfg.gold_parquet_dir.glob(cfg.gold_glob))[: cfg.max_files]
        dataset = GoldFuturesWindowDataset(
            files, cfg.required_cols,
            cfg.context, cfg.pred, cfg.stride_rows,
            cfg.max_windows, cfg.seed, cfg.cache_size
        )
        
        train_ds, val_ds = dataset.split_validation(cfg.val_split_pct)
        
        collate = get_collate_fn()
        train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, 
                              collate_fn=collate, num_workers=0)
        val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            collate_fn=collate, num_workers=0)

        # 4. Training Setup
        # Only pass trainable parameters to the optimizer
        trainable_params = [p for p in model_wrapper.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainable_params, lr=cfg.lr)
        scheduler = get_linear_schedule_with_warmup(
            opt, num_warmup_steps=cfg.warmup_steps, num_training_steps=cfg.max_steps
        )
        
        print(f"Starting Training: {cfg.run_id}")
        step = 0
        running_loss = 0.0
        
        while step < cfg.max_steps:
            for batch in train_dl:
                context = batch["context"].to(device)
                context_mask = batch["context_mask"].to(device)
                future_target = batch["future_target"].to(device)
                future_target_mask = batch["future_target_mask"].to(device)
                
                # Forward
                if step == 0:
                     print(f"[Debug] Input device: {context.device}")
                     # Probe wrapper's underlying model device
                     print(f"[Debug] Model device: {next(model_wrapper.model.parameters()).device}")
                out = model_wrapper(
                    context,
                    context_mask=context_mask,
                    future_target=future_target,
                    future_target_mask=future_target_mask
                )
                
                loss = out.loss if hasattr(out, "loss") else None
                if loss is None:
                     pred = getattr(out, "prediction", None) or getattr(out, "predictions", None)
                     if pred is None:
                         raise RuntimeError("Chronos native forward did not return loss or predictions.")
                     pred = torch.as_tensor(pred)
                     if pred.ndim == 2:
                         pred = pred.unsqueeze(1).unsqueeze(-1)
                     elif pred.ndim == 3:
                         pred = pred.unsqueeze(-1)
                     loss = torch.mean((pred - future_target) ** 2)
                
                # Accumulate
                loss_scaled = loss / cfg.grad_accum
                loss_scaled.backward()
                
                running_loss += loss.item()
                
                if (step + 1) % cfg.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model_wrapper.parameters(), 1.0)
                    opt.step()
                    scheduler.step()
                    opt.zero_grad()
                    
                    # Report
                    avg_loss = running_loss / cfg.grad_accum
                    running_loss = 0.0
                    
                    if step % cfg.log_every == 0:
                         # Weight check
                         w_sum = sum(p.data.norm().item() for p in model_wrapper.model.parameters() if p.requires_grad)
                         
                         # RAM Check
                         mem = psutil.Process().memory_info().rss / 1e9
                         print(f"Step {step} | Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e} | W_Norm: {w_sum:.2f} | RAM: {mem:.2f}GB")
                         
                         if mem > 28.0:
                              print("WARN: Critical RAM usage detected. Clearing caches...")
                              # Trigger dataset cache clear if it was a custom method, but we can't easily reach it via wrapper
                              gc.collect()
                              if torch.cuda.is_available():
                                   torch.cuda.empty_cache()

                    # Validation & Checkpoint
                    if step > 0 and step % cfg.checkpoint_every == 0:
                         val_loss = validation_loop(model_wrapper, val_dl, device)
                         print(f"Step {step} | Val Loss: {val_loss:.4f}")
                         
                         ckpt_dir = cfg.out_dir / f"checkpoint-{step}"
                         ckpt_dir.mkdir(parents=True, exist_ok=True)
                         model_wrapper.save_pretrained(ckpt_dir)
                         
                step += 1
                if step >= cfg.max_steps: break
                
        # Final Save
        model_wrapper.save_pretrained(cfg.out_dir)
        
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
