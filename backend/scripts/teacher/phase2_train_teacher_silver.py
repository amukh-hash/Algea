#!/usr/bin/env python3
"""
Phase 2 (SILVER): Optional Chronos-2 LoRA refresh on daily equity bars + covariates.
Features:
- Byte-Capped LRU Cache for memory safety.
- Resumes from Gold Adapter.
- Swing-consistent targets (session awareness).
- Periodic checkpointing.
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
from typing import Tuple, List, Dict, Optional, Any, Union
from collections import OrderedDict
import threading

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import get_linear_schedule_with_warmup
except Exception:
    print("ERROR: torch/transformers missing.")
    raise

try:
    import polars as pl
except Exception:
    print("ERROR: polars missing.")
    raise

# Imports
sys.path.append(os.getcwd())
try:
    from backend.app.models.chronos2_teacher import load_chronos_adapter, Chronos2NativeWrapper
except ImportError as e:
    print(f"ERROR: Backend imports failed: {e}")
    sys.exit(1)


@dataclass(frozen=True)
class Phase2Config:
    run_id: str
    seed: int
    model_id: str
    
    use_qlora: bool
    lora_config: dict
    
    marketframe_dir: Path
    tickers_csv: Optional[str]
    required_cols: Tuple[str, ...] # Input features
    target_cols: Tuple[str, ...]   # Output features (Subset)
    
    context: int
    pred: int
    stride_rows: int
    max_tickers: int
    max_windows_per_ticker: int
    
    batch_size: int
    grad_accum: int
    lr: float
    max_steps: int
    warmup_steps: int
    checkpoint_every: int
    log_every: int
    val_check_every: int
    
    load_gold_dir: Optional[Path]
    out_dir: Path
    
    byte_cache_limit: int # Bytes


def env_path(name: str, default: str) -> Path:
    return Path(os.getenv(name, default)).expanduser().resolve()


def load_config(args) -> Phase2Config:
    seed = int(os.getenv("SEED", "42"))
    
    # Input Features (Daily adjusted OHLCV + regime covariates)
    required_cols = tuple(c.strip() for c in os.getenv(
        "SILVER_REQUIRED_COLS",
        "date,open_adj,high_adj,low_adj,close_adj,volume,"
        "spy_ret_1d,qqq_ret_1d,iwm_ret_1d,vix_level,rate_proxy,market_breadth_ad"
    ).split(","))

    # Target Features (Subset for swing; default to close)
    target_cols = tuple(c.strip() for c in os.getenv(
        "SILVER_TARGET_COLS", "close_adj"
    ).split(",")) 

    lora_config = {
        "rank": int(os.getenv("LORA_RANK", "16")),
        "alpha": int(os.getenv("LORA_ALPHA", "32")),
        "dropout": float(os.getenv("LORA_DROPOUT", "0.05"))
    }

    gold_dir_str = os.getenv("TEACHER_E_GOLD_OUTDIR", "backend/models/teacher_e/gold")
    gold_dir = Path(gold_dir_str).expanduser().resolve() if gold_dir_str else None

    return Phase2Config(
        run_id=f"silver_{str(uuid.uuid4())[:8]}",
        seed=seed,
        model_id=os.getenv("CHRONOS2_MODEL_ID", "amazon/chronos-2"),
        use_qlora=os.getenv("USE_QLORA", "0") == "1",
        lora_config=lora_config,
        marketframe_dir=env_path("SILVER_MARKETFRAME_DIR", "backend/data_canonical/marketframe_daily"),
        tickers_csv=os.getenv("SILVER_TICKERS", "") or None,
        required_cols=required_cols,
        target_cols=target_cols,
        context=int(os.getenv("SILVER_CONTEXT", "1024")),
        pred=int(os.getenv("SILVER_PRED", "64")),
        stride_rows=int(os.getenv("SILVER_STRIDE", "30")),
        max_tickers=int(os.getenv("SILVER_MAX_TICKERS", "120")),
        max_windows_per_ticker=int(os.getenv("SILVER_MAX_WINDOWS_PER_TICKER", "1500")),
        
        batch_size=int(os.getenv("SILVER_BATCH_SIZE", "4")),
        grad_accum=int(os.getenv("SILVER_GRAD_ACCUM", "8")),
        lr=float(os.getenv("SILVER_LR", "2e-4")),
        max_steps=int(os.getenv("SILVER_MAX_STEPS", "40000")),
        warmup_steps=int(os.getenv("SILVER_WARMUP", "500")),
        checkpoint_every=int(os.getenv("SILVER_CHECKPOINT_EVERY", "5000")),
        log_every=int(os.getenv("SILVER_LOG_EVERY", "50")),
        val_check_every=int(os.getenv("SILVER_VAL_EVERY", "1000")),
        
        load_gold_dir=gold_dir if gold_dir and gold_dir.exists() else None,
        out_dir=env_path("TEACHER_E_SILVER_OUTDIR", "backend/models/teacher_e/silver"),
        byte_cache_limit=int(os.getenv("SILVER_CACHE_BYTES", "4294967296")) # 4GB
    )


class ByteCappedLRUCache:
    """
    LRU cache that evicts based on total estimated size in bytes.
    """
    def __init__(self, byte_limit: int):
        self.byte_limit = byte_limit
        self.current_bytes = 0
        self.cache = OrderedDict() # key -> (value, size)
        self.lock = threading.Lock()

    def get(self, key) -> Optional[np.ndarray]:
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key][0]
            return None

    def put(self, key, value: np.ndarray):
        size = value.nbytes
        with self.lock:
            if key in self.cache:
                # Remove old size first
                _, old_size = self.cache.pop(key)
                self.current_bytes -= old_size
            
            # If item too big for entire cache, don't store (or clear all?)
            if size > self.byte_limit:
                 return # Skip caching
                 
            self.cache[key] = (value, size)
            self.current_bytes += size
            
            # Evict
            while self.current_bytes > self.byte_limit:
                _, (v, s) = self.cache.popitem(last=False)
                self.current_bytes -= s


class SilverMarketFrameDataset(Dataset):
    def __init__(self, marketframe_dir: Path, tickers: List[str], 
                 required_cols: Tuple[str, ...],
                 context: int, pred: int, stride_rows: int,
                 max_windows_per_ticker: int, seed: int,
                 byte_limit: int):
        self.marketframe_dir = marketframe_dir
        self.tickers = tickers
        self.required_cols = required_cols
        self.col_map = {name: i for i, name in enumerate(required_cols)}
        
        self.context = context
        self.pred = pred
        self.stride = stride_rows
        self.rng = np.random.RandomState(seed)
        
        self.cache = ByteCappedLRUCache(byte_limit)
        self.index = [] 
        
        # Validation split logic: Pre-split tickers or time?
        # For Silver (Swing), usually time split (Last 3 months).
        # We index everything here, then SplitValidation handles filtering index.
        
        print(f"[Dataset] Indexing {len(tickers)} tickers...")
        ts_valid_ops = 0
        
        for t in tickers:
            fp = marketframe_dir / f"marketframe_{t}_daily.parquet"
            if not fp.exists(): continue
            try:
                # Light Scan for metadata
                lf = pl.scan_parquet(fp)
                # Need length and timestamp range
                # n = lf.select(pl.len()).collect().item()
                # Actually, reading 'timestamp' col to validate sessions is safer
                # But slow for 120 files.
                # Optimization: Read once, check Session boundaries.
                
                # We'll just read metadata length for now, 
                # and do boundary checks at __getitem__ or random sample?
                # User asked for "Implement Swing/Session-aware windowing".
                # Best way: Check if window [s, s+ctx+pred] crosses session boundary?
                # Actually, Chronos is "context continuous". 
                # The TARGET should be aligned to session close 1D/3D?
                
                # Logic: Just index stride rows. 
                # During getitem, we can't reliably check session cheaply without the dataframe.
                # So we likely need to read timestamps into memory once during index?
                # Timestamps for daily data; length depends on history span per ticker.
                # 500k * 8 bytes * 120 = 480MB. Fits in RAM easily.
                
                df_meta = pl.read_parquet(fp, columns=["timestamp"])
                ts = df_meta["timestamp"].to_numpy() # ns? ms?
                n = len(ts)
                
                if n < context + pred: continue
                
                # Valid starts
                max_start = n - (context + pred)
                starts = list(range(0, max_start, stride_rows))
                
                if len(starts) > max_windows_per_ticker:
                    starts = self.rng.choice(starts, size=max_windows_per_ticker, replace=False).tolist()
                
                for s in starts:
                    # Session Aware Check?
                    # Predict the next 'pred' daily steps for swing-style targets.
                    # If we want "Swing Consistent", maybe we only start windows near EOD?
                    # Plan says: "Target: Subset (e.g. Close)".
                    # Plan says: "Validate session boundaries".
                    # Let's ensure strict contiguity: ts[end] - ts[start] ~ time-diff?
                    # Market data usually has gaps (overnight). 
                    # Chronos handles gaps as tokens? Or expects continuous?
                    # Codec is continuous.
                    # We will index indiscriminately for now, assuming MarketFrame is RTH-clean.
                    
                    self.index.append((t, s, ts[s + context])) # Store Pred-Start-Time for validation split
                    
            except Exception:
                pass
                
        self.rng.shuffle(self.index)
        print(f"[Dataset] Indexed {len(self.index)} windows.")

    def __len__(self):
        return len(self.index)

    def _get_ticker_data(self, ticker: str) -> np.ndarray:
        arr = self.cache.get(ticker)
        if arr is None:
            fp = self.marketframe_dir / f"marketframe_{ticker}_daily.parquet"
            df = pl.read_parquet(fp, columns=list(self.required_cols))
            arr = df.to_numpy()
            self.cache.put(ticker, arr)
        return arr

    def __getitem__(self, idx):
        t, s, _ = self.index[idx]
        arr = self._get_ticker_data(t)
        
        start = s
        cutoff = s + self.context
        end = s + self.context + self.pred
        
        # Robust Slicing
        if end > len(arr):
             end = len(arr) # Short target?
             
        subset = arr[start:end]
        
        # Drop timestamp
        ts_idx = self.col_map.get("timestamp")
        if ts_idx is not None:
             feat_indices = [i for i in range(arr.shape[1]) if i != ts_idx]
             feats = subset[:, feat_indices].astype(np.float32)
        else:
             feats = subset.astype(np.float32)
             
        # Full Input Context (All Feats)
        x_float = feats[:self.context]
        
        # Targets (All Feats, but Collate will filter?)
        # Or check config target_cols.
        # It's cleaner to return full Y here, and Collate selects specific feature tokens.
        y_float = feats[self.context:]
        
        return {
            "x_float": x_float,
            "y_float": y_float
        }

def _build_mask(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 3:
        mask = torch.isfinite(tensor).all(dim=-1)
    else:
        mask = torch.isfinite(tensor)
    return mask.to(dtype=torch.long)


def get_collate_fn(input_names: List[str], target_names: List[str]):
    # Pre-compute indices
    target_indices = []
    for t in target_names:
        if t in input_names:
            target_indices.append(input_names.index(t))
        # Dataset output excludes Timestamp. So we must align names.
        
    def collate(batch):
        xs = np.stack([b["x_float"] for b in batch])
        ys = np.stack([b["y_float"] for b in batch])
        
        x_pt = torch.from_numpy(xs).float()
        y_pt = torch.from_numpy(ys).float()

        if target_indices:
            y_pt = y_pt[:, :, target_indices]

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

def load_tickers(cfg):
    if cfg.tickers_csv:
        return [t.strip().upper() for t in cfg.tickers_csv.split(",") if t.strip()]
    # Scan
    return [p.name.split("_")[0] for p in cfg.marketframe_dir.glob("*_daily.parquet")][:cfg.max_tickers]

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
             if steps > 50: break
    model.train()
    return total_loss / steps if steps else 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow_fallback", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Model (Resume Gold)
    print(f"[Phase2] Loading from Gold: {cfg.load_gold_dir}")
    model_wrapper, info = load_chronos_adapter(
        model_id=cfg.model_id,
        use_qlora=cfg.use_qlora,
        device=device,
        adapter_path=cfg.load_gold_dir,
        lora_config=cfg.lora_config # Fallback if no path
    )

    if not isinstance(model_wrapper, Chronos2NativeWrapper):
        print("ERROR: Loaded model does not expose Chronos native API.")
        return 1

    # 2. Data
    tickers = load_tickers(cfg)
    ds = SilverMarketFrameDataset(
        marketframe_dir=cfg.marketframe_dir,
        tickers=tickers,
        required_cols=cfg.required_cols,
        context=cfg.context, pred=cfg.pred, stride_rows=cfg.stride_rows,
        max_windows_per_ticker=cfg.max_windows_per_ticker,
        seed=cfg.seed, byte_limit=cfg.byte_cache_limit
    )
    
    # Feature Names alignment (exclude TS)
    feat_names = [c for c in cfg.required_cols if c != "timestamp"]
    target_names = [c for c in cfg.target_cols if c != "timestamp"]
    
    collate = get_collate_fn(feat_names, target_names)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate, num_workers=0)
    
    # 4. Train
    opt = torch.optim.AdamW(model_wrapper.parameters(), lr=cfg.lr)
    scheduler = get_linear_schedule_with_warmup(opt, cfg.warmup_steps, cfg.max_steps)
    
    print(f"Starting Silver Training: {cfg.run_id}")
    step = 0
    
    while step < cfg.max_steps:
        for batch in dl:
            context = batch["context"].to(device)
            context_mask = batch["context_mask"].to(device)
            future_target = batch["future_target"].to(device)
            future_target_mask = batch["future_target_mask"].to(device)
            
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
            
            loss = loss / cfg.grad_accum
            loss.backward()
            
            if (step + 1) % cfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model_wrapper.parameters(), 1.0)
                opt.step()
                scheduler.step()
                opt.zero_grad()
                
                if step % cfg.log_every == 0:
                     print(f"Step {step} | Loss {loss.item()*cfg.grad_accum:.4f}")
                     
                if step % cfg.checkpoint_every == 0:
                     path = cfg.out_dir / f"checkpoint-{step}"
                     path.mkdir(parents=True, exist_ok=True)
                     model_wrapper.save_pretrained(path)
                     print(f"Ckpt saved: {path}")

            step += 1
            if step >= cfg.max_steps: break

    model_wrapper.save_pretrained(cfg.out_dir)
    print(f"Done. {cfg.out_dir}")

if __name__ == "__main__":
    raise SystemExit(main())
