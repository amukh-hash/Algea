#!/usr/bin/env python3
"""
Validate Trained Chronos-2 Adapter (Gold Phase)
"""
import os
import sys
import argparse
import torch
import numpy as np
import json
from pathlib import Path
from torch.utils.data import DataLoader

# Add backend to path
sys.path.append(os.getcwd())

from backend.app.models.chronos2_codec import Chronos2Codec, CodecConfig
from backend.app.models.chronos2_teacher import load_chronos_adapter
from backend.scripts.teacher.phase1_train_teacher_gold import (
    load_config, GoldFuturesWindowDataset, get_collate_fn
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to adapter checkpoint directory")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"], help="Split to validate on")
    parser.add_argument("--limit", type=int, default=100, help="Max batches to evaluate")
    args = parser.parse_args()

    # Load Config (Environment based, but we override adapter)
    # We rely on env vars being set or defaults.
    # We can pass dummy args to load_config
    class DummyArgs:
        probe_only = False
        allow_fallback = False
    
    cfg = load_config(DummyArgs())
    
    print(f"Validating Adapter: {args.adapter_path}")
    print(f"Dataset Glob: {cfg.gold_glob}")
    
    # 1. Codec
    if not cfg.codec_path.exists():
        print(f"ERROR: Codec {cfg.codec_path} missing.")
        return 1
    
    codec = Chronos2Codec(CodecConfig())
    codec.load(cfg.codec_path)
    
    # 2. Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_wrapper, info = load_chronos_adapter(
        model_id=cfg.model_id,
        use_qlora=cfg.use_qlora,
        device=device,
        adapter_path=Path(args.adapter_path),
        lora_config=cfg.lora_config,
        eval_mode=True
    )
    
    # 3. Data
    files = sorted(cfg.gold_parquet_dir.glob(cfg.gold_glob))[: cfg.max_files]
    dataset = GoldFuturesWindowDataset(
        files, cfg.required_cols,
        cfg.context, cfg.pred, cfg.stride_rows,
        cfg.max_windows, cfg.seed, cfg.cache_size
    )
    
    train_ds, val_ds = dataset.split_validation(cfg.val_split_pct)
    target_ds = val_ds if args.split == "val" else train_ds
    
    print(f"Validation Dataset Size: {len(target_ds)}")
    
    collate = get_collate_fn(codec)
    dl = DataLoader(target_ds, batch_size=cfg.batch_size, shuffle=False, 
                    collate_fn=collate, num_workers=0)
    
    # 4. Params (for loss)
    # T5 loss is cross entropy on vocab
    
    model_wrapper.eval()
    total_loss = 0
    steps = 0
    
    print("Starting Evaluation...")
    with torch.no_grad():
        for batch in dl:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            out = model_wrapper(input_ids, mask, labels=labels)
            
            if hasattr(out, "loss"):
                loss = out.loss
            else:
                logits = out.logits
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), 
                    labels.reshape(-1)
                )
            
            total_loss += loss.item()
            steps += 1
            
            if steps % 10 == 0:
                print(f"Step {steps}: Loss={loss.item():.4f}")
            
            if steps >= args.limit:
                break
                
    avg_loss = total_loss / steps if steps > 0 else 0.0
    print(f"Validation Complete. Steps: {steps}")
    print(f"Average Loss: {avg_loss:.4f}")
    
    # Save result?
    res_path = Path(args.adapter_path) / f"validation_{args.split}.json"
    with open(res_path, "w") as f:
        json.dump({"loss": avg_loss, "steps": steps, "split": args.split}, f)
    print(f"Result saved to {res_path}")

if __name__ == "__main__":
    main()
