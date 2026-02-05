#!/usr/bin/env python3
# Chronos-2 inspection utility for daily equity bars + covariates (protocol-aligned).
import os
import sys
import inspect
import json
from typing import List

try:
    import torch
    from transformers import AutoConfig, AutoModel
except ImportError:
    print("Transformers/Torch not installed.")
    sys.exit(1)

def inspect_model(model_candidates: List[str]):
    all_results = {}
    for model_id in model_candidates:
        print(f"--- Trying {model_id} ---")
        try:
            config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
            model = AutoModel.from_pretrained(model_id, trust_remote_code=True, device_map=None)
            
            res = {
                "model_id": model_id,
                "model_type": config.model_type,
                "architectures": getattr(config, 'architectures', []),
                "model_class": str(type(model)),
                "config": {k: str(getattr(config, k)) for k in ['n_positions', 'n_ctx', 'max_position_embeddings', 'vocab_size'] if hasattr(config, k)}
            }
            
            if hasattr(model, "forward"):
                sig = inspect.signature(model.forward)
                res["forward_signature"] = str(sig)
                params = list(sig.parameters.keys())
                res["params"] = params
                res["accepts_input_ids"] = 'input_ids' in params
                res["accepts_attention_mask"] = 'attention_mask' in params
                res["accepts_cache_position"] = 'cache_position' in params
                res["accepts_use_cache"] = 'use_cache' in params
            
            all_results[model_id] = res
            return all_results
            
        except Exception as e:
            print(f"Failed {model_id}: {e}")
            all_results[model_id] = {"error": str(e)}
            continue
    return all_results

def main():
    candidates = ["amazon/chronos-2"]
    results = inspect_model(candidates)
    
    out_path = "backend/reports/inspect_v2_io.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nReport saved to {out_path}")

if __name__ == "__main__":
    main()
