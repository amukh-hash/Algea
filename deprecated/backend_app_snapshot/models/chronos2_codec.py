"""
Chronos 2 Codec Module (Phase 0.5)

Handles tokenization/encoding of multivariate time series.
Strategy:
1. Introspection-driven: If model needs 'input_ids', Codec is mandatory.
2. True Multivariate: Encodes `[B, T, F]` -> `[B, T, F]` IDs. 
   Does NOT flatten/stack by default (wrapper handles that).
3. Binning: Supports Quantile Binning (fit) or Linear Fallback.

Usage:
    codec = Chronos2Codec(config)
    codec.load_artifact("codec_v1.json") 
    enc = codec.encode(x_float) # -> input_ids, scale, meta
"""

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any

import numpy as np
import torch

@dataclass
class CodecConfig:
    vocab_size: int = 4096 
    special_tokens: int = 2 # e.g. PAD=0, UNK=1
    allow_fallback: bool = False
    
    # Artifact fields (populated on load/fit)
    feature_names: List[str] = field(default_factory=list)
    quantiles: Dict[str, List[float]] = field(default_factory=dict)
    strategy: str = "quantile" # or "linear"

class Chronos2Codec:
    def __init__(self, config: CodecConfig):
        self.config = config
        self.native_tokenizer = None
        self.is_native = False

        # Try loading native tokenizer
        try:
            from chronos import ChronosTokenizer
            # Usually requires model_id from somewhere? 
            # We assume if user wants native, they configured environment or passed it?
            # For now, we support "if loaded externally" or rely on fallback logic.
            # Ideally: ChronosTokenizer.from_pretrained(os.getenv("CHRONOS2_MODEL_ID"))
            pass
        except ImportError:
            pass
        
        # Validation
        if not self.config.allow_fallback and not self.is_native and not self.config.quantiles:
            # We delay strict failure until encode is called OR explicit 'validate()'
            pass

    def fit_quantiles(self, data: Union[np.ndarray, torch.Tensor], feature_names: List[str], n_bins: int = None):
        """
        Fits quantile bin edges per feature.
        data: [Samples, Features] (Time dim flattened or handled)
        """
        if n_bins is None:
            n_bins = self.config.vocab_size - self.config.special_tokens
        
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
            
        if data.ndim != 3:
             raise ValueError("Fit data must be [Windows, Time, Features] (3D) to compute local scaling.")

        B, T, F = data.shape
        
        if len(feature_names) != F:
            raise ValueError(f"Feature names {len(feature_names)} != Data Cols {F}")
            
        self.config.feature_names = feature_names
        self.config.quantiles = {}
        
        print(f"[Codec] Fitting {n_bins} quantiles for {len(feature_names)} features...")
        
        # Compute local scale
        # [B, T, F]
        # scale = mean(abs(x), dim=1) -> [B, 1, F]
        abs_data = np.abs(data)
        scale = np.nanmean(abs_data, axis=1, keepdims=True)
        scale[scale == 0] = 1.0
        
        scaled_data = data / scale # [B, T, F]
        
        # Now flatten to [Points, F]
        flat = scaled_data.reshape(-1, scaled_data.shape[-1])
        
        # Fit Quantiles
        qs = np.linspace(0, 1, n_bins + 1)
        
        for i, name in enumerate(feature_names):
            vals = flat[:, i]
            # Drop NaNs
            vals = vals[~np.isnan(vals)]
            edges = np.quantile(vals, qs)
            # Make strictly increasing?
            # edges = np.unique(edges) # Might reduce bin count
            # Keep edges as is, but bucketize handles duplicates?
            # Better to nudge?
            edges = np.sort(edges)
            self.config.quantiles[name] = edges.tolist()

        self.config.strategy = "quantile"

    def save(self, path: Path):
        data = {
            "vocab_size": self.config.vocab_size,
            "special_tokens": self.config.special_tokens,
            "feature_names": self.config.feature_names,
            "quantiles": self.config.quantiles,
            "strategy": self.config.strategy
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: Path):
        with open(path, "r") as f:
            data = json.load(f)
        self.config.vocab_size = data.get("vocab_size", 4096)
        self.config.feature_names = data.get("feature_names", [])
        self.config.quantiles = data.get("quantiles", {})
        self.config.strategy = data.get("strategy", "quantile")
        if not self.config.quantiles and not self.config.allow_fallback:
             print("WARN: Loaded codec has no quantiles.")

    def encode(self, x: torch.Tensor, feature_indices: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        x: [B, T, F] float32
        Returns: input_ids [B, T, F], attention_mask [B, T], scale [B, 1, F]
        NO STACKING/FLATTENING here. Wrapper handles it.
        """
        B, T, F = x.shape
        device = x.device
        
        if self.native_tokenizer:
             # Native path (stub)
             pass
             
        if not self.config.quantiles and not self.config.allow_fallback:
            raise RuntimeError("Codec not fit and no native tokenizer.")

        # 1. Scale
        scale = torch.mean(torch.abs(x), dim=1, keepdim=True) # [B, 1, F]
        scale[scale == 0] = 1.0 # Avoid div/0
        
        x_scaled = x / scale
        
        # 2. Binning
        input_ids = torch.zeros((B, T, F), dtype=torch.long, device=device)
        
        # If feature_indices specific?
        feat_names = self.config.feature_names  # Full list
        
        # Assuming x matches feature_names order by default unless indices passed
        # Or x is subset?
        # We assume x is [B, T, F] corresponding to indices if passed, or all if not.
        
        for f_idx in range(F):
            # Map column index to name
            # If feature_indices provided, map f_idx (in x) -> global_idx
            global_idx = feature_indices[f_idx] if feature_indices else f_idx
            
            if global_idx >= len(feat_names):
                 # Out of bounds? 
                 pass
                 
            fname = feat_names[global_idx]
            edges = self.config.quantiles.get(fname)
            
            if edges is None:
                # Fallback Linear Logic?
                if self.config.allow_fallback:
                     # Simple -10 to 10 clamp
                     limit = 10.0
                     val = torch.clamp(x_scaled[:, :, f_idx], -limit, limit)
                     # Normalize
                     n_bins = self.config.vocab_size - self.config.special_tokens
                     norm = (val + limit) / (2*limit)
                     ids = (norm * (n_bins-1)).long()
                else:
                    raise KeyError(f"No bins for feature {fname}")
            else:
                # Quantile bucketize
                # edges is list. Convert to tensor.
                # Bucketize: 
                # boundaries are strictly increasing.
                # bucketize returns index of bin.
                # We need to map to [0, n_bins-1] -> + special_tokens
                bins = torch.tensor(edges, device=device)
                ids = torch.bucketize(x_scaled[:, :, f_idx], bins)
                # Clip to valid range [0, n_bins-1]
                ids = torch.clamp(ids, min=0, max=len(edges)-2)
                
            # Shift for special tokens
            ids = ids + self.config.special_tokens
            input_ids[:, :, f_idx] = ids
            
        # Attention Mask
        # [B, T]
        attention_mask = torch.ones((B, T), dtype=torch.long, device=device)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "scale": scale,
            "meta": {
                "strategy": "multivariate",
                "quantiles": True if self.config.quantiles else False
            }
        }

    def encode_targets(self, y: torch.Tensor, scale: torch.Tensor, feature_indices: Optional[List[int]] = None):
        """
        Encode target values using the SAME scale as context.
        y: [B, Pred, F]
        scale: [B, 1, F] (from context)
        """
        # Similar logic but use passed scale
        y_scaled = y / scale
        
        input_ids = torch.zeros_like(y, dtype=torch.long)
        B, P, F = y.shape
        device = y.device
        feat_names = self.config.feature_names
        
        for f_idx in range(F):
            global_idx = feature_indices[f_idx] if feature_indices else f_idx
            fname = feat_names[global_idx]
            edges = self.config.quantiles.get(fname)
            
            if edges:
                bins = torch.tensor(edges, device=device)
                ids = torch.bucketize(y_scaled[:, :, f_idx], bins)
                ids = torch.clamp(ids, min=0, max=len(edges)-2)
            else:
                # Fallback linear
                limit = 10.0
                val = torch.clamp(y_scaled[:, :, f_idx], -limit, limit)
                n_bins = self.config.vocab_size - self.config.special_tokens
                norm = (val + limit) / (2*limit)
                ids = (norm * (n_bins-1)).long()
                
            input_ids[:, :, f_idx] = ids + self.config.special_tokens
            
        return input_ids

    def decode(self, ids: torch.Tensor, scale: torch.Tensor, feature_indices: Optional[List[int]] = None) -> torch.Tensor:
        """
        Reverse binning: ID -> Value.
        ids: [B, T, F] (Long)
        scale: [B, 1, F] (Float)
        returns: values [B, T, F] (Float)
        """
        B, T, F = ids.shape
        device = ids.device
        values = torch.zeros((B, T, F), params=None, dtype=torch.float32, device=device)
        
        feat_names = self.config.feature_names
        
        for f_idx in range(F):
            global_idx = feature_indices[f_idx] if feature_indices else f_idx
            fname = feat_names[global_idx]
            token_ids = ids[:, :, f_idx]
            
            # Remove special tokens offset
            bin_ids = token_ids - self.config.special_tokens
            
            # Mask special tokens? if id < special_tokens -> 0 or NaN?
            # We assume generated tokens are valid content tokens.
            
            edges = self.config.quantiles.get(fname)
            if edges:
                # Quantile Decoding
                # Map bin_id `k` to (edges[k] + edges[k+1]) / 2
                # edges has n_bins + 1 elements.
                # bin_ids in [0, n_bins-1]
                
                # Convert list to tensor
                bins = torch.tensor(edges, device=device, dtype=torch.float32)
                
                # Clamp for safety
                n_bins = len(edges) - 1
                safe_ids = torch.clamp(bin_ids, 0, n_bins - 1)
                
                # Gather edges
                # left edge = bins[safe_ids]
                # right edge = bins[safe_ids + 1]
                left = bins[safe_ids]
                
                # Handle right edge safely
                # If safe_ids == n_bins - 1, right is bins[n_bins] (last element)
                # But we can just use torch.gather or simple indexing if safe_ids shape matches
                # bins[safe_ids + 1] works because safe_ids max is len(bins)-2
                
                right = bins[safe_ids + 1]
                
                # Center
                vals_scaled = (left + right) / 2.0
                
            else:
                # Linear Fallback
                limit = 10.0
                n_bins = self.config.vocab_size - self.config.special_tokens
                norm = bin_ids.float() / (n_bins - 1)
                vals_scaled = norm * (2 * limit) - limit
                
            # Apply scale: val = scaled * scale
            # scale slice: [B, 1]
            s = scale[:, :, f_idx] # [B, 1]
            values[:, :, f_idx] = vals_scaled * s
            
        return values
