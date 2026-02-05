"""
Chronos 2 Teacher Shared Module.
Contains logic for:
1. Model Architecture Probing (identifying Seq2Seq vs Causal, signature inspection).
2. LoRA Target Discovery (Regex-based).
3. Unified Model Loading (Base + QLoRA + Adapter).
4. Model Wrapper (Shape Adaptation).
5. Teacher Inference (Priors Generation).
"""

import sys
import re
import inspect
from typing import Tuple, Dict, Any, List, Optional, Union
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
from transformers import (
    AutoConfig, AutoModel, AutoModelForSeq2SeqLM, AutoModelForCausalLM,
    BitsAndBytesConfig, PreTrainedModel
)
from peft import get_peft_model, LoraConfig, PeftModel, prepare_model_for_kbit_training, TaskType
from backend.app.models.signal_types import ChronosPriors

class Chronos2ModelWrapper(nn.Module):
    """
    Wraps the HF Model to handle input shape adaptation (Multivariate -> Univariate).
    """
    def __init__(self, model: nn.Module, model_type: str, forward_params: List[str]):
        super().__init__()
        self.model = model
        self.model_type = model_type
        self.forward_params = forward_params
        
        # Heuristic: T5 is univariate-only usually.
        # If config says "chronos" specific, maybe it supports [B, T, F].
        # For now, we assume standard HF models are [B, T].
        self.force_univariate = True
        if "chronos" in model_type.lower() and "t5" not in model_type.lower():
             # Pure Chronos model check? 
             pass

    def prepare_inputs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                       labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Adapts [B, T, F] inputs to model expectations (e.g. [B*F, T]).
        """
        B, T, F = input_ids.shape
        
        if self.force_univariate:
            # Flatten: [B, T, F] -> [B*F, T]
            # We treat each feature as an independent sample in the batch
            # Transpose first? [B, F, T] -> [B*F, T] guarantees (b0f0, b0f1, ...)
            
            # Input IDs
            # Permute to [B, F, T] then reshape
            flat_ids = input_ids.permute(0, 2, 1).reshape(B * F, T)
            
            # Mask
            # Mask is [B, T]. Repeat for features.
            # [B, 1, T] -> [B, F, T] -> [B*F, T]
            flat_mask = attention_mask.unsqueeze(1).expand(B, F, T).reshape(B * F, T)
            
            out = {
                "input_ids": flat_ids,
                "attention_mask": flat_mask
            }
            
            if labels is not None:
                # Labels: [B, Pred, F]
                # Flatten similarly: [B, F, Pred] -> [B*F, Pred]
                _, P, _ = labels.shape
                flat_labels = labels.permute(0, 2, 1).reshape(B * F, P)
                out["labels"] = flat_labels
                
            return out
            
        else:
            # Native Multivariate Support
            out = {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }
            if labels is not None:
                out["labels"] = labels
            return out

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        # Prepare adapted inputs (e.g. [B*F, T])
        inputs = self.prepare_inputs(input_ids, attention_mask, labels)
        
        # Merged prepared inputs with extra kwargs, but filter for model signature
        call_kwargs = {}
        for k in self.forward_params:
            if k in inputs:
                call_kwargs[k] = inputs[k]
            elif k in kwargs:
                call_kwargs[k] = kwargs[k]
        
        # Force use_cache=False during training to prevent cache_position CUDA crashes
        if "use_cache" in self.forward_params:
             call_kwargs["use_cache"] = False
        
        # Explicit model config override just in case
        if hasattr(self.model, "config"):
             self.model.config.use_cache = False
        
        # Debug: Print signature vs call if it's the first step
        if not hasattr(self, "_probed"):
             print(f"[Wrapper] Model Signature: {self.forward_params}")
             print(f"[Wrapper] Calling with keys: {list(call_kwargs.keys())}")
             self._probed = True
             
        return self.model(**call_kwargs)
    
    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                 prediction_length: int = 10, num_samples: int = 20, **kwargs) -> torch.Tensor:
        """
        Generate forecasts.
        input_ids: [B, T, F]
        Returns: [B, NumSamples, PredictionLength, F]
        """
        B, T, F = input_ids.shape
        inputs = self.prepare_inputs(input_ids, attention_mask)

        # Generate: [B*F, NumSamples, PredictionLength] (assuming standard HF generate)
        # Note: HF generate outputs usually [Batch, SeqLen] (if greedy) or [Batch*NumSamples, SeqLen] or [Batch, NumSamples, SeqLen]
        # Depending on configuration.
        # For T5/Seq2Seq: output is [Batch, SeqLen]

        # We need to map back to [B, NumSamples, Pred, F]

        # This implementation depends heavily on the underlying model's generate behavior.
        # Assuming we can use model.generate()

        # If T5:
        # outputs = model.generate(input_ids=..., num_return_sequences=num_samples, max_new_tokens=prediction_length)
        # Output shape: [B*F * num_samples, prediction_length]

        gen_kwargs = {
            "max_new_tokens": prediction_length,
            "num_return_sequences": num_samples,
            "do_sample": True,
            "use_cache": True
        }
        gen_kwargs.update(kwargs)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        # outputs: [B*F * num_samples, pred_len] (usually padded)
        # Reshape logic
        # First dim is (Batch * F) repeated num_samples times?
        # Usually HF interleaves: sample1_for_input1, sample2_for_input1, ...

        # Total inputs = B * F
        # Total outputs = B * F * num_samples

        # Reshape to [B*F, num_samples, pred_len]
        # But wait, output is tensor of token IDs.

        # We assume the caller handles decoding (detokenization).
        # We return the token IDs shaped as [B, F, NumSamples, PredLen]

        # If output is [N_seqs, Len]
        pred_len = outputs.shape[1]

        # [B*F, num_samples, pred_len]
        outputs = outputs.view(B*F, num_samples, pred_len)

        # [B, F, num_samples, pred_len]
        outputs = outputs.view(B, F, num_samples, pred_len)

        # Permute to [B, num_samples, pred_len, F]
        outputs = outputs.permute(0, 2, 3, 1)

        return outputs

    def save_pretrained(self, out_dir: Union[str, Path]):
        # Save underlying model/adapter
        self.model.save_pretrained(out_dir)


def probe_model_architecture(model_id: str, use_qlora: bool) -> Tuple[Any, Dict[str, Any]]:
    print(f"[Probe] Inspecting {model_id}...")
    try:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load config for {model_id}: {e}")

    info = {
        "architectures": getattr(config, "architectures", []),
        "model_type": getattr(config, "model_type", "unknown"),
        "vocab_size": getattr(config, "vocab_size", getattr(config, "n_positions", "unknown")),
        "d_model": getattr(config, "d_model", getattr(config, "hidden_size", "unknown")),
    }
    print(f"[Probe] Config: {info}")

    model_class = None
    archs = info["architectures"]
    
    if archs:
        if any(x in a for a in archs for x in ("T5", "Chronos")):
             model_class = AutoModelForSeq2SeqLM
        elif any("CausalLM" in a for a in archs):
             model_class = AutoModelForCausalLM
    
    if not model_class:
        t = info["model_type"].lower()
        if "t5" in t:
             model_class = AutoModelForSeq2SeqLM
        elif "gpt" in t or "llama" in t:
             model_class = AutoModelForCausalLM

    if not model_class:
        print("[Probe] WARN: Could not infer specific class, using AutoModel.")
        model_class = AutoModel

    print(f"[Probe] Selected Model Class: {model_class.__name__}")
    return model_class, info


def load_chronos_adapter(
    model_id: str,
    use_qlora: bool,
    device: torch.device,
    adapter_path: Optional[Path] = None,
    lora_config: Optional[Dict[str, Any]] = None,
    eval_mode: bool = False
) -> Tuple[Chronos2ModelWrapper, Dict[str, Any]]:
    """
    Loads model, applies QLoRA, wraps in Chronos2ModelWrapper.
    """
    ModelClass, info = probe_model_architecture(model_id, use_qlora)
    
    bnb_config = None
    if use_qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    print(f"[Load] Model Class: {ModelClass.__name__}")
    try:
        # Load with device_map="auto" for QLoRA/Quantization
        model = ModelClass.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto" if use_qlora else None,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        # Force disable cache globally
        model.config.use_cache = False
        
        # Force to device early if not quantized
        if not use_qlora:
            model.to(device)
            print(f"[Load] Model moved to {device}")
            
        # Debug: Check vocab size vs embeddings
        print(f"[Load] Config Vocab: {model.config.vocab_size}")
        if model.config.vocab_size < 4096:
             print("[Load] Resizing embeddings to 4096...")
             model.resize_token_embeddings(4096)
             print(f"[Load] New Vocab: {model.config.vocab_size}")

        # Fix T5 IDs for training if missing
        if model.config.pad_token_id is None:
             model.config.pad_token_id = 0
        if model.config.decoder_start_token_id is None:
             model.config.decoder_start_token_id = 0
        if model.config.eos_token_id is None:
             model.config.eos_token_id = 1

        if hasattr(model, "get_input_embeddings"):
             emb = model.get_input_embeddings()
             print(f"[Load] Embedding Shape: {emb.weight.shape}")
    except Exception as e:
        print(f"[Load] from_pretrained failed: {e}")
        raise e

    # Sig check
    forward_params = []
    if hasattr(model, "forward"):
        # Handle cases where forward is a property or has complex wrapper
        try:
            sig = inspect.signature(model.forward)
            forward_params = list(sig.parameters.keys())
        except:
             # Fallback if signature fails
             forward_params = ["input_ids", "attention_mask", "labels"]
        info["forward_params"] = forward_params
    
    print(f"[Load] Forward Params: {forward_params}")

    # K-bit prep
    if use_qlora:
        print("[Load] Preparing for kbit training...")
        model = prepare_model_for_kbit_training(model)
        # Quantized models usually need to stay on the device they were loaded on (auto-mapped)
        # But since we didn't use device_map, we might need it now.
        # However, we disabled QLoRA for now.

    # Adapter
    if adapter_path and adapter_path.exists():
        print(f"[Load] Loading adapter from {adapter_path}...")
        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=not eval_mode)
    elif lora_config:
        print(f"[Load] Init NEW LoRA...")
        targets = find_lora_targets(model)
        print(f"[Load] LoRA Targets: {targets}")
        
        peft_cfg = LoraConfig(
            r=lora_config.get("rank", 16),
            lora_alpha=lora_config.get("alpha", 32),
            target_modules=targets,
            lora_dropout=lora_config.get("dropout", 0.05),
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM if "t5" in info["model_type"].lower() else TaskType.CAUSAL_LM
        )
        model = get_peft_model(model, peft_cfg)
        
        # Ensure resized embeddings/head are trainable
        # PEFT often freezes these if they are considered "base model"
        for name, param in model.named_parameters():
             if "shared" in name or "embed_tokens" in name or "lm_head" in name or "wte" in name:
                  param.requires_grad = True

    # Enable gradient checkpointing AFTER PEFT wrapping
    if not eval_mode and use_qlora:
        if hasattr(model, "gradient_checkpointing_enable"):
             print("[Load] Enabling gradient checkpointing (non-reentrant)...")
             model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        
    if not use_qlora:
        model.to(device)
        
    if eval_mode:
        model.eval()
    else:
        model.train()
        
    # Wrap
    wrapper = Chronos2ModelWrapper(model, info["model_type"], forward_params)
    return wrapper, info


def find_lora_targets(model: nn.Module) -> List[str]:
    modules = {name for name, _ in model.named_modules() if isinstance(_, nn.Linear)}
    patterns = [
        r".*attention.*(q|k|v|o)_proj",
        r".*SelfAttention\.(q|k|v|o)",
        r".*(q|v)_proj",
        r".*(q|v)$"
    ]
    targets = set()
    for p in patterns:
        regex = re.compile(p, re.IGNORECASE)
        matches = {m.split('.')[-1] for m in modules if regex.match(m)}
        if matches:
            targets.update(matches)
            
    if not targets:
        return ["q", "v"]
        
    final = [t for t in targets if any(x in t.lower() for x in ('q', 'v'))]
    if not final:
         final = list(targets)
    return list(final)

def infer_priors(
    model: Chronos2ModelWrapper,
    codec: Any, # Chronos2Codec
    input_tensor: torch.Tensor, # [B, T, F]
    horizon_days: int = 20,
    n_samples: int = 20
) -> List[ChronosPriors]:
    """
    Generates priors for a batch of tickers.
    input_tensor: [B, T, F] normalized/raw depending on codec expectations.
                  Codec encode needs raw? Usually codec expects raw.
                  We assume input_tensor is RAW here.
    """
    device = input_tensor.device

    # 1. Encode
    # Codec expects raw [B, T, F]
    encoded = codec.encode(input_tensor)
    input_ids = encoded["input_ids"] # [B, T, F]
    attention_mask = encoded["attention_mask"] # [B, T]
    scale = encoded["scale"] # [B, 1, F]

    # 2. Generate
    # Output: [B, NumSamples, PredLen, F] (Tokens)
    token_preds = model.generate(input_ids, attention_mask, prediction_length=horizon_days, num_samples=n_samples)

    # 3. Decode
    # We need a decode method in codec or manual.
    # We will assume a 'decode' method that takes tokens + scale -> values
    # [B, NumSamples, PredLen, F]

    # Flatten to decode batchwise if needed, or loop?
    # Codec decode usually takes [B, T, F] tokens + [B, 1, F] scale
    # Here we have extra dims.

    # Let's trust codec.decode can handle or we loop.
    # Simpler: Compute stats on tokens? No, need values.

    # Assume decode works on [*, T, F]
    B, N, P, F = token_preds.shape

    # Reshape to [B*N, P, F]
    flat_tokens = token_preds.reshape(B*N, P, F)
    # Expand scale: [B, 1, F] -> [B, N, 1, F] -> [B*N, 1, F]
    flat_scale = scale.unsqueeze(1).expand(B, N, 1, F).reshape(B*N, 1, F)

    # Decode
    # decode returns [B*N, P, F] values
    # We need codec.decode to support this.
    # If not implemented, we implement basic logic here based on binning.

    # Quick implementation of decoding if codec doesn't fully support
    if hasattr(codec, "decode"):
         flat_values = codec.decode(flat_tokens, flat_scale)
    else:
         # Minimal fallback if codec.decode not ready
         # Assume buckets
         # This is fragile. We should ensure Codec has decode.
         # For now, return dummy if failing? No.
         raise NotImplementedError("Codec must implement decode()")

    # Reshape back: [B, N, P, F]
    values = flat_values.reshape(B, N, P, F)

    # 4. Compute Statistics (Priors)
    # Target feature: usually "close" or "log_return"?
    # We assume feature 0 is the primary price/return feature for stats.
    # Or we calculate for all?
    # Let's assume input_tensor[:, :, 0] is price-like or return-like.
    # If it's Price:
    # Drift = (End - Start) / Start
    # Vol = std(returns)

    # If it's Returns:
    # Drift = mean(returns)
    # Vol = std(returns)

    # We assume the model predicts Prices if raw input was prices.
    # Let's assume Price.

    # values: [B, N, P, F]
    # Use feature 0 (Close)
    price_paths = values[:, :, :, 0] # [B, N, P]

    # Current Price (last observed)
    current_price = input_tensor[:, -1, 0].unsqueeze(1) # [B, 1]

    # Calculate returns relative to current
    # paths / current - 1.0
    # [B, N, P]
    cum_returns = (price_paths / current_price.unsqueeze(2)) - 1.0

    # Terminal returns (at horizon)
    terminal_returns = cum_returns[:, :, -1] # [B, N]

    priors_list = []

    for b in range(B):
        # Drift: Median terminal return
        drift = float(torch.median(terminal_returns[b]))

        # Vol: Std of terminal returns across samples (uncertainty of outcome)
        # OR volatility of the path?
        # Usually "Vol" in priors means implied vol regime.
        # Let's use std of terminal returns as proxy for uncertainty.
        vol = float(torch.std(terminal_returns[b]))

        # Downside Q10: 10th percentile of terminal returns
        downside_q10 = float(torch.quantile(terminal_returns[b], 0.10))

        # Trend Conf: Fraction of paths > 0
        trend_conf = float((terminal_returns[b] > 0).float().mean())

        priors_list.append(ChronosPriors(
            drift=drift,
            vol_forecast=vol,
            tail_risk=downside_q10,
            trend_conf=trend_conf,
            metadata={"n_samples": n_samples, "horizon": horizon_days}
        ))

    return priors_list
