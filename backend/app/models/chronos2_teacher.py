"""
Chronos 2 Teacher Shared Module.
Contains logic for:
1. Model Architecture Probing (identifying Seq2Seq vs Causal, signature inspection).
2. LoRA Target Discovery (Regex-based).
3. Unified Model Loading (Base + QLoRA + Adapter).
4. Model Wrapper (Shape Adaptation).
"""

import sys
import re
import inspect
from typing import Tuple, Dict, Any, List, Optional, Union
from pathlib import Path

import torch
import torch.nn as nn
from transformers import (
    AutoConfig, AutoModel, AutoModelForSeq2SeqLM, AutoModelForCausalLM,
    BitsAndBytesConfig, PreTrainedModel
)
from peft import get_peft_model, LoraConfig, PeftModel, prepare_model_for_kbit_training, TaskType

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
