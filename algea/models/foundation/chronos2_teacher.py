"""
Chronos-2 Teacher — model wrappers, quantile head, LoRA loading, and inference.

Ported from deprecated/backend_app_snapshot/models/chronos2_teacher.py.

Heavy imports (``chronos``, ``peft``, ``transformers``) are guarded behind
try/except so this module loads cleanly on CPU-only environments.
"""
from __future__ import annotations

import inspect
import logging
import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Optional heavy-weight imports
# ---------------------------------------------------------------------------
try:
    from transformers import (
        AutoConfig,
        AutoModel,
        AutoModelForCausalLM,
        AutoModelForSeq2SeqLM,
        BitsAndBytesConfig,
    )
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False

try:
    from peft import (
        LoraConfig,
        PeftModel,
        TaskType,
        get_peft_model,
        inject_adapter_in_model,
        prepare_model_for_kbit_training,
    )
    _HAS_PEFT = True
except ImportError:
    _HAS_PEFT = False


# ═══════════════════════════════════════════════════════════════════════════
# Utility helpers
# ═══════════════════════════════════════════════════════════════════════════

logger = logging.getLogger(__name__)

# One-shot warning flags for inference API
_INFERENCE_WARNINGS_ISSUED: set = set()


def _warn_once(key: str, msg: str) -> None:
    """Emit a UserWarning at most once per key per process."""
    if key not in _INFERENCE_WARNINGS_ISSUED:
        _INFERENCE_WARNINGS_ISSUED.add(key)
        warnings.warn(msg, UserWarning, stacklevel=3)


def _inverse_softplus(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Numerically stable inverse of softplus: ``log(exp(x) - 1)``."""
    return torch.where(x > 20.0, x, torch.log(torch.expm1(x).clamp(min=eps)))


def _get_standard_quantile_levels(count: int = 21) -> List[float]:
    """Standard quantile levels shared across the system."""
    levels = [
        0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
        0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99,
    ]
    return levels[:count]


# ═══════════════════════════════════════════════════════════════════════════
# LearnedQuantileShape
# ═══════════════════════════════════════════════════════════════════════════

class LearnedQuantileShape(nn.Module):
    """
    Monotone quantile shape ``z_τ``.

    Produces a strictly non-decreasing tensor ``z`` of shape ``[Q]`` for *Q*
    quantile levels, initialised to standard-normal quantile values.

    Parameterisation:
        ``z_raw[0] = z0``; ``z_raw[i] = z_raw[i-1] + softplus(u[i-1]) + ε``
    Optionally normalised to ``mean=0, std=1`` so that σ stays interpretable.
    """

    def __init__(self, quantiles_count: int = 21, normalize: bool = True) -> None:
        super().__init__()
        self.quantiles_count = quantiles_count
        self.normalize = normalize
        self.eps = 1e-6

        from scipy.stats import norm as _norm

        tau_levels = _get_standard_quantile_levels(quantiles_count)
        z_init = np.array([_norm.ppf(tau) for tau in tau_levels], dtype=np.float32)
        z_init_t = torch.from_numpy(z_init)

        if normalize:
            z_mean = z_init_t.mean()
            z_std = z_init_t.std() + 1e-6
            z_init_normalized = (z_init_t - z_mean) / z_std
        else:
            z_init_normalized = z_init_t
        self.register_buffer("z_init_normalized", z_init_normalized)

        self.z0 = nn.Parameter(torch.tensor(z_init[0]))
        gaps = np.diff(z_init)
        gaps_t = torch.from_numpy(gaps)
        u_init = _inverse_softplus(gaps_t - self.eps)
        self.u = nn.Parameter(u_init)

    def forward(self) -> torch.Tensor:
        d = F.softplus(self.u) + self.eps
        z_raw = torch.cat([self.z0.unsqueeze(0), self.z0 + torch.cumsum(d, dim=0)])
        if self.normalize:
            z_mean = z_raw.mean()
            z_std = z_raw.std() + 1e-6
            z = (z_raw - z_mean) / z_std
        else:
            z = z_raw
        return z


# ═══════════════════════════════════════════════════════════════════════════
# Chronos2QuantileHead
# ═══════════════════════════════════════════════════════════════════════════

class Chronos2QuantileHead(nn.Module):
    """
    Location-scale quantile head with learned monotone shape and temperature.

    ``q_τ = μ + (T · σ) · z_τ``

    * ``z_τ`` — learned monotone shape (``LearnedQuantileShape``)
    * ``T``  — global temperature scalar (clamped to ``[1/3, 3]``)
    * ``μ``, ``σ`` — sample-dependent via linear projections
    """

    def __init__(
        self,
        quantiles_count: int = 21,
        hidden_size: int = 128,
        input_size: int = 180,
        target_mean: float = -0.00088,
        target_std: float = 0.032,
    ) -> None:
        super().__init__()
        self.quantiles_count = quantiles_count
        self.sigma_floor = 1e-4

        self.encoder = nn.Sequential(nn.Linear(input_size, hidden_size), nn.GELU())

        self.mu_head = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.mu_head.weight)
        nn.init.constant_(self.mu_head.bias, target_mean)

        self.sigma_head = nn.Linear(hidden_size, 1)
        sigma_bias_init = math.log(math.exp(target_std) - 1)
        nn.init.zeros_(self.sigma_head.weight)
        nn.init.constant_(self.sigma_head.bias, sigma_bias_init)

        self.shape = LearnedQuantileShape(quantiles_count=quantiles_count, normalize=True)

        self.logT = nn.Parameter(torch.tensor(0.0))
        self._logT_lo = math.log(1.0 / 3.0)
        self._logT_hi = math.log(3.0)

    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(torch.clamp(self.logT, self._logT_lo, self._logT_hi))

    @property
    def temperature_value(self) -> float:
        return self.temperature.item()

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Handle migration from old ``t_raw`` (softplus) to ``logT`` (clamped exp)."""
        if "logT" not in state_dict and "t_raw" in state_dict:
            t_raw_val = state_dict.pop("t_raw")
            T_old = F.softplus(t_raw_val) + 1e-4
            logT_migrated = torch.log(T_old)
            state_dict["logT"] = logT_migrated
            print(f"[Migration] Converted t_raw -> logT: T_old={T_old.item():.4f}, logT={logT_migrated.item():.4f}")
        return super().load_state_dict(state_dict, strict=strict)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Returns monotonic quantiles ``[B, Q]``."""
        h = self.encoder(embeddings)
        mu = self.mu_head(h).squeeze(-1)
        raw_sigma = self.sigma_head(h).squeeze(-1)
        sigma = F.softplus(raw_sigma) + self.sigma_floor

        z = self.shape()
        T = self.temperature
        sigma_eff = sigma * T
        quantiles = mu.unsqueeze(1) + sigma_eff.unsqueeze(1) * z.unsqueeze(0)
        return quantiles


# ═══════════════════════════════════════════════════════════════════════════
# Chronos2ModelWrapper (HF Seq2Seq / CausalLM)
# ═══════════════════════════════════════════════════════════════════════════

class Chronos2ModelWrapper(nn.Module):
    """Wraps a HuggingFace model for input shape adaptation (multivariate → univariate)."""

    def __init__(self, model: nn.Module, model_type: str, forward_params: List[str]) -> None:
        super().__init__()
        self.model = model
        self.model_type = model_type
        self.forward_params = forward_params
        self.force_univariate = True

    def prepare_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B, T, F = input_ids.shape
        if self.force_univariate:
            flat_ids = input_ids.permute(0, 2, 1).reshape(B * F, T)
            flat_mask = attention_mask.unsqueeze(1).expand(B, F, T).reshape(B * F, T)
            out: Dict[str, torch.Tensor] = {"input_ids": flat_ids, "attention_mask": flat_mask}
            if labels is not None:
                _, P, _ = labels.shape
                out["labels"] = labels.permute(0, 2, 1).reshape(B * F, P)
            return out
        return {"input_ids": input_ids, "attention_mask": attention_mask, **({"labels": labels} if labels is not None else {})}

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        inputs = self.prepare_inputs(input_ids, attention_mask, labels)
        call_kwargs = {}
        for k in self.forward_params:
            if k in inputs:
                call_kwargs[k] = inputs[k]
            elif k in kwargs:
                call_kwargs[k] = kwargs[k]
        if "use_cache" in self.forward_params:
            call_kwargs["use_cache"] = False
        if hasattr(self.model, "config"):
            self.model.config.use_cache = False
        return self.model(**call_kwargs)

    def save_pretrained(self, out_dir: Union[str, Path]) -> None:
        self.model.save_pretrained(out_dir)


# ═══════════════════════════════════════════════════════════════════════════
# Chronos2NativeWrapper (patch-based Chronos-2)
# ═══════════════════════════════════════════════════════════════════════════

class Chronos2NativeWrapper(nn.Module):
    """
    Wraps native Chronos-2 patch-based models.

    Includes an optional ``Chronos2QuantileHead`` for 10-day quantile prediction
    (disabled by default until encoder extraction is validated).
    """

    def __init__(
        self,
        model: nn.Module,
        model_type: str,
        forward_params: List[str],
        quantiles_count: int = 21,
        head_hidden_size: int = 128,
    ) -> None:
        super().__init__()
        self.model = model
        self.model_type = model_type
        self.forward_params = forward_params
        self._enable_q10d_head = False

        input_size = 180
        if hasattr(model, "config"):
            input_size = getattr(
                model.config, "d_model",
                getattr(model.config, "hidden_size",
                        getattr(model.config, "n_embd", 180)),
            )

        self.quantile_head = Chronos2QuantileHead(
            quantiles_count=quantiles_count,
            hidden_size=head_hidden_size,
            input_size=input_size,
        )
        try:
            head_device = next(model.parameters()).device
            self.quantile_head.to(head_device)
        except StopIteration:
            pass

    # ------ helpers -------------------------------------------------------

    @staticmethod
    def _build_mask(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 3:
            mask = torch.isfinite(tensor).all(dim=-1)
        else:
            mask = torch.isfinite(tensor)
        return mask.to(dtype=torch.long)

    @staticmethod
    def _sanitize(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if tensor is None:
            return None
        return torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)

    def _pool_context_representation(
        self, context: torch.Tensor, outputs: Optional[Any] = None
    ) -> torch.Tensor:
        hidden = None
        if outputs is not None:
            # Try attribute-based access (HF model outputs)
            for attr in ("encoder_last_hidden_state", "last_hidden_state"):
                hidden = getattr(outputs, attr, None)
                if hidden is not None:
                    break
            if hidden is None and getattr(outputs, "hidden_states", None):
                hidden = outputs.hidden_states[-1]
            if hidden is None and isinstance(outputs, dict):
                hidden = outputs.get("encoder_last_hidden_state") or outputs.get("last_hidden_state")
        if hidden is not None:
            return hidden.mean(dim=1)
        if context.ndim == 3:
            return context.mean(dim=1)
        return context

    @staticmethod
    def _attach_q10d(outputs: Any, q_10d: torch.Tensor) -> Any:
        if hasattr(outputs, "__dict__"):
            outputs.q_10d = q_10d
            return outputs
        return {"outputs": outputs, "q_10d": q_10d}

    # ------ 10-day quantile prediction ------------------------------------

    def predict_quantiles_10d(
        self,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        past_covariates: Optional[torch.Tensor] = None,
        future_covariates: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not getattr(self, "_enable_q10d_head", True):
            raise RuntimeError("predict_quantiles_10d called but quantile head is disabled.")

        context = self._sanitize(context)
        past_covariates = self._sanitize(past_covariates)
        future_covariates = self._sanitize(future_covariates)
        if context_mask is None:
            context_mask = self._build_mask(context)

        outputs = None
        encoder_hidden = None
        if "num_output_patches" in self.forward_params:
            try:
                dummy_target = torch.zeros(context.shape[0], 1, 1, device=context.device, dtype=context.dtype)
                model_output = self.model(
                    context=context, context_mask=context_mask,
                    future_target=dummy_target, num_output_patches=1, return_dict=True,
                )
                enc_out = getattr(model_output, "encoder_outputs", None)
                if enc_out is not None:
                    if isinstance(enc_out, tuple) and len(enc_out) > 0:
                        encoder_hidden = enc_out[0]
                    else:
                        encoder_hidden = getattr(enc_out, "last_hidden_state", None)
                if encoder_hidden is None:
                    encoder_hidden = getattr(model_output, "encoder_last_hidden_state", None)
                outputs = model_output
            except Exception:
                pass

        if encoder_hidden is not None and encoder_hidden.ndim == 3:
            rep = encoder_hidden.mean(dim=1)
        else:
            rep = self._pool_context_representation(context, outputs)
        return self.quantile_head(rep)

    # ------ num_output_patches inference ----------------------------------

    def _infer_num_output_patches(self, pred_len: int) -> Optional[int]:
        if pred_len <= 0:
            return None
        patch_len = None
        if hasattr(self.model, "config"):
            patch_len = getattr(self.model.config, "patch_length", None) or getattr(self.model.config, "patch_size", None)
        if not patch_len:
            patch_len = 16
        return int(np.ceil(pred_len / patch_len))

    # ------ forward -------------------------------------------------------

    def forward(
        self,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        future_target: Optional[torch.Tensor] = None,
        future_target_mask: Optional[torch.Tensor] = None,
        past_covariates: Optional[torch.Tensor] = None,
        future_covariates: Optional[torch.Tensor] = None,
        future_covariates_mask: Optional[torch.Tensor] = None,
        group_ids: Optional[torch.Tensor] = None,
        num_output_patches: Optional[int] = None,
        **kwargs,
    ):
        context = self._sanitize(context)
        future_target = self._sanitize(future_target)
        past_covariates = self._sanitize(past_covariates)
        future_covariates = self._sanitize(future_covariates)

        if context_mask is None:
            context_mask = self._build_mask(context)
        if future_target is not None and future_target_mask is None:
            future_target_mask = self._build_mask(future_target)
        if num_output_patches is None and future_target is not None:
            num_output_patches = self._infer_num_output_patches(future_target.shape[1])

        # Collapse multi-feature future covariates
        if future_covariates is not None and future_covariates.ndim == 3:
            future_covariates = future_covariates.mean(dim=-1)
            if future_covariates_mask is not None and future_covariates_mask.ndim == 3:
                future_covariates_mask = future_covariates_mask.all(dim=-1).to(dtype=torch.long)

        arg_map = {
            "context": context, "context_mask": context_mask,
            "future_target": future_target, "future_target_mask": future_target_mask,
            "future_mask": future_target_mask,
            "past_covariates": past_covariates,
            "future_covariates": future_covariates,
            "future_covariates_mask": future_covariates_mask,
            "past_dynamic_feat": past_covariates,
            "future_dynamic_feat": future_covariates,
            "known_covariates": future_covariates,
            "group_ids": group_ids,
            "num_output_patches": num_output_patches,
        }

        call_kwargs = {}
        for k in self.forward_params:
            if k in arg_map and arg_map[k] is not None:
                call_kwargs[k] = arg_map[k]
            elif k in kwargs:
                call_kwargs[k] = kwargs[k]

        if "use_cache" in self.forward_params:
            call_kwargs["use_cache"] = False
        if hasattr(self.model, "config"):
            self.model.config.use_cache = False
        if num_output_patches is not None:
            call_kwargs["num_output_patches"] = num_output_patches

        return self.model(**call_kwargs)

    # ------ generate ------------------------------------------------------

    def generate(
        self,
        context: torch.Tensor,
        prediction_length: int = 10,
        num_samples: int = 20,
        context_mask: Optional[torch.Tensor] = None,
        past_covariates: Optional[torch.Tensor] = None,
        future_covariates: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Generate forecast sample paths.

        Input contract:
            context: ``[B, T, F]`` or ``[T, F]`` (auto-batched to ``[1, T, F]``).
            Univariate (F=1) is the standard case for NLL-trained teachers.

        Output contract:
            ``[B, S, P, F]`` where S=num_samples (1 for deterministic),
            P=prediction_length, F=features.

        For ``native_nll`` models backed by ``Chronos2Pipeline.predict``:
            - ``num_samples``, ``past_covariates``, ``future_covariates`` are
              silently ignored (the pipeline is deterministic).
            - Context is reshaped from ``[B, T, F]`` to ``[B, F, T]``
              (``n_series, n_variates, history_length``).
        """
        # --- Input validation & auto-expand --------------------------------
        context = self._sanitize(context)
        if context.ndim == 2:
            context = context.unsqueeze(0)  # [T, F] -> [1, T, F]
        if context.ndim != 3:
            raise ValueError(
                f"generate() expects context of shape [B, T, F] or [T, F], "
                f"got ndim={context.ndim}, shape={tuple(context.shape)}"
            )

        past_covariates = self._sanitize(past_covariates)
        future_covariates = self._sanitize(future_covariates)
        if context_mask is None:
            context_mask = self._build_mask(context)

        # --- Reshape context for pipeline.predict --------------------------
        # Pipeline expects (n_series, n_variates, history_length) — [B, F, T]
        ctx_for_predict = context.permute(0, 2, 1)  # [B, T, F] -> [B, F, T]

        # Guard against accidental truncation via context_length kwarg
        if "context_length" in kwargs:
            expected_t = context.shape[1]
            if kwargs["context_length"] != expected_t:
                raise ValueError(
                    f"context_length={kwargs['context_length']} does not match "
                    f"actual context time-steps T={expected_t}. This would cause "
                    f"silent truncation or padding."
                )

        with torch.no_grad():
            if hasattr(self.model, "predict"):
                # Chronos2Pipeline.predict API:
                #   predict(inputs, prediction_length, batch_size, context_length,
                #           cross_learning, limit_prediction_length)
                # It does NOT accept past/future_covariates or num_samples.
                _PIPELINE_PARAMS = ("batch_size", "context_length", "cross_learning")
                pipeline_kwargs = {k: v for k, v in kwargs.items()
                                   if k in _PIPELINE_PARAMS}
                outputs = self.model.predict(
                    ctx_for_predict, prediction_length=prediction_length,
                    limit_prediction_length=False,
                    **pipeline_kwargs,
                )
            elif hasattr(self.model, "generate"):
                outputs = self.model.generate(
                    context=ctx_for_predict, context_mask=context_mask,
                    prediction_length=prediction_length, num_samples=num_samples,
                    past_covariates=past_covariates, future_covariates=future_covariates,
                    **kwargs,
                )
            else:
                raise RuntimeError("Chronos native model lacks predict/generate.")

        # --- Normalize output shape to [B, S, P, F] -----------------------
        # pipeline.predict may return a list of tensors (one per series)
        if isinstance(outputs, (list, tuple)):
            outputs = torch.stack([torch.as_tensor(o) for o in outputs])
        outputs = torch.as_tensor(outputs, device=context.device)
        if outputs.ndim == 2:
            outputs = outputs.unsqueeze(1)  # [B, P] -> [B, 1, P]
        if outputs.ndim == 3:
            outputs = outputs.unsqueeze(-1)  # [B, S, P] -> [B, S, P, 1]
        return outputs

    def save_pretrained(self, out_dir: Union[str, Path]) -> None:
        self.model.save_pretrained(out_dir)


# ═══════════════════════════════════════════════════════════════════════════
# Model loading & LoRA target discovery
# ═══════════════════════════════════════════════════════════════════════════

def find_lora_targets(model: nn.Module, include_mlp: bool = False) -> List[str]:
    """Auto-discover LoRA-targetable modules (attention and optionally MLP)."""
    modules = {name for name, _ in model.named_modules() if isinstance(_, nn.Linear)}

    patterns = [
        r".*attention.*(q|k|v|o)_proj",
        r".*SelfAttention\.(q|k|v|o)",
        r".*(q|v)_proj",
        r".*(q|v)$",
    ]
    if include_mlp:
        patterns.extend([
            r".*wi.*", r".*wo.*", r".*mlp.*(dense|fc).*", r".*(gate|up|down)_proj",
        ])

    targets: set[str] = set()
    for p in patterns:
        regex = re.compile(p, re.IGNORECASE)
        matches = {m.split(".")[-1] for m in modules if regex.match(m)}
        targets.update(matches)

    if not targets:
        start = ["q", "v"]
        if include_mlp:
            start.extend(["wi_0", "wi_1", "wo"])
        return start
    return list(targets)


def probe_model_architecture(model_id: str, use_qlora: bool) -> Tuple[Any, Dict[str, Any]]:
    """Probe Hugging Face model config to determine architecture class."""
    if not _HAS_TRANSFORMERS:
        raise ImportError("transformers is required for probe_model_architecture")

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    info: Dict[str, Any] = {
        "architectures": getattr(config, "architectures", []),
        "model_type": getattr(config, "model_type", "unknown"),
        "vocab_size": getattr(config, "vocab_size", getattr(config, "n_positions", "unknown")),
        "d_model": getattr(config, "d_model", getattr(config, "hidden_size", "unknown")),
    }

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
        model_class = AutoModel

    return model_class, info


def load_chronos_adapter(
    model_id: str,
    use_qlora: bool,
    device: torch.device,
    adapter_path: Optional[Path] = None,
    lora_config: Optional[Dict[str, Any]] = None,
    eval_mode: bool = False,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Load model, apply QLoRA / adapter, wrap in ``Chronos2NativeWrapper`` or
    ``Chronos2ModelWrapper``.
    """
    if not _HAS_TRANSFORMERS:
        raise ImportError("transformers is required for load_chronos_adapter")

    ModelClass, info = probe_model_architecture(model_id, use_qlora)

    bnb_config = None
    if use_qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )

    # Chronos 2 native pipeline loading
    if "chronos-2" in model_id.lower():
        try:
            from chronos import Chronos2Pipeline

            pipeline = Chronos2Pipeline.from_pretrained(
                model_id, device_map="auto" if use_qlora else None, dtype=torch.bfloat16,
            )
            model = pipeline.model
            if hasattr(pipeline, "predict"):
                model.predict = pipeline.predict
            if hasattr(pipeline, "generate"):
                model.generate = pipeline.generate
        except ImportError:
            raise ImportError("'chronos' package is required for Chronos-2 native loading")
        except Exception:
            model = ModelClass.from_pretrained(
                model_id, quantization_config=bnb_config,
                device_map="auto" if use_qlora else None,
                trust_remote_code=True, torch_dtype=torch.bfloat16,
            )
    else:
        model = ModelClass.from_pretrained(
            model_id, quantization_config=bnb_config,
            device_map="auto" if use_qlora else None,
            trust_remote_code=True, torch_dtype=torch.bfloat16,
        )

    if hasattr(model, "config"):
        model.config.use_cache = False
        if not use_qlora:
            model.to(device)

        # Vocab resize guard (skip for patch-based Chronos-2)
        if "chronos-2" not in model_id.lower() and model.config.vocab_size < 4096:
            try:
                model.resize_token_embeddings(4096)
            except Exception:
                pass

        if model.config.pad_token_id is None:
            model.config.pad_token_id = 0
        if model.config.decoder_start_token_id is None:
            model.config.decoder_start_token_id = 0
        if model.config.eos_token_id is None:
            model.config.eos_token_id = 1

    # Forward signature
    forward_params: List[str] = []
    if hasattr(model, "forward"):
        try:
            sig = inspect.signature(model.forward)
            forward_params = list(sig.parameters.keys())
        except (ValueError, TypeError):
            forward_params = ["input_ids", "attention_mask", "labels"]
    info["forward_params"] = forward_params

    # K-bit prep
    if use_qlora and _HAS_PEFT:
        model = prepare_model_for_kbit_training(model)

    # Adapter / LoRA
    if _HAS_PEFT:
        if adapter_path and adapter_path.exists():
            model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=not eval_mode)
        elif lora_config:
            target_mod_config = lora_config.get("target_modules", None)
            if target_mod_config == "all":
                targets = find_lora_targets(model, include_mlp=True)
            elif isinstance(target_mod_config, list):
                targets = target_mod_config
            elif isinstance(target_mod_config, str) and "," in target_mod_config:
                targets = [t.strip() for t in target_mod_config.split(",")]
            else:
                targets = find_lora_targets(model, include_mlp=False)

            if "chronos-2" in model_id.lower():
                peft_cfg = LoraConfig(
                    r=lora_config.get("rank", 16), lora_alpha=lora_config.get("alpha", 32),
                    target_modules=targets, lora_dropout=lora_config.get("dropout", 0.05), bias="none",
                )
                model = inject_adapter_in_model(peft_cfg, model)
                for name, param in model.named_parameters():
                    param.requires_grad = "lora_" in name.lower()
            else:
                t = info["model_type"].lower()
                task_type = TaskType.SEQ_2_SEQ_LM if "t5" in t else TaskType.CAUSAL_LM
                peft_cfg = LoraConfig(
                    r=lora_config.get("rank", 16), lora_alpha=lora_config.get("alpha", 32),
                    target_modules=targets, lora_dropout=lora_config.get("dropout", 0.05),
                    bias="none", task_type=task_type,
                )
                model = get_peft_model(model, peft_cfg)
                for name, param in model.named_parameters():
                    if any(kw in name for kw in ("shared", "embed_tokens", "lm_head", "wte")):
                        param.requires_grad = True

    # Gradient checkpointing
    if not eval_mode and use_qlora and hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except Exception:
            pass

    if not use_qlora:
        model.to(device)

    if eval_mode:
        model.eval()
    else:
        model.train()

    is_native = "context" in forward_params or "future_target" in forward_params
    wrapper: nn.Module
    if is_native:
        wrapper = Chronos2NativeWrapper(model, info["model_type"], forward_params)
    else:
        wrapper = Chronos2ModelWrapper(model, info["model_type"], forward_params)
    return wrapper, info


# ═══════════════════════════════════════════════════════════════════════════
# Inference — prior generation
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ChronosPriors:
    """Container for Chronos-derived distributional priors.

    All return fields represent **end-of-horizon cumulative returns**
    (terminal forecast / current price − 1).

    Core fields (backward-compatible):
        drift:        Median terminal return (≡ q50).
        vol_forecast: Std-dev of terminal returns across samples.
        tail_risk:    10th-percentile terminal return (≡ q10).
        trend_conf:   P(positive return) — alias for ``prob_up``.
        metadata:     Dict with ``mode``, ``horizon``, and path-specific info.

    Extended schema (selector-ready):
        q10, q50, q90: Terminal return quantiles.
        dispersion:    q90 − q10 (always ≥ 0 after validation).
        prob_up:       P(terminal return > 0), clamped to [0, 1].
    """

    drift: float
    vol_forecast: float
    tail_risk: float
    trend_conf: float
    metadata: Dict[str, Any]
    # Extended schema fields — default to 0.0 for backward compat
    q10: float = 0.0
    q50: float = 0.0
    q90: float = 0.0
    dispersion: float = 0.0
    prob_up: float = 0.5

    def validate(self, strict: bool = False) -> "ChronosPriors":
        """Enforce priors contract invariants.  Returns ``self`` for chaining.

        Invariants:
            1. All numeric fields must be finite (always fatal).
            2. ``q10 ≤ q50 ≤ q90`` — strict raises; else auto-sort.
            3. ``dispersion ≥ 0`` — strict raises; else recompute.
            4. ``prob_up ∈ [0, 1]`` — strict raises; else clamp.
        """
        # 1. Finite check (always fatal — garbage-in must not propagate)
        for name in ("drift", "vol_forecast", "tail_risk", "q10", "q50", "q90",
                     "dispersion", "prob_up"):
            v = getattr(self, name)
            if not math.isfinite(v):
                raise ValueError(f"ChronosPriors.{name} is not finite: {v}")

        # 2. Monotonic quantiles
        if not (self.q10 <= self.q50 <= self.q90):
            if strict:
                raise ValueError(
                    f"Quantile monotonicity violated: "
                    f"q10={self.q10}, q50={self.q50}, q90={self.q90}"
                )
            sq = sorted([self.q10, self.q50, self.q90])
            self.q10, self.q50, self.q90 = sq[0], sq[1], sq[2]
            _warn_once(
                "monotonic",
                f"Quantile monotonicity violated; auto-sorted to "
                f"q10={self.q10:.4f}, q50={self.q50:.4f}, q90={self.q90:.4f}",
            )

        # 3. Non-negative dispersion
        if self.dispersion < 0:
            if strict:
                raise ValueError(f"Negative dispersion: {self.dispersion}")
            self.dispersion = self.q90 - self.q10

        # 4. prob_up bounds
        if not (0.0 <= self.prob_up <= 1.0):
            if strict:
                raise ValueError(f"prob_up out of [0,1]: {self.prob_up}")
            self.prob_up = max(0.0, min(1.0, self.prob_up))
            self.trend_conf = self.prob_up

        return self


def infer_priors(
    model: nn.Module,
    input_tensor: torch.Tensor,
    horizon_days: int = 20,
    n_samples: int = 20,
    codec: Optional[Any] = None,
    past_covariates: Optional[torch.Tensor] = None,
    future_covariates: Optional[torch.Tensor] = None,
    *,
    mode: Literal["auto", "native_nll", "quantile_head"] = "auto",
    strict: bool = False,
) -> List[ChronosPriors]:
    """Generate distributional priors for a batch of tickers.

    Parameters
    ----------
    model : nn.Module
        A ``Chronos2NativeWrapper`` (for native NLL or quantile-head) or
        a token-based Chronos model requiring ``codec``.
    input_tensor : torch.Tensor
        Price context of shape ``[B, T, F]`` or ``[B, T]``.
    horizon_days : int
        Forecast horizon in trading days.
    n_samples : int
        Number of sample paths.  **Ignored in ``native_nll`` mode** (pipeline
        is deterministic); a warning is emitted if ``n_samples != 1``.
    codec : optional
        Token codec for legacy Chronos-1 models.
    past_covariates, future_covariates : optional
        **Ignored in ``native_nll`` mode** (``amazon/chronos-2`` pipeline does
        not support covariates); a warning is emitted.
    mode : {"auto", "native_nll", "quantile_head"}
        Explicit inference mode.  ``"auto"`` resolves from model type.
    strict : bool
        If True, raise on unsupported kwargs and validation failures
        instead of warning/auto-fixing.

    Returns
    -------
    List[ChronosPriors]
        One entry per batch element with validated fields::

            drift, vol_forecast, tail_risk, trend_conf,
            q10, q50, q90, dispersion, prob_up, metadata

    Examples
    --------
    Teacher-10d (ranking priors)::

        priors = infer_priors(model_10d, price_tensor, horizon_days=10)

    Teacher-30d (regime priors)::

        priors = infer_priors(model_30d, price_tensor, horizon_days=30)
    """
    # --- Resolve inference mode -------------------------------------------
    if mode == "auto":
        if isinstance(model, Chronos2NativeWrapper):
            if getattr(model, "_enable_q10d_head", False):
                resolved_mode = "quantile_head"
            else:
                resolved_mode = "native_nll"
        else:
            resolved_mode = "token_codec"
    else:
        resolved_mode = mode

    # --- Mode-specific guards ---------------------------------------------
    if resolved_mode == "native_nll":
        if n_samples != 1:
            msg = (
                "native_nll inference is deterministic; n_samples is ignored. "
                "Pass n_samples=1 to silence this warning."
            )
            if strict:
                raise ValueError(msg)
            _warn_once("n_samples", msg)
        if past_covariates is not None or future_covariates is not None:
            msg = (
                "amazon/chronos-2 pipeline does not support covariates; "
                "they will be ignored."
            )
            if strict:
                raise ValueError(msg)
            _warn_once("covariates", msg)
            past_covariates = None
            future_covariates = None

    # --- Ensure [B, T, F] -------------------------------------------------
    if input_tensor.ndim == 2:
        input_tensor = input_tensor.unsqueeze(-1)

    # ===== Quantile-head fast path ========================================
    if (
        resolved_mode == "quantile_head"
        and isinstance(model, Chronos2NativeWrapper)
        and hasattr(model, "predict_quantiles_10d")
        and getattr(model, "_enable_q10d_head", False)
    ):
        quantiles = _get_standard_quantile_levels(21)
        q_10d = model.predict_quantiles_10d(
            input_tensor,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
        )
        idx_50 = quantiles.index(0.5)
        idx_10 = quantiles.index(0.1)
        idx_90 = quantiles.index(0.9)

        def _prob_up_qh(values: torch.Tensor) -> float:
            if torch.all(values > 0):
                return 1.0
            if torch.all(values < 0):
                return 0.0
            for i in range(len(values) - 1):
                if values[i] <= 0 <= values[i + 1]:
                    q_lo, q_hi = quantiles[i], quantiles[i + 1]
                    denom = values[i + 1] - values[i]
                    if denom == 0:
                        return 1 - q_hi
                    frac = (0 - values[i]) / denom
                    return 1 - (q_lo + frac * (q_hi - q_lo))
            return 0.5

        results = []
        for b in range(q_10d.shape[0]):
            _q10 = float(q_10d[b, idx_10])
            _q50 = float(q_10d[b, idx_50])
            _q90 = float(q_10d[b, idx_90])
            _pu = _prob_up_qh(q_10d[b])
            p = ChronosPriors(
                drift=_q50,
                vol_forecast=_q90 - _q10,
                tail_risk=_q10,
                trend_conf=_pu,
                metadata={"quantiles": quantiles, "horizon": 10, "mode": resolved_mode},
                q10=_q10,
                q50=_q50,
                q90=_q90,
                dispersion=_q90 - _q10,
                prob_up=_pu,
            )
            p.validate(strict=strict)
            results.append(p)
        return results

    # ===== Sample-path approach ===========================================
    if isinstance(model, Chronos2NativeWrapper):
        values = model.generate(
            input_tensor,
            prediction_length=horizon_days,
            num_samples=n_samples,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
        )
    else:
        if codec is None:
            raise ValueError("Codec required for token-based Chronos models.")
        encoded = codec.encode(input_tensor)
        token_preds = model.generate(
            encoded["input_ids"], encoded["attention_mask"],
            prediction_length=horizon_days, num_samples=n_samples,
        )
        B, N, P, F = token_preds.shape
        flat_tokens = token_preds.reshape(B * N, P, F)
        flat_scale = encoded["scale"].unsqueeze(1).expand(B, N, 1, F).reshape(
            B * N, 1, F
        )
        values = codec.decode(flat_tokens, flat_scale).reshape(B, N, P, F)

    # --- Compute end-of-horizon cumulative returns ------------------------
    price_paths = values[:, :, :, 0]
    current_price = input_tensor[:, -1, 0].unsqueeze(1)
    cum_returns = (price_paths / current_price.unsqueeze(2)) - 1.0
    terminal_returns = cum_returns[:, :, -1]

    results = []
    for b in range(terminal_returns.shape[0]):
        tr = terminal_returns[b]
        _q10 = float(torch.quantile(tr, 0.10))
        _q50 = float(torch.median(tr))
        _q90 = float(torch.quantile(tr, 0.90))
        _pu = float((tr > 0).float().mean())
        _vol = float(torch.std(tr)) if tr.numel() > 1 else (_q90 - _q10)
        p = ChronosPriors(
            drift=_q50,
            vol_forecast=_vol,
            tail_risk=_q10,
            trend_conf=_pu,
            metadata={"horizon": horizon_days, "mode": resolved_mode},
            q10=_q10,
            q50=_q50,
            q90=_q90,
            dispersion=_q90 - _q10,
            prob_up=_pu,
        )
        p.validate(strict=strict)
        results.append(p)
    return results
