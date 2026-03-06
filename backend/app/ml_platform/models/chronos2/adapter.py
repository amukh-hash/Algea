"""
Chronos2 Adapter — Production-grade inference bridge.

Replaces the deterministic linear-extrapolation stub with a fail-closed
adapter that calls the real ``Chronos2NativeWrapper`` from
``algae.models.foundation.chronos2_teacher``.

If no trained weights are available, all calls raise ``RuntimeError``
to prevent the Orchestrator DAG from routing garbage signals.
"""
from __future__ import annotations

import logging
from pathlib import Path
from statistics import pstdev
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# Re-export the original stub under an explicit name so existing CI tests
# can pin to it without reaching the real teacher.
_STUB_ACTIVE = True  # Flipped to False once a real model is loaded


def deterministic_quantile_forecast(
    series: list[float], prediction_length: int, quantiles: list[float],
) -> dict[str, list[float]]:
    """Legacy stub — linear extrapolation.

    .. deprecated::
        This function exists only for backward-compatibility with CI tests
        that exercise the Orchestrator DAG in ``noop`` mode.  **Production
        code must use** ``Chronos2Adapter.forecast()`` instead.
    """
    if not series:
        raise ValueError("series must be non-empty")
    last = float(series[-1])
    drift = (float(series[-1]) - float(series[0])) / max(len(series) - 1, 1)
    vol = pstdev(series) if len(series) > 1 else 0.0
    out: dict[str, list[float]] = {}
    for q in quantiles:
        q_key = f"{q:.2f}"
        spread = (q - 0.5) * 2.0 * max(vol, 1e-6)
        out[q_key] = [last + drift * (i + 1) + spread for i in range(prediction_length)]
    return out


def summarize_uncertainty(forecast: dict[str, list[float]]) -> dict[str, float]:
    q10 = forecast.get("0.10") or next(iter(forecast.values()))
    q90 = forecast.get("0.90") or next(reversed(forecast.values()))
    iqr = [abs(b - a) for a, b in zip(q10, q90)]
    if not iqr:
        return {"iqr_mean": 0.0, "iqr_max": 0.0}
    return {"iqr_mean": sum(iqr) / len(iqr), "iqr_max": max(iqr)}


# ═══════════════════════════════════════════════════════════════════════════
# Production Adapter — bridges to the real Chronos2 teacher
# ═══════════════════════════════════════════════════════════════════════════

class Chronos2Adapter:
    """Production-grade inference adapter for the Chronos2 foundation model.

    Loads the real ``Chronos2NativeWrapper`` from ``algae.models.foundation``
    and performs quantile forecasting on GPU.  Fails closed if weights are
    missing — the serving layer must never silently fall back to the stub.

    Parameters
    ----------
    model_weights_path : str or None
        Path to a ``.pt`` checkpoint or HuggingFace model directory.
        If ``None``, the adapter is inert and all calls raise ``RuntimeError``.
    device : str
        CUDA device string (default ``"cuda:1"`` for the 3090 Ti).
    context_length : int
        Number of historical observations the model expects.
    """

    def __init__(
        self,
        model_weights_path: Optional[str] = None,
        device: str = "cuda:1",
        context_length: int = 32,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.context_length = context_length
        self.is_loaded = False
        self._model = None

        if model_weights_path:
            self._load_model(model_weights_path)
        else:
            logger.warning(
                "Chronos2Adapter initialized WITHOUT trained weights.  "
                "Fail-closed enforced — all inference calls will raise RuntimeError."
            )

    def _load_model(self, path: str) -> None:
        """Load model weights.

        Supports two modes:
        1. HuggingFace model ID (e.g., ``"amazon/chronos-2"``) — loads base model only.
        2. Local ``.pt`` LoRA checkpoint from our Optuna training script — loads
           base model with LoRA (r=64, alpha=128) then injects the trained weights.
        """
        try:
            from algae.models.foundation.chronos2_teacher import load_chronos_adapter

            # Detect if path is a .pt file (trained LoRA weights) vs HF model_id
            pt_path = Path(path) if path.endswith(".pt") else None

            if pt_path and pt_path.exists():
                # ── Load base model WITH matching LoRA architecture ──
                wrapper, info = load_chronos_adapter(
                    model_id="amazon/chronos-2",
                    use_qlora=False,
                    device=self.device,
                    lora_config={
                        "rank": 64,
                        "alpha": 128,
                        "dropout": 0.05,
                        "target_modules": ["q", "k", "v", "o", "wi", "wo"],
                    },
                    eval_mode=True,
                )
                # ── Inject trained LoRA weights ──
                lora_state = torch.load(pt_path, map_location=self.device, weights_only=True)
                wrapper.load_state_dict(lora_state, strict=False)
                logger.info(
                    "Chronos2 LoRA adapter loaded from %s (%d keys, r=64)",
                    pt_path.name, len(lora_state),
                )
                self._model = wrapper
            else:
                # HuggingFace model ID path (base model only)
                wrapper, info = load_chronos_adapter(
                    model_id=path,
                    use_qlora=False,
                    device=self.device,
                    eval_mode=True,
                )
                self._model = wrapper

            self.is_loaded = True
            logger.info("Chronos2 model loaded  arch=%s", info.get("model_type"))
        except ImportError as e:
            logger.error("Cannot load Chronos2 — missing dependency: %s", e)
        except Exception as e:
            logger.error("Failed to load Chronos2 weights from %s: %s", path, e)

    @torch.inference_mode()
    def forecast(
        self,
        context_series: list[float],
        prediction_length: int = 3,
        quantiles: list[float] | None = None,
    ) -> dict[str, list[float]]:
        """Execute a deterministic quantile forecast.

        Parameters
        ----------
        context_series : list[float]
            Historical price/return observations.
        prediction_length : int
            Number of future steps to forecast.
        quantiles : list[float] or None
            Quantile levels to produce.  Defaults to ``[0.10, 0.50, 0.90]``.

        Returns
        -------
        dict[str, list[float]]
            ``{"0.10": [...], "0.50": [...], "0.90": [...]}``

        Raises
        ------
        RuntimeError
            If the model has no trained weights loaded.
        ValueError
            If the context series is empty or too short.
        """
        if not self.is_loaded or self._model is None:
            raise RuntimeError(
                "Chronos2 Service fail-closed: no trained weights loaded.  "
                "Set ENABLE_CHRONOS2_SLEEVE=0 or provide a valid model path."
            )

        if not context_series:
            raise ValueError("Input context_series cannot be empty.")
        if len(context_series) < self.context_length:
            raise ValueError(
                f"Context series too short: got {len(context_series)}, "
                f"need >= {self.context_length}"
            )

        if quantiles is None:
            quantiles = [0.10, 0.50, 0.90]

        # Build input tensor — [1, T, 1] for univariate
        series_tensor = torch.tensor(
            context_series[-self.context_length:],
            dtype=torch.float32,
        )
        context = series_tensor.unsqueeze(0).unsqueeze(-1)  # [1, T, 1]
        context = context.to(self.device, dtype=torch.bfloat16)

        # Forward pass via the real Chronos2 wrapper
        outputs = self._model.generate(
            context=context,
            prediction_length=prediction_length,
            num_samples=20,
        )
        # outputs shape: [B, S, P, F] — aggregate quantiles
        samples = outputs.squeeze(-1)  # [1, S, P]
        if samples.ndim == 2:
            samples = samples.unsqueeze(0)

        result: dict[str, list[float]] = {}
        for q in quantiles:
            q_key = f"{q:.2f}"
            q_vals = torch.quantile(samples.float(), q, dim=1)  # [B, P]
            result[q_key] = q_vals[0].cpu().tolist()

        return result
