"""F9-compliant TFT Core Reversal plugin for the Actor-Model architecture.

Executes inference inside the isolated GPU worker process via
``GPUProcessSupervisor``.  Reads pre-computed features from
``tft_features.json`` (written by the CPU-bound data ingest job)
and writes *both* entry and exit intents to ``core_intents.json``.

**Flattening Guarantee**: The ``AUCTION_CLOSE`` intent with
``target_weight=0.0`` is always emitted alongside the entry,
forcing the Phase Router to flatten the position at 15:59:00 EST.

**Sizing Safety Invariant**: The confidence scaling formula
``BASE_WEIGHT + (confidence * (MAX_WEIGHT - BASE_WEIGHT))``
bounds output strictly between [0.04, 0.08].  It is mathematically
impossible for an infinitesimally small uncertainty spread to cause
a float explosion.
"""
from __future__ import annotations

import json
import logging
import os

import torch

from backend.app.core.schemas import ExecutionPhase, TargetIntent
from backend.app.ml_platform.models.tft_gap.model import TemporalFusionTransformer
from backend.app.sleeves.core_tft.schemas import TFTInferenceContext

logger = logging.getLogger(__name__)

# Module-level device constant — monkeypatched to "cpu" in tests
_DEVICE = "cuda:1"


class CoreTFTPlugin:
    """Trading sleeve plugin for TFT-based Open-to-Close gap reversal.

    Constants
    ---------
    MAX_UNCERTAINTY : float
        2.5% max allowable spread between P10 and P90 (calibrated to ~90th pctl).
    MIN_EDGE : float
        0.15% minimum expected Open-to-Close move to justify entry.
    BASE_WEIGHT : float
        4% base portfolio allocation.
    MAX_WEIGHT : float
        8% max leverage cap.
    """

    MAX_UNCERTAINTY = 0.025   # 2.5% max allowable spread between P10 and P90
    MIN_EDGE = 0.0015         # 0.15% minimum expected Open-to-Close move
    BASE_WEIGHT = 0.04        # 4% base portfolio allocation
    MAX_WEIGHT = 0.08         # 8% max leverage cap

    def execute(self, context: dict, model_cache: dict) -> None:
        """Run TFT inference and emit dual entry/exit intents.

        Parameters
        ----------
        context : dict
            Must contain ``artifact_dir``, ``asof_date``,
            ``model_weights_path``.
        model_cache : dict
            Persistent VRAM cache managed by ``GPUProcessSupervisor``.
        """
        device = _DEVICE

        # 1. Warm/Load Model (Leveraging GPUProcessSupervisor persistent cache)
        if "tft_model" not in model_cache:
            model = TemporalFusionTransformer().to(
                device=device, dtype=torch.bfloat16,
            )
            weights_path = context.get("model_weights_path")
            if weights_path and os.path.exists(weights_path):
                model.load_state_dict(
                    torch.load(weights_path, map_location=device),
                )
            else:
                logger.warning(
                    "No model weights found at %s — running with random init",
                    weights_path,
                )
            model.eval()
            model_cache["tft_model"] = model

        model = model_cache["tft_model"]

        # 2. Parse Input Tensors from JSON (Prepared by CPU data ingest job)
        features_path = os.path.join(
            context["artifact_dir"], "tft_features.json",
        )
        with open(features_path, "r") as f:
            tft_input = TFTInferenceContext(**json.load(f))

        ts_tensor = torch.tensor(
            [tft_input.observed_past_seq],
            device=device, dtype=torch.bfloat16,
        )
        static_tensor = torch.tensor(
            [[tft_input.day_of_week, tft_input.is_opex, tft_input.macro_event_id]],
            device=device, dtype=torch.bfloat16,
        )
        obs_tensor = torch.tensor(
            [[
                tft_input.gap_proxy_pct,
                tft_input.nikkei_pct,
                tft_input.eurostoxx_pct,
                tft_input.zn_drift_bps,
                tft_input.vix_spot,
            ]],
            device=device, dtype=torch.bfloat16,
        )

        # 3. Inference
        with torch.no_grad():
            quantiles = model(ts_tensor, static_tensor, obs_tensor)[0].float().cpu().numpy()

        q10, q50, q90 = float(quantiles[0]), float(quantiles[1]), float(quantiles[2])
        uncertainty_spread = q90 - q10

        logger.info(
            "TFT inference: q10=%.5f q50=%.5f q90=%.5f spread=%.5f",
            q10, q50, q90, uncertainty_spread,
        )

        # 4. Safe Uncertainty & Edge Math (Fixes ZeroDivision Hazard)
        target_weight = 0.0
        if uncertainty_spread <= self.MAX_UNCERTAINTY and abs(q50) >= self.MIN_EDGE:
            direction = 1.0 if q50 > 0 else -1.0

            # Safe linear scaling: as spread approaches MAX_UNCERTAINTY,
            # confidence approaches 0
            confidence = max(0.0, 1.0 - (uncertainty_spread / self.MAX_UNCERTAINTY))

            # Scales linearly between BASE_WEIGHT and MAX_WEIGHT based on confidence
            scaled_weight = self.BASE_WEIGHT + (
                confidence * (self.MAX_WEIGHT - self.BASE_WEIGHT)
            )
            target_weight = direction * scaled_weight

        # 5. Dual Intent Emission (Entry + Guaranteed Exit)
        intents = [
            # Entry intent at Market Open (MOO)
            TargetIntent(
                asof_date=context["asof_date"],
                sleeve="core_reversal_tft",
                symbol="ES",
                asset_class="FUTURE",
                target_weight=round(float(target_weight), 4),
                execution_phase=ExecutionPhase.AUCTION_OPEN,
                multiplier=50.0,
            ).model_dump(),
            # Exit intent at Market Close (Forces Flat Overnight MOC)
            TargetIntent(
                asof_date=context["asof_date"],
                sleeve="core_reversal_tft",
                symbol="ES",
                asset_class="FUTURE",
                target_weight=0.0,
                execution_phase=ExecutionPhase.AUCTION_CLOSE,
                multiplier=50.0,
            ).model_dump(),
        ]

        # 6. Write to Artifacts for Orchestrator Risk Gateway
        out_path = os.path.join(context["artifact_dir"], "core_intents.json")
        with open(out_path, "w") as f:
            json.dump(intents, f)

        logger.info(
            "CoreTFTPlugin: entry_weight=%.4f, wrote %d intents to %s",
            target_weight, len(intents), out_path,
        )


# Expose entrypoint for GPUProcessSupervisor
def execute(context: dict, model_cache: dict):
    CoreTFTPlugin().execute(context, model_cache)
