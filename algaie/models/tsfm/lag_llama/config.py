"""
Lag-Llama configuration — all tuneable parameters for the risk forecaster.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class LagLlamaConfig:
    """Immutable configuration for the Lag-Llama risk forecaster."""

    # --- Mode ---
    mode: str = "zero_shot"                            # "zero_shot" | "finetune"
    model_id: str = "time-series-foundation-models/Lag-Llama"

    # --- Model architecture ---
    context_length: int = 256
    prediction_length: int = 10
    quantiles: Tuple[float, ...] = (0.50, 0.90, 0.95, 0.99)

    # --- Series targets ---
    series_types: Tuple[str, ...] = ("sqret",)         # "sqret", "abs_neg_ret"
    upper_clip_percentile: float = 99.5                # clip extreme squared returns

    # --- Data requirements ---
    min_history_days: int = 400

    # --- Reproducibility ---
    device: str = "cuda"
    seed: int = 42
    num_samples: int = 100                             # for quantile estimation

    # --- Hardening ---
    rv_clamp_max: float = 1.50                         # 150% annualised
    rv_clamp_min: float = 0.02                         # floor
    baseline_blend_weight: float = 0.30                # default blend with EWMA
    calibration_coverage_min: float = 0.80             # min quantile coverage

    # --- Splits (for validation) ---
    train_pct: float = 0.70
    val_pct: float = 0.15
    test_pct: float = 0.15

    # --- EWMA baseline ---
    ewma_span: int = 20                                # span for baseline RV estimator

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LagLlamaConfig":
        for k in ("quantiles", "series_types"):
            if k in d and isinstance(d[k], list):
                d[k] = tuple(d[k])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
