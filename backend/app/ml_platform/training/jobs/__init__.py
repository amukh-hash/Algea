from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .chronos2_lora import TrainChronos2LoRAJob
from .smoe_ranker import TrainSMoERankerJob
from .vol_surface_forecaster import TrainVolSurfaceForecasterJob
from .itransformer import TrainITransformerJob
from .rl_policy import TrainRLPolicyJob


@dataclass
class TrainingJob:
    job_type: str
    model_name: str
    version: str
    params: dict[str, Any] = field(default_factory=dict)


def build_job(payload: dict[str, Any]) -> TrainingJob | TrainChronos2LoRAJob | TrainSMoERankerJob | TrainVolSurfaceForecasterJob | TrainITransformerJob | TrainRLPolicyJob:
    if payload.get("job_type") == "train_chronos2_lora":
        return TrainChronos2LoRAJob(**payload)
    if payload.get("job_type") == "train_smoe_ranker":
        return TrainSMoERankerJob(**payload)
    if payload.get("job_type") == "train_vol_surface_forecaster":
        return TrainVolSurfaceForecasterJob(**payload)
    if payload.get("job_type") == "train_itransformer":
        return TrainITransformerJob(**payload)
    if payload.get("job_type") == "train_rl_policy":
        return TrainRLPolicyJob(**payload)
    return TrainingJob(**payload)


__all__ = ["TrainingJob", "TrainChronos2LoRAJob", "TrainSMoERankerJob", "TrainVolSurfaceForecasterJob", "TrainITransformerJob", "TrainRLPolicyJob", "build_job"]
