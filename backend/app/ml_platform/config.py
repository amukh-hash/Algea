from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os
import warnings

from .utils.downsample import parse_duration_to_timedelta


def _resolve_tsfm_downsample_freq() -> str:
    # New canonical variable
    dur = os.getenv("TSFM_DOWNSAMPLE_FREQ_DURATION")
    # Deprecated variable retained temporarily for migration
    legacy = os.getenv("TSFM_DOWNSAMPLE_FREQ")
    if dur is None and legacy is not None:
        warnings.warn(
            "TSFM_DOWNSAMPLE_FREQ is deprecated; use TSFM_DOWNSAMPLE_FREQ_DURATION with duration strings like '5min'",
            DeprecationWarning,
            stacklevel=2,
        )
        dur = legacy
    if dur is None:
        dur = "1min"
    # Validate eagerly so config construction fails fast for bad values
    parse_duration_to_timedelta(dur)
    return dur


@dataclass(frozen=True)
class MLPlatformConfig:
    train_device: str = field(default_factory=lambda: os.getenv("TRAIN_DEVICE", "cuda:0"))
    infer_device: str = field(default_factory=lambda: os.getenv("INFER_DEVICE", "cuda:1"))
    model_root: Path = field(default_factory=lambda: Path(os.getenv("MODEL_ARTIFACT_ROOT", "backend/artifacts/models")))
    registry_db_path: Path = field(
        default_factory=lambda: Path(os.getenv("MODEL_REGISTRY_DB", "backend/artifacts/model_registry.sqlite"))
    )
    trace_root: Path = field(default_factory=lambda: Path(os.getenv("TRACE_ROOT", "backend/artifacts/traces")))
    hf_cache_root: Path = field(default_factory=lambda: Path(os.getenv("HF_CACHE_ROOT", "backend/artifacts/hf_cache")))
    min_train_vram_gb: float = field(default_factory=lambda: float(os.getenv("MIN_TRAIN_VRAM_GB", "8")))
    min_infer_vram_gb: float = field(default_factory=lambda: float(os.getenv("MIN_INFER_VRAM_GB", "6")))
    selector_model_alias: str = field(default_factory=lambda: os.getenv("SELECTOR_MODEL_ALIAS", "prod"))
    vrp_model_alias: str = field(default_factory=lambda: os.getenv("VRP_MODEL_ALIAS", "prod"))
    itransformer_model_alias: str = field(default_factory=lambda: os.getenv("ITRANSFORMER_MODEL_ALIAS", "prod"))
    rl_policy_alias_vrp: str = field(default_factory=lambda: os.getenv("RL_POLICY_ALIAS_VRP", "prod"))
    rl_policy_alias_statarb: str = field(default_factory=lambda: os.getenv("RL_POLICY_ALIAS_STATARB", "prod"))
    enable_rl_overlay_vrp: bool = field(default_factory=lambda: os.getenv("ENABLE_RL_OVERLAY_VRP", "1") == "1")
    enable_rl_overlay_statarb: bool = field(default_factory=lambda: os.getenv("ENABLE_RL_OVERLAY_STATARB", "1") == "1")
    rl_fail_mode: str = field(default_factory=lambda: os.getenv("RL_FAIL_MODE", "halt"))
    tsfm_downsample_freq: str = field(default_factory=_resolve_tsfm_downsample_freq)
    sla_chronos2_ms: int = field(default_factory=lambda: int(os.getenv("SLA_CHRONOS2_MS", "500")))
    sla_smoe_rank_ms: int = field(default_factory=lambda: int(os.getenv("SLA_SMOE_RANK_MS", "500")))
    sla_vol_surface_forecast_ms: int = field(default_factory=lambda: int(os.getenv("SLA_VOL_SURFACE_FORECAST_MS", "500")))
    sla_vol_surface_grid_forecast_ms: int = field(default_factory=lambda: int(os.getenv("SLA_VOL_SURFACE_GRID_FORECAST_MS", "500")))
    sla_itransformer_signal_ms: int = field(default_factory=lambda: int(os.getenv("SLA_ITRANSFORMER_SIGNAL_MS", "500")))
    sla_rl_policy_act_ms: int = field(default_factory=lambda: int(os.getenv("SLA_RL_POLICY_ACT_MS", "200")))

    def inference_endpoint_timeouts_ms(self) -> dict[str, int]:
        return {
            "chronos2_forecast": self.sla_chronos2_ms,
            "smoe_rank": self.sla_smoe_rank_ms,
            "vol_surface_forecast": self.sla_vol_surface_forecast_ms,
            "vol_surface_grid_forecast": self.sla_vol_surface_grid_forecast_ms,
            "itransformer_signal": self.sla_itransformer_signal_ms,
            "rl_policy_act": self.sla_rl_policy_act_ms,
        }
