from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field, fields as dc_fields
from typing import Any, Dict, Literal, Sequence, Type, TypeVar

logger = logging.getLogger(__name__)

AnchorMethod = Literal["MID", "VWAP_WINDOW", "BID_ASK_EXEC"]

_T = TypeVar("_T")


def _from_dict_helper(cls: Type[_T], d: Dict[str, Any], *, warn_unknown: bool = True) -> _T:
    """Build a frozen dataclass from a dict, preserving defaults for missing keys.

    Unknown keys are rejected with a warning (or silently ignored if
    warn_unknown=False).  This is the ONLY approved way to convert dicts
    to config dataclasses inside the codebase.
    """
    valid_fields = {f.name for f in dc_fields(cls)}
    unknown = set(d.keys()) - valid_fields
    if unknown and warn_unknown:
        logger.warning("%s.from_dict: ignoring unknown keys %s", cls.__name__, sorted(unknown))
    return cls(**{k: v for k, v in d.items() if k in valid_fields})

_FULL_UNIVERSE = (
    "ES", "NQ", "YM", "RTY",
    "CL", "GC", "SI", "ZN", "ZB",
    "6E", "6J", "HG", "6B", "6A",
)


@dataclass(frozen=True)
class CostConfig:
    commission_per_contract: float = 2.0
    impact_k: float = 0.05
    use_impact_proxy: bool = True

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CostConfig":
        return _from_dict_helper(cls, d)


@dataclass(frozen=True)
class ShockConfig:
    """Execution-time shock gating parameters."""
    enabled: bool = True
    shock_z_threshold: float = 2.0
    gross_multiplier_on_shock: float = 0.5  # 0.0 = block entry entirely
    per_instrument: bool = True             # apply per-token if True

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ShockConfig":
        return _from_dict_helper(cls, d)


@dataclass(frozen=True)
class ModelConfig:
    backend: str = "chronos2_prior+cross_asset_transformer_residual"
    estimator_type: str = "Ridge"  # "Ridge" or "CSTransformer"
    two_head: bool = True           # two-head model (score + risk)
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 3
    dropout: float = 0.1
    lr: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 60
    batch_size: int = 32
    early_stop_patience: int = 10
    min_panel_N: int = 8            # reject training days with fewer instruments
    seeds: tuple = (11, 22, 33)     # multi-seed training
    data_provider: str = "yfinance" # "yfinance" (research) or "ibkr_hist" (promotion)
    # risk target
    risk_target_transform: str = "log_abs"  # "log_abs" or "raw_abs"
    risk_target_eps: float = 1e-6
    # derived score stabilizers
    risk_pred_clamp_min: float = 0.05
    risk_pred_clamp_max: float = 5.0
    score_tanh: bool = False
    derived_score_clip: float = 10.0

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelConfig":
        return _from_dict_helper(cls, d)


@dataclass(frozen=True)
class CVConfig:
    fold_size_days: int = 40
    embargo_days: int = 2

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CVConfig":
        return _from_dict_helper(cls, d)


@dataclass(frozen=True)
class LossConfig:
    """Loss function configuration for RankingMultiTaskLoss."""
    rank_loss_weight: float = 1.0
    reg_loss_weight: float = 0.1
    risk_loss_weight: float = 0.5
    collapse_weight: float = 0.1
    temp_y: float = 1.0
    temp_pred: float = 1.0
    risk_df: float = 5.0
    sigma_floor: float = 1e-4
    collapse_var_threshold: float = 1e-4

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LossConfig":
        return _from_dict_helper(cls, d)


@dataclass(frozen=True)
class TrainingConfig:
    """Training hyperparameters (separate from model architecture)."""
    seeds: tuple = (11, 22, 33, 44, 55, 66, 77)
    cv_fold_days: int = 40
    embargo_days: int = 2
    oos_months: int = 6

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainingConfig":
        if "seeds" in d and isinstance(d["seeds"], list):
            d = {**d, "seeds": tuple(d["seeds"])}
        return _from_dict_helper(cls, d)


@dataclass(frozen=True)
class PromotionConfig:
    """Promotion gating parameters."""
    require_ibkr: bool = True
    min_sharpe_delta: float = 0.10
    max_drawdown_tolerance: float = -0.10
    worst_1pct_tolerance_bps: float = 25.0
    min_hit_rate: float = 0.45
    stress_required: int = 1
    windows: tuple = ()  # tuple of {"name": str, "start": str, "end": str}
    # F5 additions
    ic_tail_floor: float = -0.05
    catastrophic_sharpe_floor: float = -0.50
    catastrophic_dd_worse_than_baseline: float = -0.20
    tier_for_gates: str = "TIER2"
    # G2 additions
    catastrophic_delta_floor: float = -0.25
    max_delta_cv: float = 0.30
    sharpe_delta_min_oos: float = 0.05
    worst_1pct_tolerance: float = 0.02

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PromotionConfig":
        if "windows" in d and isinstance(d["windows"], list):
            d = {**d, "windows": tuple(d["windows"])}
        return _from_dict_helper(cls, d)


@dataclass(frozen=True)
class COOCReversalConfig:
    universe: Sequence[str] = _FULL_UNIVERSE
    include_micros: bool = False
    anchor_method_open: AnchorMethod = "MID"
    anchor_method_close: AnchorMethod = "MID"
    lookback: int = 20
    micro_window_minutes: int = 5
    gross_target: float = 0.8
    caution_scale: float = 0.5
    max_contracts_per_instrument: int = 20
    max_weight_per_instrument: float = 0.35
    net_cap: float = 0.05
    price_limit_policy: Literal["block_entry", "force_flatten", "none"] = "block_entry"
    target_vol: float | None = None
    enforce_market_neutral: bool = True
    entry_time_et: str = "09:30:00"
    close_time_et: str = "16:00:00"
    emergency_flatten_time_et: str = "16:10:00"
    # Data coverage
    min_instruments_per_day: int = 12
    warmup_days: int = 252
    # Feature schema (2 = V2 default, 3 = V3 opt-in)
    schema_version: int = 2
    # Contract spec mismatch tolerance
    allow_contract_spec_mismatch: bool = False
    # Sub-configs
    costs: CostConfig = field(default_factory=CostConfig)
    shock: ShockConfig = field(default_factory=ShockConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    cv: CVConfig = field(default_factory=CVConfig)
    promotion: PromotionConfig = field(default_factory=PromotionConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "COOCReversalConfig":
        sub_map = {
            "costs": CostConfig, "shock": ShockConfig,
            "model": ModelConfig, "cv": CVConfig,
            "promotion": PromotionConfig, "loss": LossConfig,
            "training": TrainingConfig,
        }
        kwargs: Dict[str, Any] = {}
        for k, v in d.items():
            if k in sub_map and isinstance(v, dict):
                kwargs[k] = sub_map[k].from_dict(v)
            elif k in {f.name for f in dc_fields(cls)}:
                kwargs[k] = v
            else:
                logger.warning("COOCReversalConfig.from_dict: ignoring key %r", k)
        if "universe" in kwargs and isinstance(kwargs["universe"], list):
            kwargs["universe"] = tuple(kwargs["universe"])
        return cls(**kwargs)


YAML_SCHEMA = {
    "type": "object",
    "required": ["universe", "gross_target"],
    "properties": {
        "universe": {"type": "array", "items": {"type": "string"}},
        "anchor_method_open": {"enum": ["MID", "VWAP_WINDOW", "BID_ASK_EXEC"]},
        "anchor_method_close": {"enum": ["MID", "VWAP_WINDOW", "BID_ASK_EXEC"]},
        "lookback": {"type": "integer", "minimum": 2},
        "gross_target": {"type": "number", "minimum": 0.0, "maximum": 2.0},
        "caution_scale": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
}


def load_config_from_yaml(path: str) -> COOCReversalConfig:
    """Load a COOCReversalConfig from a YAML file.

    Handles nested sub-configs (costs, shock, model, cv) and falls back to
    dataclass defaults for any missing key.  The ``data`` section is flattened
    into the top-level config (``universe``, ``min_instruments_per_day``, etc.).
    """
    import yaml
    from pathlib import Path as _P

    raw = yaml.safe_load(_P(path).read_text(encoding="utf-8"))
    if raw is None:
        return COOCReversalConfig()

    # Flatten nested sections
    data = raw.pop("data", {})
    features = raw.pop("features", {})
    execution = raw.pop("execution", {})
    model_raw = raw.pop("model", {})
    cv_raw = raw.pop("cv", {})

    # Build sub-configs
    costs_raw = execution.pop("costs", raw.pop("costs", {}))
    shock_raw = execution.pop("shock", raw.pop("shock", {}))
    promotion_raw = raw.pop("promotion", {})

    costs = CostConfig(**{k: v for k, v in costs_raw.items() if k in CostConfig.__dataclass_fields__}) if costs_raw else CostConfig()
    shock = ShockConfig(**{k: v for k, v in shock_raw.items() if k in ShockConfig.__dataclass_fields__}) if shock_raw else ShockConfig()
    model = ModelConfig(**{k: v for k, v in model_raw.items() if k in ModelConfig.__dataclass_fields__}) if model_raw else ModelConfig()
    cv = CVConfig(**{k: v for k, v in cv_raw.items() if k in CVConfig.__dataclass_fields__}) if cv_raw else CVConfig()

    # Build promotion config
    promo_kwargs = {}
    for k in ("require_ibkr", "min_sharpe_delta", "max_drawdown_tolerance",
              "worst_1pct_tolerance_bps", "min_hit_rate", "stress_required"):
        if k in promotion_raw:
            promo_kwargs[k] = promotion_raw[k]
    if "promotion_windows" in promotion_raw:
        promo_kwargs["windows"] = tuple(
            {"name": w.get("name", f"window_{i}"), "start": str(w["start"]), "end": str(w["end"])}
            for i, w in enumerate(promotion_raw["promotion_windows"])
        )
    promotion = PromotionConfig(**promo_kwargs) if promo_kwargs else PromotionConfig()

    # Merge data/features/execution sections into top-level kwargs
    top_kwargs: dict = {}

    # From data section
    if "universe" in data:
        top_kwargs["universe"] = tuple(data["universe"])
    for k in ("min_instruments_per_day", "warmup_days"):
        if k in data:
            top_kwargs[k] = data[k]

    # From features section
    if "schema_version" in features:
        top_kwargs["schema_version"] = features["schema_version"]

    # From execution section (remaining non-nested keys)
    for k in ("gross_target", "caution_scale", "max_contracts_per_instrument",
              "max_weight_per_instrument", "net_cap", "target_vol",
              "enforce_market_neutral", "entry_time_et", "close_time_et",
              "emergency_flatten_time_et", "price_limit_policy"):
        if k in execution:
            top_kwargs[k] = execution[k]

    # From top-level raw keys
    for k in COOCReversalConfig.__dataclass_fields__:
        if k in raw and k not in ("costs", "shock", "model", "cv"):
            top_kwargs[k] = raw[k]

    # Convert universe to tuple if needed
    if "universe" in top_kwargs and isinstance(top_kwargs["universe"], list):
        top_kwargs["universe"] = tuple(top_kwargs["universe"])

    return COOCReversalConfig(
        costs=costs, shock=shock, model=model, cv=cv,
        promotion=promotion,
        **top_kwargs,
    )
