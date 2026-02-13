from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

AnchorMethod = Literal["MID", "VWAP_WINDOW", "BID_ASK_EXEC"]


@dataclass(frozen=True)
class CostConfig:
    commission_per_contract: float = 2.0
    impact_k: float = 0.05
    use_impact_proxy: bool = True


@dataclass(frozen=True)
class ModelConfig:
    backend: str = "chronos2_prior+cross_asset_transformer_residual"
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1
    lr: float = 1e-3
    epochs: int = 5
    batch_size: int = 32


@dataclass(frozen=True)
class CVConfig:
    fold_size_days: int = 40
    embargo_days: int = 2


@dataclass(frozen=True)
class COOCReversalConfig:
    universe: Sequence[str] = ("ES", "NQ", "YM", "RTY")
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
    costs: CostConfig = field(default_factory=CostConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    cv: CVConfig = field(default_factory=CVConfig)


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
