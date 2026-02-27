from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BaseSchema:
    name: str
    required_fields: tuple[str, ...]
    normalization_rules: dict[str, str] = field(default_factory=dict)
    window_semantics: str = "rolling"
    missing_value_policy: str = "impute_zero"
    stable_ordering: str = "lexicographic_symbol"


TSFMSeriesSchema = BaseSchema(
    name="TSFMSeriesSchema",
    required_fields=("timestamp", "symbol", "target", "context"),
    window_semantics="context+prediction",
    missing_value_policy="ffill_then_drop",
)

CrossSectionalSchema = BaseSchema(
    name="CrossSectionalSchema",
    required_fields=("asof", "symbol", "features", "label"),
    stable_ordering="universe_then_symbol",
)

VolSurfaceSchema = BaseSchema(
    name="VolSurfaceSchema",
    required_fields=("timestamp", "underlier", "dte_bucket", "moneyness_bucket", "iv"),
    missing_value_policy="grid_interpolate",
)

MultivariatePanelSchema = BaseSchema(
    name="MultivariatePanelSchema",
    required_fields=("timestamp", "asset_index", "features"),
    stable_ordering="fixed_asset_index",
)

RLEnvSchema = BaseSchema(
    name="RLEnvSchema",
    required_fields=("timestamp", "state", "action", "reward", "done"),
    missing_value_policy="forbid",
)
