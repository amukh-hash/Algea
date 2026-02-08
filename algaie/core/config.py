from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("PyYAML is required to load YAML config files") from exc
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@dataclass(frozen=True)
class BacktestConfig:
    start: Optional[str] = None
    end: Optional[str] = None
    fill_price_mode: str = "next_open"
    rounding_policy: str = "fractional"
    slippage_model: str = "none"
    slippage_bps: float = 0.0
    slippage_volume_impact: float = 0.0
    commission_per_trade: float = 0.0
    commission_per_share: float = 0.0
    commission_bps: float = 0.0
    commission_min: float = 0.0
    walk_forward: bool = False
    train_window_days: int = 504
    test_window_days: int = 126
    step_days: int = 126
    holdout_pct: float = 0.1
    expanding_window: bool = True


@dataclass(frozen=True)
class PortfolioConfig:
    top_k: int = 50
    weight_method: str = "softmax"
    softmax_temp: float = 1.0
    max_weight_per_name: float = 0.1
    max_names: int = 50
    min_dollar_position: float = 0.0
    cash_buffer_pct: float = 0.05
    exit_policy: str = "hybrid"
    hold_days: Optional[int] = 10
    hold_days_min: Optional[int] = None
    hold_days_max: Optional[int] = None


@dataclass(frozen=True)
class BrokerConfig:
    mode: str = "paper"
    dry_run: bool = False
    max_orders_per_day: int = 200
    max_notional_per_order: float = 100000.0
    fractional_policy: str = "round"
    rounding_policy: str = "round"
    alpaca_base_url: Optional[str] = None
    ibkr_gateway_url: Optional[str] = None


@dataclass(frozen=True)
class PipelineConfig:
    artifact_root: Path = Path("backend/artifacts")
    max_invalid_frac: float = 0.001
    enable_quantiles: bool = False
    run_id: Optional[str] = None
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    broker: BrokerConfig = field(default_factory=BrokerConfig)
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Dict[str, Any], *, strict: bool = False) -> "PipelineConfig":
        data = dict(payload)
        artifact_root = Path(data.pop("artifact_root", cls.artifact_root))
        max_invalid_frac = float(data.pop("max_invalid_frac", cls.max_invalid_frac))
        enable_quantiles = bool(data.pop("enable_quantiles", cls.enable_quantiles))
        run_id = data.pop("run_id", None)
        backtest_payload = data.pop("backtest", {})
        portfolio_payload = data.pop("portfolio", {})
        broker_payload = data.pop("broker", {})
        if strict and data:
            raise ValueError(f"Unknown config keys: {sorted(data.keys())}")
        return cls(
            artifact_root=artifact_root,
            max_invalid_frac=max_invalid_frac,
            enable_quantiles=enable_quantiles,
            run_id=run_id,
            backtest=BacktestConfig(**backtest_payload),
            portfolio=PortfolioConfig(**portfolio_payload),
            broker=BrokerConfig(**broker_payload),
            extra=data,
        )


def load_config(path: Path | str, *, strict: bool = False) -> PipelineConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    suffix = config_path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        payload = _load_yaml(config_path)
    elif suffix == ".json":
        payload = _load_json(config_path)
    else:
        raise ValueError(f"Unsupported config extension: {suffix}")
    return PipelineConfig.from_mapping(payload, strict=strict)


def ensure_run_id(config: PipelineConfig) -> str:
    if config.run_id:
        return config.run_id
    return datetime.utcnow().strftime("RUN-%Y%m%d-%H%M%S")
