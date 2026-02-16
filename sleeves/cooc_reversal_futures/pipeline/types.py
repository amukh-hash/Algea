"""Immutable dataclasses shared across the pipeline.

CHANGE LOG (2026-02-14):
  - D4: Added TradeProxyConfig with score_semantics, baseline_semantics, equity.
  - D5: Expanded TradeProxyReport with diagnostics fields.
"""
from __future__ import annotations

import json
from enum import IntEnum
from dataclasses import asdict, dataclass, field, fields
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple



# ---------------------------------------------------------------------------
# Realism tiers (F3)
# ---------------------------------------------------------------------------

class RealismTier(IntEnum):
    TIER0_ZERO_COST = 0
    TIER1_SIMPLE_COST = 1
    TIER2_SPREAD_IMPACT = 2


@dataclass(frozen=True)
class Tier2ImpactConfig:
    """Market-impact model for Tier2 realism."""
    base_bps: float = 0.5
    k: float = 0.1
    p: float = 0.5
    adv_window: int = 20
    min_adv_contracts: float = 100.0
    impact_cap_bps: float = 20.0
    downscale_on_cap: bool = True

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Tier2ImpactConfig":
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid})


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def _default_serializer(obj: object) -> Any:
    """Canonical JSON serializer for pipeline types."""
    if isinstance(obj, (datetime,)):
        return obj.isoformat()
    if isinstance(obj, date):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "to_dict"):
        return obj.to_dict()  # type: ignore[union-attr]
    raise TypeError(f"Cannot serialize {type(obj)}")


def _to_json(d: dict[str, Any]) -> str:
    return json.dumps(d, default=_default_serializer, sort_keys=True, indent=2)


# ---------------------------------------------------------------------------
# Bronze
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BronzeManifest:
    """Manifest produced by the bronze ingestion stage."""
    roots: Tuple[str, ...]
    paths: Dict[str, str]          # root -> parquet path
    checksums: Dict[str, str]      # root -> sha256 hex
    vendor: str
    retrieval_ts: str              # ISO-8601 UTC
    params: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BronzeValidationReport:
    """Result of bronze bar validation for a single root."""
    root: str
    ok: bool
    monotonic_ts: bool
    no_duplicates: bool
    ohlc_sane: bool
    non_negative_volume: bool
    gap_report: Tuple[Tuple[str, str, int], ...]  # (gap_start, gap_end, bdays)
    row_count: int
    violations: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["gap_report"] = [list(g) for g in self.gap_report]
        d["violations"] = list(self.violations)
        return d


# ---------------------------------------------------------------------------
# Canonicalization
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CanonicalizationManifest:
    """Manifest produced by the canonicalization stage."""
    roots: Tuple[str, ...]
    silver_path: str
    gold_path: str
    contract_map_path: str
    trading_days: int
    row_count: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetManifest:
    """Manifest produced by the dataset assembly stage."""
    dataset_path: str
    row_count: int
    feature_columns: Tuple[str, ...]
    label_column: str
    provenance_columns: Tuple[str, ...]
    data_version_hash: str
    code_version_hash: str
    config_hash: str
    dropped_features: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["feature_columns"] = list(self.feature_columns)
        d["provenance_columns"] = list(self.provenance_columns)
        d["dropped_features"] = list(self.dropped_features)
        return d


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SplitSpec:
    """Time-based split boundaries."""
    train_start: str              # ISO date
    train_end: str
    val_start: str
    val_end: str
    test_start: Optional[str]
    test_end: Optional[str]
    embargo_days: int
    fold_index: int = 0
    universe_snapshot: Dict[str, List[str]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelBundle:
    """Saved model bundle from training."""
    model_path: str
    feature_order: Tuple[str, ...]
    scaler_path: str
    nan_fill_values: Dict[str, float]
    chosen_params: Dict[str, Any]
    trial_log: Tuple[Dict[str, Any], ...]
    primary_metric: str
    primary_metric_value: float

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["feature_order"] = list(self.feature_order)
        d["trial_log"] = list(self.trial_log)
        return d


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GateResult:
    """Single validation gate result."""
    name: str
    passed: bool
    detail: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ValidationReport:
    """Result of all validation gates."""
    all_passed: bool
    gates: Tuple[GateResult, ...]
    baseline_ic: float
    model_ic: float

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "all_passed": self.all_passed,
            "gates": [g.to_dict() for g in self.gates],
            "baseline_ic": self.baseline_ic,
            "model_ic": self.model_ic,
        }
        return d


# ---------------------------------------------------------------------------
# Phase 1.5 — Alignment & Operational Readiness
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SessionSemanticsReport:
    """Result of yfinance-vs-IBKR session bar comparison."""
    per_field_stats: Dict[str, Dict[str, float]]
    gate_passed: bool
    sample_rows_path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FeatureParityReport:
    """Result of training-vs-runtime feature parity check."""
    per_feature_mismatch_rate: Dict[str, float]
    worst_offenders: Tuple[Dict[str, Any], ...]
    gate_passed: bool

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["worst_offenders"] = list(self.worst_offenders)
        return d


@dataclass(frozen=True)
class CoverageReport:
    """Result of cross-section panel coverage check."""
    days_total: int
    days_below_threshold: int
    min_roots_per_day: int
    gate_passed: bool
    histogram: Dict[int, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["histogram"] = {str(k): v for k, v in self.histogram.items()}
        return d


@dataclass(frozen=True)
class TradeProxyConfig:
    """Configuration for trade proxy evaluation.

    Controls score polarity, cost model, and portfolio sizing.
    """
    top_k: int = 1
    gross_target: float = 1.0
    proxy_equity_usd: float = 1_000_000.0
    slippage_bps_open: float = 1.0
    slippage_bps_close: float = 1.0
    cost_per_contract: float = 2.5
    fill_scale: float = 1.0
    shock_slippage_mult: float = 2.0
    shock_gross_mult: float = 0.5
    score_semantics: Literal["alpha_high_long", "alpha_low_long"] = "alpha_low_long"
    baseline_semantics: Literal["r_co_meanrevert", "r_co_momentum"] = "r_co_meanrevert"
    allow_insufficient_universe: bool = False
    require_not_worse_than_baseline: bool = True
    sharpe_tolerance: float = 0.05

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TradeProxyConfig":
        """Build from dict, preserving dataclass defaults for missing keys."""
        valid = {f.name for f in fields(cls)}
        unknown = set(d.keys()) - valid
        if unknown:
            import logging
            logging.getLogger(__name__).warning(
                "TradeProxyConfig.from_dict: ignoring unknown keys %s", sorted(unknown)
            )
        return cls(**{k: v for k, v in d.items() if k in valid})


@dataclass(frozen=True)
class TradeProxyReport:
    """Result of cost-aware trade-proxy evaluation."""
    sharpe_model: float
    sharpe_baseline: float
    hit_rate: float
    max_drawdown: float
    mean_daily_return: float
    worst_1pct_return: float
    gate_passed: bool
    # D5 diagnostics
    n_days: int = 0
    vol: float = 0.0
    skew: float = 0.0
    kurtosis: float = 0.0
    cvar_1pct: float = 0.0
    n_zero_return_days: int = 0
    n_insufficient_days: int = 0
    # R4: exposure gating diagnostics (None when not using gross_schedule)
    pct_crash_days: Optional[float] = None
    pct_caution_days: Optional[float] = None
    avg_gross_scale: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TradeProxyRealism:
    """Per-root cost/slippage config + shock/partial-fill realism."""
    cost_per_contract_by_root: Dict[str, float] = field(default_factory=dict)
    slippage_bps_open_by_root: Dict[str, float] = field(default_factory=dict)
    slippage_bps_close_by_root: Dict[str, float] = field(default_factory=dict)
    shock_slippage_multiplier: float = 2.0
    shock_gross_multiplier: float = 0.5       # must match execution shock config
    shock_z_threshold: float = 2.0            # must match execution shock config
    partial_fill_prob_shock: float = 0.0
    partial_fill_seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def cost_for_root(self, root: str, fallback: float = 2.5) -> float:
        return self.cost_per_contract_by_root.get(root, fallback)

    def slippage_open_for_root(self, root: str, fallback: float = 1.0) -> float:
        return self.slippage_bps_open_by_root.get(root, fallback)

    def slippage_close_for_root(self, root: str, fallback: float = 1.0) -> float:
        return self.slippage_bps_close_by_root.get(root, fallback)


@dataclass(frozen=True)
class PromotionWindow:
    """A named date-range window for promotion evaluation."""
    name: str
    start: str  # ISO date
    end: str    # ISO date
    sharpe_model: float = 0.0
    sharpe_baseline: float = 0.0
    hit_rate: float = 0.0
    max_drawdown: float = 0.0
    passed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PromotionWindowsReport:
    """Result of multi-window promotion evaluation."""
    windows: Tuple["PromotionWindow", ...] = ()
    primary_passed: bool = False
    stress_passed_count: int = 0
    stress_required: int = 1
    overall_passed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["windows"] = [w.to_dict() for w in self.windows]
        return d


@dataclass(frozen=True)
class Phase15Report:
    """Consolidated Phase 1.5 alignment report."""
    session_semantics: Optional[SessionSemanticsReport] = None
    feature_parity: Optional[FeatureParityReport] = None
    coverage: Optional[CoverageReport] = None
    trade_proxy: Optional[TradeProxyReport] = None
    all_passed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"all_passed": self.all_passed}
        if self.session_semantics is not None:
            d["session_semantics"] = self.session_semantics.to_dict()
        if self.feature_parity is not None:
            d["feature_parity"] = self.feature_parity.to_dict()
        if self.coverage is not None:
            d["coverage"] = self.coverage.to_dict()
        if self.trade_proxy is not None:
            d["trade_proxy"] = self.trade_proxy.to_dict()
        return d


# ---------------------------------------------------------------------------
# R1: Provider invariance
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProviderInvarianceReport:
    """Result of cross-provider session bar comparison."""
    session_semantics: Optional[SessionSemanticsReport] = None
    baseline_proxy_correlation: Dict[str, float] = field(default_factory=dict)
    r_co_quantile_comparison: Dict[str, Dict[str, float]] = field(default_factory=dict)
    missing_open_close_counts: Dict[str, Dict[str, int]] = field(default_factory=dict)
    overall_consistent: bool = True
    flags: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.session_semantics is not None:
            d["session_semantics"] = self.session_semantics.to_dict()
        d["flags"] = list(self.flags)
        return d


# ---------------------------------------------------------------------------
# R2: Tier2 calibration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Tier2CostDecomposition:
    """Per-tier cost breakdown in bps."""
    commission_bps_mean: float = 0.0
    slippage_bps_mean: float = 0.0
    impact_bps_mean: float = 0.0
    total_cost_bps_mean: float = 0.0
    contracts_adv_distribution: Dict[str, float] = field(default_factory=dict)
    downscale_trigger_rate: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Tier2CalibrationReport:
    """Full Tier2 realism calibration report."""
    ladder: Dict[str, Any] = field(default_factory=dict)
    cost_decomposition: Dict[str, Any] = field(default_factory=dict)
    sensitivity: Dict[str, Dict[str, float]] = field(default_factory=dict)
    adv_by_root: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Tier2CostDecomposition values may be nested — serialize them
        for tier_key, val in d.get("cost_decomposition", {}).items():
            if hasattr(val, "to_dict"):
                d["cost_decomposition"][tier_key] = val.to_dict()
        return d


# ---------------------------------------------------------------------------
# R3: Promotion summary
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PromotionSummary:
    """Single go/no-go document for paper-trading promotion."""
    run_id: str = ""
    timestamp: str = ""
    decision: str = "HOLD"   # PROMOTE / HOLD / FAIL
    integrity_checks: Dict[str, bool] = field(default_factory=dict)
    tier2_gates: Dict[str, Any] = field(default_factory=dict)
    provider_invariance: Optional["ProviderInvarianceReport"] = None
    tier2_calibration: Optional["Tier2CalibrationReport"] = None
    feature_list_used: Tuple[str, ...] = ()
    features_dropped: Tuple[str, ...] = ()
    config_hash: str = ""
    seed_list: Tuple[int, ...] = ()
    data_version_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "decision": self.decision,
            "integrity_checks": dict(self.integrity_checks),
            "tier2_gates": dict(self.tier2_gates),
            "feature_list_used": list(self.feature_list_used),
            "features_dropped": list(self.features_dropped),
            "config_hash": self.config_hash,
            "seed_list": list(self.seed_list),
            "data_version_hash": self.data_version_hash,
        }
        if self.provider_invariance is not None:
            d["provider_invariance"] = self.provider_invariance.to_dict()
        if self.tier2_calibration is not None:
            d["tier2_calibration"] = self.tier2_calibration.to_dict()
        return d


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunManifest:
    """Top-level manifest for a complete pipeline run."""
    run_id: str
    run_dir: str
    seed: int
    start_date: str
    end_date: str
    config_hash: str
    bronze: Optional[BronzeManifest] = None
    canonicalization: Optional[CanonicalizationManifest] = None
    dataset: Optional[DatasetManifest] = None
    splits: Tuple[SplitSpec, ...] = ()
    model: Optional[ModelBundle] = None
    validation: Optional[ValidationReport] = None
    phase15: Optional[Phase15Report] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "run_id": self.run_id,
            "run_dir": self.run_dir,
            "seed": self.seed,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "config_hash": self.config_hash,
        }
        if self.bronze is not None:
            d["bronze"] = self.bronze.to_dict()
        if self.canonicalization is not None:
            d["canonicalization"] = self.canonicalization.to_dict()
        if self.dataset is not None:
            d["dataset"] = self.dataset.to_dict()
        d["splits"] = [s.to_dict() for s in self.splits]
        if self.model is not None:
            d["model"] = self.model.to_dict()
        if self.validation is not None:
            d["validation"] = self.validation.to_dict()
        if self.phase15 is not None:
            d["phase15"] = self.phase15.to_dict()
            d["phase15_status"] = "PASS" if self.phase15.all_passed else "FAIL"
        return d

    def to_json(self) -> str:
        return _to_json(self.to_dict())
