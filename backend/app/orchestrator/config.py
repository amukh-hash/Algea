from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class SessionWindow:
    start: str
    end: str


@dataclass
class OrchestratorConfig:
    timezone: str = field(default_factory=lambda: os.getenv("ORCH_TIMEZONE", "America/New_York"))
    exchange: str = field(default_factory=lambda: os.getenv("ORCH_EXCHANGE", "XNYS"))
    mode: str = field(default_factory=lambda: os.getenv("ORCH_MODE", "paper"))
    poll_interval_s: int = field(default_factory=lambda: int(os.getenv("ORCH_POLL_INTERVAL_S", "60")))
    artifact_root: Path = field(
        default_factory=lambda: Path(os.getenv("ORCH_ARTIFACT_ROOT", "backend/artifacts/orchestrator"))
    )
    db_path: Path = field(
        default_factory=lambda: Path(os.getenv("ORCH_DB_PATH", "backend/artifacts/orchestrator_state/state.sqlite3"))
    )

    enabled_jobs: list[str] = field(default_factory=list)
    disabled_jobs: list[str] = field(default_factory=list)
    paper_only: bool = field(default_factory=lambda: os.getenv("ORCH_PAPER_ONLY", "1") == "1")
    account_equity: float = field(default_factory=lambda: float(os.getenv("ORCH_ACCOUNT_EQUITY", "100000")))
    max_order_notional: float = field(default_factory=lambda: float(os.getenv("ORCH_MAX_ORDER_NOTIONAL", "5000")))
    max_total_order_notional: float = field(default_factory=lambda: float(os.getenv("ORCH_MAX_TOTAL_ORDER_NOTIONAL", "25000")))
    max_orders: int = field(default_factory=lambda: int(os.getenv("ORCH_MAX_ORDERS", "20")))
    # ── Safety Lockdown (2026-03-03) ──────────────────────────────────────
    # DO NOT enable these sleeves until their ML models have trained weights.
    # The serving layer currently contains stubs:
    #   - Chronos2: linear extrapolation (adapter.py) — real teacher exists in algae.models
    #   - SMoE Selector: hardcoded expert weights — real RankTransformer exists in algae.models
    #   - Vol Surface VRP: patch averaging — no real architecture exists
    #   - StatArb iTransformer: weighted sum — real architecture now in algae.models.tsfm
    # Enabling any of these will route deterministic garbage or random-weight
    # noise through the risk pipeline, poisoning MMD baselines and ECE calibration.
    # ────────────────────────────────────────────────────────────────────────
    enable_chronos2_sleeve: bool = field(default_factory=lambda: os.getenv("ENABLE_CHRONOS2_SLEEVE", "0") == "1")
    enable_smoe_selector: bool = field(default_factory=lambda: os.getenv("ENABLE_SMOE_SELECTOR", "0") == "1")
    selector_model_alias: str = field(default_factory=lambda: os.getenv("SELECTOR_MODEL_ALIAS", "prod"))
    enable_vol_surface_vrp: bool = field(default_factory=lambda: os.getenv("ENABLE_VOL_SURFACE_VRP", "0") == "1")
    vrp_model_alias: str = field(default_factory=lambda: os.getenv("VRP_MODEL_ALIAS", "prod"))
    enable_statarb_sleeve: bool = field(default_factory=lambda: os.getenv("ENABLE_STATARB_SLEEVE", "0") == "1")
    itransformer_model_alias: str = field(default_factory=lambda: os.getenv("ITRANSFORMER_MODEL_ALIAS", "prod"))
    # ── Intent Supremacy Feature Flags ────────────────────────────────────
    # These flags control the phased migration to the canonical sleeve output
    # and unified risk/planning pipeline. All default OFF.
    FF_CANONICAL_SLEEVE_OUTPUTS: bool = field(default_factory=lambda: os.getenv("FF_CANONICAL_SLEEVE_OUTPUTS", "0") == "1")
    FF_CANONICAL_RISK_ENGINE: bool = field(default_factory=lambda: os.getenv("FF_CANONICAL_RISK_ENGINE", "0") == "1")
    FF_CANONICAL_PLANNER: bool = field(default_factory=lambda: os.getenv("FF_CANONICAL_PLANNER", "0") == "1")
    FF_WRITE_COMPAT_TARGETS: bool = field(default_factory=lambda: os.getenv("FF_WRITE_COMPAT_TARGETS", "1") == "1")
    FF_STRICT_CORE_CONFIG: bool = field(default_factory=lambda: os.getenv("FF_STRICT_CORE_CONFIG", "1") == "1")
    FF_STRICT_STATARB_DECOMPOSITION: bool = field(default_factory=lambda: os.getenv("FF_STRICT_STATARB_DECOMPOSITION", "1") == "1")
    session_windows: dict[str, SessionWindow] = field(
        default_factory=lambda: {
            "PREMARKET": SessionWindow("07:00", "09:25"),
            "OPEN": SessionWindow("09:25", "09:35"),
            "INTRADAY": SessionWindow("09:35", "15:45"),
            "PRECLOSE": SessionWindow("15:45", "15:58"),
            "CLOSE": SessionWindow("15:58", "16:10"),
            "AFTERHOURS": SessionWindow("16:10", "20:00"),
        }
    )
