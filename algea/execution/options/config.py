"""
VRP strategy configuration — all tuneable parameters in one serialisable dataclass.

v2: adds hardened defaults for gamma guard, de-risk policy, scenario budgets,
concentration controls, margin semantics, assignment guard, forecast thresholds,
dynamic shocks, and allocator stability.  Crash thresholds lowered to trigger
realistically during transition-to-crash regimes (2018, 2020, 2022).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Tuple, FrozenSet


@dataclass(frozen=True)
class VRPConfig:
    """Immutable configuration for the VRP sleeve."""

    # --- Universe ---
    underlyings: Tuple[str, ...] = ("SPY", "QQQ", "IWM")

    # --- Expiry / strike ---
    dte_range: Tuple[int, int] = (30, 45)
    delta_target: float = 0.15             # short leg delta (absolute)
    spread_width: int = 5                  # points between legs
    structure_types: Tuple[str, ...] = ("PUT_CREDIT_SPREAD",)

    # --- Entry filters ---
    iv_rank_threshold: float = 0.50
    iv_minus_rv_threshold: float = 0.02
    max_spread_pct: float = 0.10           # max bid-ask spread as % of mid
    min_open_interest: int = 100
    min_volume: int = 50
    # Entry quality gates (Phase 2 expansion)
    entry_iv_rank_normal: float = 0.45     # IV rank floor for normal entries
    entry_iv_rank_caution: float = 0.70    # much higher bar in caution
    entry_forecast_p90_normal: float = 0.25  # max forecast p90 for normal entry
    entry_scenario_pct_normal: float = 0.60  # max scenario loss as % of budget
    entry_min_health_caution: float = 0.90   # min forecast health for caution entry
    entry_min_headroom_caution: float = 0.50 # headroom ratio for caution entry

    # --- Exit rules ---
    profit_take_pct: float = 0.50          # close at 50% of credit captured
    stop_loss_multiple: float = 2.0        # close if debit >= 2x credit
    min_dte_exit: int = 7                  # force close at this DTE
    max_positions_per_underlying: int = 1

    # --- Regime thresholds (REALISTIC — calibrated to trigger in 2018/2020/2022) ---
    regime_vix_caution: float = 22.0       # lowered from 25
    regime_vix_crash: float = 28.0         # lowered from 35
    regime_vix_change_5d_crash: float = 0.18   # 18% in 5d (from 30%)
    regime_rv_ratio_crash: float = 1.5     # from 1.8
    regime_drawdown_crash: float = -0.08   # from -0.10
    regime_credit_spread_crash: float = 0.03   # HYG/LQD proxy threshold

    # --- Gamma danger-zone guard ---
    danger_zone_delta_threshold: float = 0.30
    danger_zone_z_threshold: float = 0.85
    danger_zone_close_if_both: bool = True

    # --- De-risk policy ---
    max_daily_derisk_actions: int = 3
    close_until_constraints_satisfied: bool = True
    liquidity_widen_factor: float = 1.7
    close_top_contributors_pct: float = 0.50

    # --- Slippage & costs ---
    slippage_bps: float = 5.0             # per-leg bps
    commission_per_contract: float = 0.65
    exchange_fee_per_contract: float = 0.20

    # --- Risk limits ---
    max_risk_per_structure_pct_nav: float = 0.02      # 2%
    max_total_vrp_risk_pct_nav: float = 0.10           # 10%
    max_vrp_notional_pct_nav: float = 0.25             # 25%
    max_vega_per_underlying: float = 500.0
    max_gamma_per_underlying: float = 50.0
    max_short_convexity_score: float = 80.0            # tightened from 100

    # --- Scenario budgets / tail controls ---
    max_worst_case_scenario_loss_pct_nav: float = 0.06  # hard stop budget

    # --- Concentration ---
    max_loss_pct_single_expiry: float = 0.50
    min_short_strike_spacing_pct: float = 0.02

    # --- Margin semantics ---
    margin_buffer_multiplier: float = 1.4
    budget_basis: str = "margin"           # "risk" | "margin" — default safer for live

    # --- Assignment guard ---
    early_assignment_extrinsic_threshold: float = 0.15

    # --- Sleeve vol targeting ---
    target_sleeve_vol: float = 0.08        # 8% annualised
    risk_scaler_lookback: int = 20

    # --- Forecast-aware thresholds (Lag-Llama) ---
    min_forecast_health: float = 0.80
    rv10_pred_p90_caution: float = 0.28    # annualised
    rv10_pred_p95_crash: float = 0.35      # annualised
    rv10_pred_p99_crash: float = 0.45
    forecast_override_weight: float = 1.0

    # --- Dynamic shocks ---
    use_dynamic_shocks: bool = True
    dynamic_shock_k: float = 2.5

    # --- Allocator ---
    w_max_vrp: float = 0.25
    allocator_grid_step: float = 0.01
    tail_penalty_lambda: float = 1.5
    dd_penalty_m: float = 0.5
    headroom_bonus_weight: float = 0.3    # bonus for strong headroom
    regime_return_weight_normal: float = 1.0
    regime_return_weight_caution: float = 0.6  # penalize caution returns
    convexity_penalty_weight: float = 0.2
    target_utilization_normal: float = 0.15  # target mean w_vrp in NORMAL
    utilization_pull_strength: float = 0.5   # soft pull toward target

    # --- Allocator stability ---
    w_smoothing_alpha: float = 0.18       # raised from 0.10 for faster ramp while safe
    w_max_daily_delta: float = 0.02       # invariant: max daily change
    w_min_hold_days: int = 3              # relaxed from 5 for better utilization

    # --- Allocator robustness ---
    w_min_deployment: float = 0.05        # snap-to-zero on decay only
    w_reentry_threshold: float = 0.06     # lowered from 0.08 → easier ramp from zero
    w_volatility_penalty: float = 0.5     # penalise weight flip-flop
    w_history_window: int = 20            # lookback for weight vol metric

    # --- Regime-conditioned exposure ---
    w_vrp_regime_cap_normal: float = 1.0  # fraction of w_opt allowed in normal
    w_vrp_regime_cap_caution: float = 0.50  # raised from 0.25 (dynamic cap also applied)
    w_vrp_regime_cap_crash: float = 0.0   # NEVER CHANGE — crash = zero exposure
    enable_dynamic_caution_cap: bool = True   # dynamic cap: min(0.50*w_opt, 0.10)
    normal_boost_multiplier: float = 1.2      # boost in NORMAL when health+headroom good
    min_headroom_normal_boost: float = 0.50   # headroom ratio required for boost
    max_dynamic_caution_weight: float = 0.10  # hard ceiling for caution dynamic cap

    # --- Regime hysteresis (Phase 2) ---
    regime_crash_entry_score: int = 2     # score >= this → enter CRASH_RISK
    regime_crash_exit_score: int = 0      # score <= this for N days → exit
    regime_caution_entry_score: int = 2
    regime_caution_exit_score: int = 0
    regime_min_days_in_state: int = 2     # consecutive days before exit

    # --- Liquidity gating ---
    max_spread_pct_live: float = 0.12     # max spread to execute optional closes
    liquidity_block_optional_closes: bool = True
    max_optional_deferral_days: int = 3   # force close after N days of deferral

    # --- Scenario headroom ---
    min_headroom_for_new_entries: float = 0.01  # 1% NAV headroom required

    # --- Event avoidance (Phase 4) ---
    avoid_open_days_before_event: int = 1
    avoid_open_days_after_event: int = 0

    # --- Live guard ---
    live_guard_hard_loss_limit: float = 0.08  # block trades above this
    live_guard_margin_limit: float = 0.90     # 90% margin utilisation
    live_guard_health_drop_threshold: float = 0.30  # health drop in 3 days
    live_guard_health_drop_reduce_pct: float = 0.50  # reduce VRP weight by 50%

    # --- Forecast health fail-safe (Phase 5) ---
    forecast_health_drop_window_days: int = 3
    forecast_health_drop_threshold: float = 0.15  # drop > this over window
    forecast_health_reduce_pct: float = 0.50      # reduce w_vrp by this fraction

    # --- Monitoring thresholds ---
    monitor_scenario_warning_pct: float = 0.80  # warn if loss > 80% of budget
    monitor_convexity_warning_pct: float = 0.80
    monitor_health_warning: float = 0.70

    # --- Multiplier ---
    contract_multiplier: int = 100

    # ---------------------------------------------------------------
    # Serialisation
    # ---------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VRPConfig":
        # Convert lists → tuples for frozen dataclass
        for k in ("underlyings", "dte_range", "structure_types"):
            if k in d and isinstance(d[k], list):
                d[k] = tuple(d[k])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_json(cls, s: str) -> "VRPConfig":
        return cls.from_dict(json.loads(s))
