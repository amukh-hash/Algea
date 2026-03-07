"""
VRP Strategy Plugin — the main strategy entry point.

v2: stores per-leg entry IV/marks, wires DeRiskPolicy instead of panic-close,
integrates danger-zone guard, stores risk_state_at_entry, exit_reason/exit_dt.
"""
from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from algae.data.options.vrp_features import (
    RegimeThresholds,
    VolRegime,
    classify_regime,
    compute_regime_features,
    compute_vrp_features,
)
from algae.execution.options.config import VRPConfig
from algae.execution.options.exits import DeRiskPolicy, ExitReason
from algae.execution.options.risk_guards import check_danger_zone
from algae.execution.options.structures import (
    DerivativesPosition,
    DerivativesPositionFrame,
    OptionLeg,
    StructureType,
)
from algae.trading.derivatives_risk import (
    check_early_assignment_risk,
    compute_scenario_with_contributors,
)


# ═══════════════════════════════════════════════════════════════════════════
# Strategy Plugin
# ═══════════════════════════════════════════════════════════════════════════

class VRPStrategy:
    """Options Volatility Risk Premium sleeve — defined-risk credit spreads."""

    def __init__(self, config: VRPConfig | None = None) -> None:
        self.config = config or VRPConfig()
        self._regime_thresholds = RegimeThresholds(
            vix_caution=self.config.regime_vix_caution,
            vix_crash=self.config.regime_vix_crash,
            vix_change_5d_crash=self.config.regime_vix_change_5d_crash,
            rv_ratio_crash=self.config.regime_rv_ratio_crash,
            drawdown_63d_crash=self.config.regime_drawdown_crash,
        )
        self._position_frame = DerivativesPositionFrame()
        self._derisk_policy = DeRiskPolicy(self.config)

    # ------------------------------------------------------------------
    # required_inputs
    # ------------------------------------------------------------------

    def required_inputs(self) -> Dict[str, Any]:
        """Declare data dependencies and schemas."""
        return {
            "option_chains": {
                "type": "parquet",
                "partitioned_by": ["date", "underlying"],
                "underlyings": list(self.config.underlyings),
            },
            "underlying_prices": {
                "type": "dataframe",
                "columns": ["date", "ticker", "close", "volume"],
            },
            "vix": {
                "type": "series",
                "description": "VIX index daily close",
            },
        }

    # ------------------------------------------------------------------
    # compute_features
    # ------------------------------------------------------------------

    def compute_features(
        self,
        as_of_date: date,
        chain: pd.DataFrame,
        close_prices: pd.Series,
        vix: pd.Series,
        vix_term_structure: Optional[pd.Series] = None,
        credit_ratio: Optional[pd.Series] = None,
        forecast_inputs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Compute VRP and regime features for a single underlying."""
        # Surface building logic decoupled to Inference Gateway
        surface = chain.attrs.get("surface_snapshot", {
            "term_slope": np.nan,
            "skew_25d": np.nan,
            "atm_iv": np.nan
        })

        vrp_feats = {
            "iv_rank_252": np.nan,
            "iv_minus_rv30": np.nan,
            "term_slope": surface.get("term_slope", np.nan),
            "skew_25d": surface.get("skew_25d", np.nan),
            "atm_iv": surface.get("atm_iv", np.nan),
        }

        regime_feats = compute_regime_features(
            close_prices, vix, vix_term_structure, credit_ratio,
        )
        if as_of_date in regime_feats.index:
            regime_row = regime_feats.loc[as_of_date]
        elif len(regime_feats) > 0:
            regime_row = regime_feats.iloc[-1]
        else:
            regime_row = pd.Series(dtype=float)

        regime = classify_regime(
            regime_row, self._regime_thresholds,
            forecast_inputs=forecast_inputs,
            config=self.config,
        )

        return {
            "vrp_features": vrp_feats,
            "regime_features": regime_row.to_dict() if not regime_row.empty else {},
            "regime": regime,
            "surface_snapshot": surface,
            "forecast_inputs": forecast_inputs,
        }

    # ------------------------------------------------------------------
    # predict (entry logic)
    # ------------------------------------------------------------------

    def predict(
        self,
        as_of_date: date,
        chain: pd.DataFrame,
        underlying: str,
        underlying_price: float,
        features: Dict[str, Any],
        nav: float,
    ) -> Optional[DerivativesPosition]:
        """Generate a target derivative position (or None if gated)."""
        cfg = self.config
        regime: VolRegime = features.get("regime", VolRegime.CRASH_RISK)

        # --- Regime gate: CRASH_RISK always blocks new entries ---
        if regime == VolRegime.CRASH_RISK:
            return None

        # --- Position count gate ---
        existing = self._position_frame.positions_by_underlying().get(underlying, [])
        if len(existing) >= cfg.max_positions_per_underlying:
            return None

        # --- IV rank gate ---
        vrp = features.get("vrp_features", {})
        iv_rank = vrp.get("iv_rank_252", np.nan)
        if not np.isnan(iv_rank) and iv_rank < cfg.iv_rank_threshold:
            return None

        # --- Reduced size in CAUTION ---
        size_scale = 1.0
        if regime == VolRegime.CAUTION:
            size_scale = 0.5

        # --- Select expiry ---
        dte_min, dte_max = cfg.dte_range
        valid_exp = chain[(chain["dte"] >= dte_min) & (chain["dte"] <= dte_max)]
        if valid_exp.empty:
            return None
        target_expiry = valid_exp.sort_values("dte").iloc[0]["expiry"]
        target_dte = int(valid_exp.sort_values("dte").iloc[0]["dte"])
        exp_chain = chain[chain["expiry"] == target_expiry]

        # --- Build put credit spread ---
        position = self._build_put_credit_spread(
            exp_chain, underlying, underlying_price, target_expiry,
            target_dte, nav, size_scale, features, as_of_date,
        )
        return position

    def _build_put_credit_spread(
        self,
        exp_chain: pd.DataFrame,
        underlying: str,
        underlying_price: float,
        expiry: date,
        dte: int,
        nav: float,
        size_scale: float,
        features: Dict[str, Any],
        as_of_date: date,
    ) -> Optional[DerivativesPosition]:
        """Construct a PUT_CREDIT_SPREAD with full entry metadata."""
        cfg = self.config
        puts = exp_chain[exp_chain["option_type"].str.lower() == "put"].copy()
        if puts.empty:
            return None

        T = dte / 365.0
        r = float(puts.get("risk_free_rate", pd.Series([0.0])).iloc[0])
        q = float(puts.get("dividend_yield", pd.Series([0.0])).iloc[0])

        # Use pre-supplied deltas decoupled to Canonical Layer
        puts = puts.copy()
        puts["_delta_abs"] = puts["delta"].abs() if "delta" in puts.columns else 0.0

        # Target short strike by delta
        puts["_d_dist"] = (puts["_delta_abs"] - cfg.delta_target).abs()
        short_idx = puts["_d_dist"].idxmin()
        short_row = puts.loc[short_idx]
        short_strike = float(short_row["strike"])
        short_iv = float(short_row["implied_vol"])
        short_mid = float(short_row["mid"])
        short_bid = float(short_row["bid"])
        short_ask = float(short_row["ask"]) if "ask" in short_row.index else short_mid * 1.01

        # Long strike = short_strike - width
        long_strike_target = short_strike - cfg.spread_width
        long_candidates = puts[puts["strike"] <= long_strike_target]
        if long_candidates.empty:
            return None
        long_row = long_candidates.iloc[long_candidates["strike"].sub(long_strike_target).abs().argmin()]
        long_strike = float(long_row["strike"])
        long_iv = float(long_row["implied_vol"])
        long_mid = float(long_row["mid"])
        long_ask = float(long_row["ask"]) if "ask" in long_row.index else long_mid * 1.01
        long_bid = float(long_row["bid"])

        # Spread credit at mid
        credit_mid = short_mid - long_mid
        if credit_mid <= 0:
            return None

        actual_width = short_strike - long_strike
        max_loss_per_share = actual_width - credit_mid
        if max_loss_per_share <= 0:
            return None

        # Risk limit check
        max_loss_dollar = max_loss_per_share * cfg.contract_multiplier
        if max_loss_dollar / nav > cfg.max_risk_per_structure_pct_nav:
            return None

        total_existing = self._position_frame.total_max_loss()
        if (total_existing + max_loss_dollar) / nav > cfg.max_total_vrp_risk_pct_nav:
            return None

        # Greeks at entry
        short_delta = float(short_row.get("delta", 0.0))
        long_delta = float(long_row.get("delta", 0.0))
        short_gamma_v = float(short_row.get("gamma", 0.0))
        long_gamma_v = float(long_row.get("gamma", 0.0))
        short_vega_v = float(short_row.get("vega", 0.0))
        long_vega_v = float(long_row.get("vega", 0.0))
        short_theta_v = float(short_row.get("theta", 0.0))
        long_theta_v = float(long_row.get("theta", 0.0))

        net_delta = -short_delta + long_delta
        net_gamma = -short_gamma_v + long_gamma_v
        net_vega = -short_vega_v + long_vega_v
        net_theta = -short_theta_v + long_theta_v

        entry_time = datetime.combine(as_of_date, datetime.min.time())
        legs = [
            OptionLeg(
                option_type="put", strike=short_strike, qty=-1, side="sell",
                entry_price_mid=short_mid,
                delta=short_delta, gamma=short_gamma_v, vega=short_vega_v, theta=short_theta_v,
                entry_iv=short_iv, entry_mid=short_mid, entry_underlying=underlying_price,
                entry_bid=short_bid, entry_ask=short_ask, entry_dt=entry_time,
            ),
            OptionLeg(
                option_type="put", strike=long_strike, qty=1, side="buy",
                entry_price_mid=long_mid,
                delta=long_delta, gamma=long_gamma_v, vega=long_vega_v, theta=long_theta_v,
                entry_iv=long_iv, entry_mid=long_mid, entry_underlying=underlying_price,
                entry_bid=long_bid, entry_ask=long_ask, entry_dt=entry_time,
            ),
        ]

        regime = features.get("regime", VolRegime.NORMAL_CARRY)
        pos = DerivativesPosition(
            underlying=underlying,
            structure_type=StructureType.PUT_CREDIT_SPREAD,
            expiry=expiry if isinstance(expiry, date) else pd.Timestamp(expiry).date(),
            legs=legs,
            premium_collected=credit_mid,
            max_loss=max_loss_per_share,
            multiplier=cfg.contract_multiplier,
            delta=net_delta,
            gamma=net_gamma,
            vega=net_vega,
            theta=net_theta,
            risk_budget_used=max_loss_dollar / nav,
            target_vol_scale=size_scale,
            entry_dt=entry_time,
            strategy_tags={
                "regime": regime.value,
                "iv_rank": features.get("vrp_features", {}).get("iv_rank_252"),
                "skew_25d": features.get("surface_snapshot", {}).get("skew_25d"),
            },
            position_id=str(uuid.uuid4())[:8],
            risk_state_at_entry={
                "regime": regime.value,
                "forecast_inputs": features.get("forecast_inputs"),
                "underlying_price": underlying_price,
                "atm_iv": features.get("surface_snapshot", {}).get("atm_iv"),
            },
        )
        return pos

    # ------------------------------------------------------------------
    # check_exits (v2 — wired to DeRiskPolicy + danger zone)
    # ------------------------------------------------------------------

    def check_exits(
        self,
        as_of_date: date,
        current_marks: Dict[str, float],
        regime: VolRegime,
        underlying_prices: Optional[Dict[str, float]] = None,
        rv_estimates: Optional[Dict[str, float]] = None,
        nav: float = 1_000_000.0,
    ) -> List[Tuple[DerivativesPosition, str]]:
        """Check exit triggers for all open positions.

        Returns list of (position, reason) pairs for positions to close.
        Uses prioritised de-risk instead of panic-close.
        """
        cfg = self.config
        exits: List[Tuple[DerivativesPosition, str]] = []
        already_closed: set = set()

        for pos in self._position_frame.open_positions:
            mark = current_marks.get(pos.position_id, pos.premium_collected)
            dte_remaining = (pos.expiry - as_of_date).days

            # 1. Time exit
            if dte_remaining <= cfg.min_dte_exit:
                exits.append((pos, ExitReason.DTE_EXIT.value))
                already_closed.add(pos.position_id)
                continue

            # 2. Profit take
            profit_threshold = pos.premium_collected * (1.0 - cfg.profit_take_pct)
            if mark <= profit_threshold:
                exits.append((pos, ExitReason.PROFIT_TAKE.value))
                already_closed.add(pos.position_id)
                continue

            # 3. Stop loss
            stop_level = pos.premium_collected * cfg.stop_loss_multiple
            if mark >= stop_level:
                exits.append((pos, ExitReason.STOP_LOSS.value))
                already_closed.add(pos.position_id)
                continue

            # 4. Early assignment guard
            if underlying_prices:
                price = underlying_prices.get(pos.underlying, 0.0)
                if price > 0 and check_early_assignment_risk(pos, price, as_of_date, cfg):
                    exits.append((pos, ExitReason.EARLY_ASSIGNMENT.value))
                    already_closed.add(pos.position_id)
                    continue

            # 5. Danger zone guard
            if underlying_prices and rv_estimates:
                price = underlying_prices.get(pos.underlying, 0.0)
                rv = rv_estimates.get(pos.underlying, 0.01)
                if price > 0:
                    dz = check_danger_zone(pos, price, rv, as_of_date, cfg)
                    if dz.in_danger:
                        exits.append((pos, ExitReason.DANGER_ZONE.value))
                        already_closed.add(pos.position_id)
                        continue

        # 6. If CRASH_RISK or CAUTION, run de-risk policy on remaining positions
        if regime in (VolRegime.CRASH_RISK, VolRegime.CAUTION):
            remaining = [
                p for p in self._position_frame.open_positions
                if p.position_id not in already_closed
            ]
            if remaining and underlying_prices:
                # Compute scenario contributions for remaining
                from algae.execution.options.structures import DerivativesPositionFrame
                temp_frame = DerivativesPositionFrame(positions=remaining)
                _, contribs = compute_scenario_with_contributors(
                    temp_frame, underlying_prices, as_of_date,
                )
                total_loss = sum(contribs.values())

                # Danger zone flags
                dz_flags: Dict[str, bool] = {}
                if rv_estimates:
                    for p in remaining:
                        price = underlying_prices.get(p.underlying, 0.0)
                        rv = rv_estimates.get(p.underlying, 0.01)
                        if price > 0:
                            dz = check_danger_zone(p, price, rv, as_of_date, cfg)
                            dz_flags[p.position_id] = dz.in_danger

                summary = self._derisk_policy.evaluate(
                    remaining, regime.value, contribs, total_loss, nav, dz_flags,
                )
                for action in summary.actions:
                    exits.append((action.position, action.reason.value))

        return exits

    # ------------------------------------------------------------------
    # risk_report
    # ------------------------------------------------------------------

    def risk_report(
        self,
        as_of_date: date,
        positions: Optional[DerivativesPositionFrame] = None,
    ) -> Dict[str, Any]:
        """Return exposure summary and limit checks."""
        pf = positions or self._position_frame
        greeks = pf.aggregate_greeks()
        total_max_loss = pf.total_max_loss()
        total_premium = pf.total_premium_collected()

        return {
            "as_of_date": as_of_date,
            "open_positions": len(pf.open_positions),
            "total_max_loss": total_max_loss,
            "total_premium_collected": total_premium,
            "aggregate_greeks": greeks,
            "positions_summary": pf.to_summary_df().to_dict("records"),
        }

    # ------------------------------------------------------------------
    # train (stub — rules-based v1)
    # ------------------------------------------------------------------

    def train(self, **kwargs: Any) -> None:
        """ML-based VRP training not yet implemented (rules-based v1)."""
        raise NotImplementedError("ML-based VRP training not yet implemented")

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    @property
    def position_frame(self) -> DerivativesPositionFrame:
        return self._position_frame

    def add_position(self, pos: DerivativesPosition) -> None:
        self._position_frame.add(pos)

    def close_position(
        self,
        pos: DerivativesPosition,
        reason: str,
        realized_pnl: float,
        as_of_date: Optional[date] = None,
    ) -> None:
        pos.is_open = False
        pos.realized_pnl = realized_pnl
        pos.exit_reason = reason
        pos.exit_dt = (
            datetime.combine(as_of_date, datetime.min.time())
            if as_of_date else datetime.now()
        )
