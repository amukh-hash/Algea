from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from algea.data.options.vrp_features import VolRegime
from algea.trading.meta_allocator import SleeveResult

from .config import COOCReversalConfig
from .contract_master import CONTRACT_MASTER
from .execution import build_entry_orders, flatten_orders
from .features import FeatureRow, compute_core_features
from .portfolio import market_neutral_weights, utility_signal
from .sizing import contracts_from_weights


@dataclass
class SleeveHealth:
    confidence: float
    no_trade: bool


class COOCReversalFuturesSleeve:
    """Pattern-B CO->OC reversal sleeve.

    Integration notes discovered while exploring repository:
    - There is no global BaseSleeve class yet; production sleeves are strategy classes
      (e.g. ``algea.execution.options.vrp_strategy.VRPStrategy``).
    - Meta allocation currently passes ``SleeveResult`` objects into
      ``algea.trading.meta_allocator.MetaAllocator.combine``.
    - Regime states come from ``VolRegime`` enum: NORMAL_CARRY/CAUTION/CRASH_RISK.
    - Model registry exists under ``algea.models.foundation.registry``.
    - Broker abstraction exists at ``algea.trading.broker_base`` with concrete IBKR adapters.
    """

    def __init__(self, config: COOCReversalConfig | None = None) -> None:
        self.config = config or COOCReversalConfig()

    def compute_signal_frame(self, frame: pd.DataFrame, decision_ts: pd.Timestamp) -> pd.DataFrame:
        feats = compute_core_features(frame, lookback=self.config.lookback)
        feats = feats[feats["feature_timestamp_end"] < decision_ts].copy()
        return feats

    def build_daily_orders(
        self,
        date_t: date,
        pred_mu: dict[str, float],
        pred_sigma: dict[str, float],
        prices: dict[str, float],
        capital: float,
        regime: VolRegime,
        no_trade_flags: dict[str, bool] | None = None,
        shock_scores: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Build daily order intents with optional shock gating.

        Parameters
        ----------
        shock_scores
            Per-instrument shock_score values (``abs(r_co)/(sigma_co+eps)``).
            If provided and ``config.shock.enabled``, instruments exceeding
            ``shock.shock_z_threshold`` have their gross scaled by
            ``shock.gross_multiplier_on_shock``.  A multiplier of 0.0 blocks
            entry entirely.  Flatten intents are **always** allowed.
        """
        flags = no_trade_flags or {}
        if regime == VolRegime.CRASH_RISK:
            return {"weights": {k: 0.0 for k in pred_mu}, "contracts": {k: 0 for k in pred_mu}, "orders": []}

        gross = self.config.gross_target * (self.config.caution_scale if regime == VolRegime.CAUTION else 1.0)

        # --- Compute per-instrument shock flags ---
        shock_cfg = self.config.shock
        shock_blocked: set[str] = set()
        per_instrument_gross: dict[str, float] = {}

        if shock_cfg.enabled and shock_scores:
            for sym, score in shock_scores.items():
                if score > shock_cfg.shock_z_threshold:
                    if shock_cfg.gross_multiplier_on_shock == 0.0:
                        shock_blocked.add(sym)
                    elif shock_cfg.per_instrument:
                        per_instrument_gross[sym] = gross * shock_cfg.gross_multiplier_on_shock
                    else:
                        # Global scaling: reduce entire gross
                        gross *= shock_cfg.gross_multiplier_on_shock

        symbols = [s for s in pred_mu if not flags.get(s, False) and s not in shock_blocked]
        mu = np.array([pred_mu[s] for s in symbols], dtype=float)
        sigma = np.array([max(1e-5, pred_sigma[s]) for s in symbols], dtype=float)
        sig = utility_signal(mu, sigma)

        # Apply per-instrument gross scaling if applicable
        if per_instrument_gross:
            effective_gross = np.array([
                per_instrument_gross.get(s, gross) for s in symbols
            ], dtype=float)
            # Weight with the minimum gross for market-neutrality, then scale individually
            base_gross = min(effective_gross) if len(effective_gross) > 0 else gross
            w = market_neutral_weights(sig, gross=base_gross, net_cap=self.config.net_cap)
            # Re-scale each weight by its instrument-specific gross ratio
            for i, s in enumerate(symbols):
                if s in per_instrument_gross:
                    w[i] *= per_instrument_gross[s] / base_gross if base_gross > 0 else 0.0
        else:
            w = market_neutral_weights(sig, gross=gross, net_cap=self.config.net_cap)

        weights = {s: float(v) for s, v in zip(symbols, w)}
        multipliers = {s: CONTRACT_MASTER[s].multiplier for s in symbols}
        contracts = contracts_from_weights(weights, capital, {s: prices[s] for s in symbols}, multipliers, self.config.max_contracts_per_instrument)

        result: dict[str, Any] = {
            "weights": weights,
            "contracts": contracts,
            "orders": build_entry_orders(contracts),
        }

        # Record shock metadata in artifacts
        if shock_cfg.enabled and shock_scores:
            result["shock_metadata"] = {
                "shock_blocked": sorted(shock_blocked),
                "shock_scaled": {s: per_instrument_gross[s] for s in sorted(per_instrument_gross)},
                "shock_scores": shock_scores,
                "threshold": shock_cfg.shock_z_threshold,
                "multiplier": shock_cfg.gross_multiplier_on_shock,
            }

        return result

    def force_eod_flatten(self, positions: dict[str, int]) -> list[Any]:
        return flatten_orders(positions)

    def sleeve_result(self, expected_return: float, realized_vol: float, tail_proxy: float, confidence: float) -> SleeveResult:
        return SleeveResult(
            name="cooc_reversal_futures",
            expected_return=expected_return,
            realized_vol=realized_vol,
            es_95=tail_proxy,
            forecast_risk={0.95: realized_vol},
        )


def assert_feature_provenance(feature_timestamp_end, decision_timestamp) -> None:
    FeatureRow(values={}, feature_timestamp_end=feature_timestamp_end, decision_timestamp=decision_timestamp).assert_no_leakage()
