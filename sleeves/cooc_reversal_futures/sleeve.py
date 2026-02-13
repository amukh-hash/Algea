from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from algaie.data.options.vrp_features import VolRegime
from algaie.trading.meta_allocator import SleeveResult

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
      (e.g. ``algaie.execution.options.vrp_strategy.VRPStrategy``).
    - Meta allocation currently passes ``SleeveResult`` objects into
      ``algaie.trading.meta_allocator.MetaAllocator.combine``.
    - Regime states come from ``VolRegime`` enum: NORMAL_CARRY/CAUTION/CRASH_RISK.
    - Model registry exists under ``algaie.models.foundation.registry``.
    - Broker abstraction exists at ``algaie.trading.broker_base`` with concrete IBKR adapters.
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
    ) -> dict[str, Any]:
        flags = no_trade_flags or {}
        if regime == VolRegime.CRASH_RISK:
            return {"weights": {k: 0.0 for k in pred_mu}, "contracts": {k: 0 for k in pred_mu}, "orders": []}

        gross = self.config.gross_target * (self.config.caution_scale if regime == VolRegime.CAUTION else 1.0)
        symbols = [s for s in pred_mu if not flags.get(s, False)]
        mu = np.array([pred_mu[s] for s in symbols], dtype=float)
        sigma = np.array([max(1e-5, pred_sigma[s]) for s in symbols], dtype=float)
        sig = utility_signal(mu, sigma)
        w = market_neutral_weights(sig, gross=gross, net_cap=self.config.net_cap)
        weights = {s: float(v) for s, v in zip(symbols, w)}
        multipliers = {s: CONTRACT_MASTER[s].multiplier for s in symbols}
        contracts = contracts_from_weights(weights, capital, {s: prices[s] for s in symbols}, multipliers, self.config.max_contracts_per_instrument)
        return {"weights": weights, "contracts": contracts, "orders": build_entry_orders(contracts)}

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
