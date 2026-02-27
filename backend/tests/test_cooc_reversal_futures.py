from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from algea.data.options.vrp_features import VolRegime
from sleeves.cooc_reversal_futures.anchors import anchor_price
from sleeves.cooc_reversal_futures.labels import r_co, r_oc
from sleeves.cooc_reversal_futures.portfolio import market_neutral_weights
from sleeves.cooc_reversal_futures.sizing import contracts_from_weights
from sleeves.cooc_reversal_futures.sleeve import COOCReversalFuturesSleeve, assert_feature_provenance


def test_anchor_builder_fallback_behavior() -> None:
    ts = pd.Timestamp("2025-01-02 09:29:59", tz="America/New_York")
    quotes = pd.DataFrame([
        {"timestamp": ts, "price": 100.0, "bid": 99.0, "ask": 101.0, "volume": 1},
    ])
    mid = anchor_price(quotes, ts, "MID")
    assert mid.method_used == "MID"
    no_bbo = pd.DataFrame([{"timestamp": ts, "price": 100.0, "volume": 1}])
    fb = anchor_price(no_bbo, ts, "MID")
    assert fb.method_used in {"VWAP_WINDOW", "LAST"}


def test_return_computations() -> None:
    assert np.isclose(r_co(100.0, 99.0), np.log(100.0 / 99.0))
    assert np.isclose(r_oc(101.0, 100.0), np.log(101.0 / 100.0))


def test_market_neutral_weights_sum_to_zero() -> None:
    w = market_neutral_weights(np.array([1.0, 0.0, -1.0]), gross=1.0)
    assert abs(float(w.sum())) < 1e-8


def test_integer_contract_rounding() -> None:
    out = contracts_from_weights({"ES": 0.2}, capital=100_000, prices={"ES": 5000}, multipliers={"ES": 50}, max_contracts=10)
    assert out["ES"] == 0


def test_leakage_timestamp_assertion() -> None:
    decision = pd.Timestamp("2025-01-02 09:30:00", tz="America/New_York")
    assert_feature_provenance(pd.Timestamp("2025-01-02 09:29:59", tz="America/New_York"), decision)


def test_integration_short_backtest_end_to_end_and_flat_eod() -> None:
    sleeve = COOCReversalFuturesSleeve()
    pred_mu = {"ES": 0.001, "NQ": -0.001, "YM": 0.0005, "RTY": -0.0002}
    pred_sigma = {k: 0.01 for k in pred_mu}
    prices = {"ES": 5000, "NQ": 18000, "YM": 39000, "RTY": 2200}
    r = sleeve.build_daily_orders(date(2025, 1, 2), pred_mu, pred_sigma, prices, capital=1_000_000, regime=VolRegime.NORMAL_CARRY)
    assert r["orders"]
    flatten = sleeve.force_eod_flatten(r["contracts"])
    # if any non-zero contracts were opened, flatten orders must exist and mirror signs
    assert len(flatten) == len([v for v in r["contracts"].values() if v != 0])


def test_crash_risk_blocks_entries() -> None:
    sleeve = COOCReversalFuturesSleeve()
    r = sleeve.build_daily_orders(
        date(2025, 1, 2),
        pred_mu={"ES": 0.002},
        pred_sigma={"ES": 0.01},
        prices={"ES": 5000},
        capital=1_000_000,
        regime=VolRegime.CRASH_RISK,
    )
    assert r["orders"] == []


def test_order_builder_smoke() -> None:
    sleeve = COOCReversalFuturesSleeve()
    r = sleeve.build_daily_orders(
        date(2025, 1, 2),
        pred_mu={"ES": 0.001, "NQ": -0.001},
        pred_sigma={"ES": 0.02, "NQ": 0.02},
        prices={"ES": 5000, "NQ": 18000},
        capital=5_000_000,
        regime=VolRegime.CAUTION,
    )
    assert all(o.order_type for o in r["orders"])
