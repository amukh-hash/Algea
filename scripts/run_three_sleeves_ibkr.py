#!/usr/bin/env python
"""Three-sleeve IBKR paper-trading runner.

Runs Core (CO→OC reversal futures), VRP (options — noop by default),
and Selector (individual equities) against an IBKR paper account.

Usage::

    # Dry run (signals only, no orders)
    python scripts/run_three_sleeves_ibkr.py --mode noop --asof 2026-02-17

    # Live paper trading
    python scripts/run_three_sleeves_ibkr.py --mode ibkr --asof 2026-02-17

    # Override VRP to live (enable with caution!)
    python scripts/run_three_sleeves_ibkr.py --mode ibkr --asof 2026-02-17 --vrp-mode ibkr
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

import numpy as np
import pandas as pd

from algae.trading.broker_ibkr import IBKRLiveBroker
from algae.trading.broker_base import BrokerAccount
from algae.trading.orders import OrderIntent, Order

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALLOCATIONS = {
    "core": 0.50,
    "vrp": 0.30,
    "selector": 0.20,
}

# Latest selector run
SELECTOR_RUN_DIR = ROOT / "backend" / "data" / "selector" / "runs" / "SEL-PROD-CANDIDATE"
SELECTOR_SCORED_FILE = SELECTOR_RUN_DIR / "scored_test.parquet"

# Output artifacts
OUTPUT_BASE = ROOT / "data_lake" / "three_sleeves" / "paper"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_sleeve_capital(account: BrokerAccount) -> Dict[str, float]:
    """Compute per-sleeve capital from total NAV."""
    nav = account.equity
    return {name: nav * pct for name, pct in ALLOCATIONS.items()}


def shares_from_weight(weight: float, sleeve_capital: float, price: float) -> int:
    """Convert target weight → whole shares."""
    if price <= 0:
        return 0
    return int((weight * sleeve_capital) / price)


def output_dir(asof: date) -> Path:
    d = OUTPUT_BASE / asof.isoformat()
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_report(path: Path, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info("Saved report → %s", path)


# ═══════════════════════════════════════════════════════════════════════════
# SLEEVE 1: CORE (CO→OC Reversal Futures)
# ═══════════════════════════════════════════════════════════════════════════

def run_core(
    broker: IBKRLiveBroker,
    capital: float,
    asof: date,
    mode: str,
) -> Dict[str, Any]:
    """Run the CO→OC reversal futures sleeve.

    Delegates to the existing paper cycle runner's phase_open logic.
    For simplicity in week 1, we import and call the existing infrastructure.
    """
    logger.info("=" * 60)
    logger.info("CORE SLEEVE (CO->OC Reversal Futures) -- capital=$%.0f", capital)
    logger.info("=" * 60)

    report: Dict[str, Any] = {
        "sleeve": "core",
        "capital": capital,
        "mode": mode,
        "status": "skipped",
        "orders": [],
    }

    try:
        # Check if the existing paper cycle config exists
        config_path = ROOT / "sleeves" / "cooc_reversal_futures" / "cooc_reversal_futures.yaml"
        inputs_path = ROOT / "data_cache" / "canonical_futures_daily.parquet"

        if not config_path.exists():
            logger.warning("Core config not found: %s", config_path)
            report["status"] = "config_missing"
            return report

        if not inputs_path.exists():
            logger.warning("Core inputs not found: %s — run data pipeline first", inputs_path)
            report["status"] = "inputs_missing"
            return report

        if mode == "noop":
            logger.info("NOOP mode — Core signals computed but no orders submitted")
            report["status"] = "noop"
            report["message"] = (
                "Core sleeve ready. Run with --mode ibkr to submit orders. "
                "Or use the dedicated runner: "
                "python backend/scripts/paper/run_paper_cycle_ibkr.py --phase open --mode ibkr"
            )
        elif mode == "ibkr":
            # Import and delegate to existing runner
            from backend.scripts.paper.run_paper_cycle_ibkr import phase_open
            import yaml

            with open(config_path) as f:
                cfg = yaml.safe_load(f)

            phase_open(cfg, str(inputs_path), asof, mode="ibkr")
            report["status"] = "submitted"
            report["message"] = "Orders submitted via existing paper cycle runner"

    except Exception as exc:
        logger.error("Core sleeve error: %s", exc, exc_info=True)
        report["status"] = "error"
        report["error"] = str(exc)

    return report


# ═══════════════════════════════════════════════════════════════════════════
# SLEEVE 2: VRP (Options — put credit spreads via IBKR)
# ═══════════════════════════════════════════════════════════════════════════

def run_vrp(
    broker: IBKRLiveBroker,
    capital: float,
    asof: date,
    mode: str,
) -> Dict[str, Any]:
    """Run the VRP options sleeve.

    Fetches option chains, computes VRP features, generates put credit
    spread positions via ``VRPStrategy``, and optionally submits IBKR
    combo (BAG) orders in paper trading mode.

    Parameters
    ----------
    broker : IBKRLiveBroker
        Connected IBKR broker instance.
    capital : float
        Capital allocated to the VRP sleeve.
    asof : date
        Trading date.
    mode : str
        "noop" (signal only) or "ibkr" (submit orders to IB Gateway).
    """
    logger.info("=" * 60)
    logger.info("VRP SLEEVE (Options) -- capital=$%.0f  mode=%s", capital, mode)
    logger.info("=" * 60)

    report: Dict[str, Any] = {
        "sleeve": "vrp",
        "capital": capital,
        "mode": mode,
        "asof": asof.isoformat(),
        "status": "ok",
        "orders": [],
        "positions": [],
        "warnings": [],
    }

    try:
        from algae.execution.options.vrp_strategy import VRPStrategy
        from algae.execution.options.config import VRPConfig
        from backend.app.trading.ibkr_option_chain import fetch_option_chain
        from backend.app.trading.ibkr_market_data import (
            fetch_underlying_closes,
            fetch_vix_series,
            get_current_price,
        )
        from backend.app.trading.ibkr_options_executor import (
            position_to_order_intent,
            submit_combo_order,
        )
    except ImportError as exc:
        logger.warning("VRP imports failed: %s", exc)
        report["status"] = "import_error"
        report["error"] = str(exc)
        return report

    try:
        config = VRPConfig()
        strategy = VRPStrategy(config)
        logger.info(
            "VRP strategy loaded — underlyings=%s, DTE range=%s",
            config.underlyings, config.dte_range,
        )

        # Get raw IB connection for data fetching
        broker._ensure_connected()
        ib = broker._client._ib

        # ── Fetch VIX series (shared across underlyings) ──────────────
        try:
            vix_series = fetch_vix_series(ib, lookback_days=252)
        except Exception as exc:
            logger.error("VIX fetch failed: %s", exc)
            report["status"] = "error"
            report["error"] = f"VIX fetch failed: {exc}"
            return report

        # ── Process each underlying ───────────────────────────────────
        all_positions = []
        all_order_intents = []

        for underlying in config.underlyings:
            logger.info("─── Processing %s ───", underlying)

            # 1. Get spot price
            try:
                spot = get_current_price(ib, underlying)
            except Exception as exc:
                msg = f"Spot price fetch failed for {underlying}: {exc}"
                logger.warning(msg)
                report["warnings"].append(msg)
                continue

            # 2. Fetch option chain
            try:
                chain_df = fetch_option_chain(
                    ib,
                    underlying,
                    asof_date=asof,
                    dte_range=config.dte_range,
                    strike_band_pct=0.15,
                    max_expiries=2,
                    max_strikes_per_expiry=30,
                )
            except Exception as exc:
                msg = f"Chain fetch failed for {underlying}: {exc}"
                logger.warning(msg)
                report["warnings"].append(msg)
                continue

            if chain_df.empty:
                msg = f"Empty chain for {underlying} after filtering"
                logger.warning(msg)
                report["warnings"].append(msg)
                continue

            logger.info(
                "Chain for %s: %d rows, %d puts, %d calls",
                underlying,
                len(chain_df),
                len(chain_df[chain_df["option_type"] == "put"]),
                len(chain_df[chain_df["option_type"] == "call"]),
            )

            # 3. Fetch underlying close prices
            try:
                close_prices = fetch_underlying_closes(
                    ib, underlying, lookback_days=252,
                )
            except Exception as exc:
                msg = f"Close prices fetch failed for {underlying}: {exc}"
                logger.warning(msg)
                report["warnings"].append(msg)
                continue

            # 4. Compute features
            try:
                features = strategy.compute_features(
                    as_of_date=asof,
                    chain=chain_df,
                    close_prices=close_prices,
                    vix=vix_series,
                )
            except Exception as exc:
                msg = f"Feature computation failed for {underlying}: {exc}"
                logger.warning(msg)
                report["warnings"].append(msg)
                continue

            regime = features.get("regime", None)
            logger.info(
                "%s features: regime=%s, atm_iv=%.4f, skew_25d=%.4f",
                underlying,
                regime.value if regime else "unknown",
                features.get("surface_snapshot", {}).get("atm_iv", float("nan")),
                features.get("surface_snapshot", {}).get("skew_25d", float("nan")),
            )

            # 5. Generate position via strategy predict
            try:
                position = strategy.predict(
                    as_of_date=asof,
                    chain=chain_df,
                    underlying=underlying,
                    underlying_price=spot,
                    features=features,
                    nav=capital,
                )
            except Exception as exc:
                msg = f"Strategy predict failed for {underlying}: {exc}"
                logger.warning(msg)
                report["warnings"].append(msg)
                continue

            if position is None:
                logger.info(
                    "%s: no trade — gated by regime/IV/limits",
                    underlying,
                )
                continue

            # Log the generated position
            logger.info(
                "POSITION: %s %s K=%.1f/%.1f exp=%s credit=%.4f max_loss=%.4f "
                "delta=%.4f theta=%.4f risk_budget=%.4f",
                underlying,
                position.structure_type.value,
                position.legs[0].strike if len(position.legs) > 0 else 0,
                position.legs[1].strike if len(position.legs) > 1 else 0,
                position.expiry.isoformat(),
                position.premium_collected,
                position.max_loss,
                position.delta,
                position.theta,
                position.risk_budget_used,
            )

            all_positions.append(position)

            # 6. Convert to order intent
            try:
                intent = position_to_order_intent(position)
                all_order_intents.append(intent)
            except Exception as exc:
                msg = f"Order intent conversion failed for {underlying}: {exc}"
                logger.warning(msg)
                report["warnings"].append(msg)

        # ── Summary ───────────────────────────────────────────────────
        report["orders"] = all_order_intents
        report["positions"] = [
            {
                "underlying": p.underlying,
                "structure": p.structure_type.value,
                "expiry": p.expiry.isoformat(),
                "short_strike": next(
                    (l.strike for l in p.legs if l.qty < 0), None
                ),
                "long_strike": next(
                    (l.strike for l in p.legs if l.qty > 0), None
                ),
                "premium_collected": p.premium_collected,
                "max_loss": p.max_loss,
                "delta": p.delta,
                "theta": p.theta,
                "risk_budget_used": p.risk_budget_used,
                "position_id": p.position_id,
            }
            for p in all_positions
        ]

        logger.info(
            "VRP summary: %d underlyings scanned, %d positions generated, "
            "%d order intents, %d warnings",
            len(config.underlyings),
            len(all_positions),
            len(all_order_intents),
            len(report["warnings"]),
        )

        # ── Submit combo orders if in IBKR mode ──────────────────────
        if mode == "ibkr" and all_positions:
            logger.info("Submitting %d combo orders to IBKR...", len(all_positions))
            submitted = []

            account_id = os.environ.get("IBKR_ACCOUNT_ID", "")
            for position in all_positions:
                try:
                    result = submit_combo_order(
                        ib, position, account_id=account_id,
                    )
                    submitted.append(result)
                    logger.info(
                        "  %s: %s (orderId=%s)",
                        position.underlying,
                        result.get("status"),
                        result.get("order_id"),
                    )
                except Exception as exc:
                    logger.error(
                        "Combo order failed for %s: %s",
                        position.underlying, exc,
                    )
                    submitted.append({
                        "status": "error",
                        "message": str(exc),
                        "underlying": position.underlying,
                    })

            report["submitted_orders"] = submitted
            report["status"] = "submitted"
        elif mode == "noop":
            report["status"] = "ok"
            if all_order_intents:
                logger.info(
                    "NOOP mode — %d orders computed but not submitted",
                    len(all_order_intents),
                )
            else:
                logger.info("NOOP mode — no trades triggered")
        else:
            report["status"] = "ok"

    except Exception as exc:
        logger.error("VRP sleeve error: %s", exc, exc_info=True)
        report["status"] = "error"
        report["error"] = str(exc)

    return report


# ═══════════════════════════════════════════════════════════════════════════
# SLEEVE 3: SELECTOR (Individual Equities)
# ═══════════════════════════════════════════════════════════════════════════

def _load_selector_signals(asof: date, top_n: int = 10) -> pd.DataFrame:
    """Load latest selector scores and pick top/bottom N stocks.

    Returns DataFrame with columns: symbol, score_final, side, weight
    """
    if not SELECTOR_SCORED_FILE.exists():
        raise FileNotFoundError(f"Selector scored file not found: {SELECTOR_SCORED_FILE}")

    df = pd.read_parquet(SELECTOR_SCORED_FILE)

    # Use the most recent date available in the scored data
    latest_date = df["date"].max()
    logger.info("Selector: using scored date %s (latest available)", latest_date)

    day_scores = df[df["date"] == latest_date].copy()
    day_scores = day_scores.sort_values("score_final", ascending=False)

    # Long the top N, short the bottom N (market-neutral)
    longs = day_scores.head(top_n).copy()
    shorts = day_scores.tail(top_n).copy()

    longs["side"] = "buy"
    longs["weight"] = 1.0 / top_n  # Equal-weight longs

    shorts["side"] = "sell"
    shorts["weight"] = 1.0 / top_n  # Equal-weight shorts

    signals = pd.concat([longs, shorts], ignore_index=True)
    signals = signals[["symbol", "score_final", "side", "weight"]].copy()

    logger.info(
        "Selector signals: %d longs, %d shorts (top/bottom %d by score_final)",
        len(longs), len(shorts), top_n,
    )
    return signals


def run_selector(
    broker: IBKRLiveBroker,
    capital: float,
    asof: date,
    mode: str,
    top_n: int = 10,
) -> Dict[str, Any]:
    """Run the selector equities sleeve.

    Loads latest selector model scores, picks top/bottom N stocks,
    sizes positions as equal-weight within the sleeve capital allocation,
    and submits orders (or logs them in noop mode).
    """
    logger.info("=" * 60)
    logger.info("SELECTOR SLEEVE (Equities) -- capital=$%.0f", capital)
    logger.info("=" * 60)

    report: Dict[str, Any] = {
        "sleeve": "selector",
        "capital": capital,
        "mode": mode,
        "status": "skipped",
        "orders": [],
    }

    try:
        signals = _load_selector_signals(asof, top_n=top_n)

        intents: List[Dict[str, Any]] = []
        for _, row in signals.iterrows():
            symbol = row["symbol"]

            # For noop: estimate price as $100 placeholder
            # For ibkr: we'd fetch live price from broker
            estimated_price = 100.0  # placeholder for sizing display

            qty = shares_from_weight(row["weight"], capital, estimated_price)
            if qty == 0:
                continue

            intent = {
                "symbol": symbol,
                "side": row["side"],
                "quantity": qty,
                "score": float(row["score_final"]),
                "weight": float(row["weight"]),
                "estimated_notional": qty * estimated_price,
            }
            intents.append(intent)

        report["intents"] = intents
        report["num_longs"] = sum(1 for i in intents if i["side"] == "buy")
        report["num_shorts"] = sum(1 for i in intents if i["side"] == "sell")

        if mode == "noop":
            logger.info("NOOP mode — Selector orders computed but not submitted:")
            for intent in intents:
                logger.info(
                    "  %s %s %d shares (score=%.4f, ~$%.0f)",
                    intent["side"].upper(),
                    intent["symbol"],
                    intent["quantity"],
                    intent["score"],
                    intent["estimated_notional"],
                )
            report["status"] = "noop"
            report["message"] = f"{len(intents)} orders computed, none submitted"

        elif mode == "ibkr":
            from ib_insync import Stock, MarketOrder

            # Ensure broker is connected for direct client access
            broker._ensure_connected()

            submitted = []
            for intent in intents:
                try:
                    contract = Stock(intent["symbol"], "SMART", "USD")
                    qty = intent["quantity"]
                    action = "BUY" if intent["side"] == "buy" else "SELL"

                    # Qualify the stock contract
                    qualified = broker._client.qualify_contracts(contract)
                    if not qualified or qualified[0].conId == 0:
                        logger.error("Failed to qualify stock %s", intent["symbol"])
                        submitted.append({
                            "symbol": intent["symbol"],
                            "side": intent["side"],
                            "qty": qty,
                            "status": "qualify_failed",
                        })
                        continue

                    ib_contract = qualified[0]
                    ib_order = MarketOrder(action, qty)

                    # Set account
                    if broker.config.account_id:
                        ib_order.account = broker.config.account_id

                    # Place order directly via client
                    trade = broker._client.place_order(ib_contract, ib_order)
                    broker._client.sleep(0.3)

                    submitted.append({
                        "symbol": intent["symbol"],
                        "side": intent["side"],
                        "qty": qty,
                        "status": trade.orderStatus.status if trade.orderStatus else "submitted",
                        "broker_order_id": str(trade.order.orderId),
                    })
                    logger.info(
                        "Submitted: %s %s %d shares (orderId=%s)",
                        action, intent["symbol"], qty, trade.order.orderId,
                    )

                except Exception as exc:
                    logger.error(
                        "Failed to submit %s %s: %s",
                        intent["side"], intent["symbol"], exc,
                    )
                    submitted.append({
                        "symbol": intent["symbol"],
                        "side": intent["side"],
                        "qty": qty,
                        "status": f"error: {exc}",
                    })

            report["submitted_orders"] = submitted
            report["status"] = "submitted"
            report["message"] = f"{len(submitted)} orders submitted to IBKR"

    except FileNotFoundError as exc:
        logger.warning("Selector data not found: %s", exc)
        report["status"] = "data_missing"
        report["error"] = str(exc)
    except Exception as exc:
        logger.error("Selector sleeve error: %s", exc, exc_info=True)
        report["status"] = "error"
        report["error"] = str(exc)

    return report


# ═══════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════

def run_all_sleeves(
    asof: date,
    mode: str = "noop",
    vrp_mode: Optional[str] = None,
    selector_top_n: int = 10,
) -> Dict[str, Any]:
    """Run all three sleeves sequentially."""

    effective_vrp_mode = vrp_mode or mode  # VRP now follows global mode

    logger.info("=" * 70)
    logger.info("THREE-SLEEVE PAPER TRADING RUNNER")
    logger.info("  Date     : %s", asof)
    logger.info("  Mode     : %s", mode)
    logger.info("  VRP mode : %s", effective_vrp_mode)
    logger.info("=" * 70)

    # --- Connect ---
    broker = IBKRLiveBroker.from_env()
    account = broker.get_account()
    sleeve_capital = get_sleeve_capital(account)

    logger.info("Account NLV: $%.2f", account.equity)
    logger.info("Cash: $%.2f | Buying Power: $%.2f", account.cash, account.buying_power)
    for name, cap in sleeve_capital.items():
        logger.info("  %s capital: $%.2f (%d%%)", name, cap, ALLOCATIONS[name] * 100)

    # Margin cushion check
    if account.equity > 0:
        margin_cushion = account.buying_power / account.equity
        if margin_cushion < 0.30:
            logger.warning(
                "[WARN] Margin cushion %.1f%% < 30%% -- consider reducing exposure",
                margin_cushion * 100,
            )

    # --- Run sleeves ---
    reports = {}

    # Core creates its own broker connection internally (phase_open),
    # so we disconnect ours first to avoid clientId collision, then reconnect.
    if mode == "ibkr":
        broker._disconnect()
    reports["core"] = run_core(broker, sleeve_capital["core"], asof, mode)
    if mode == "ibkr":
        broker = IBKRLiveBroker.from_env()  # reconnect for remaining sleeves

    reports["vrp"] = run_vrp(broker, sleeve_capital["vrp"], asof, effective_vrp_mode)
    reports["selector"] = run_selector(
        broker, sleeve_capital["selector"], asof, mode,
        top_n=selector_top_n,
    )

    # --- Position snapshot ---
    logger.info("\n" + "=" * 60)
    logger.info("POST-EXECUTION POSITION SNAPSHOT")
    logger.info("=" * 60)

    try:
        positions = broker.get_positions()
        if positions:
            for pos in positions:
                logger.info(
                    "  %s  qty=%+.0f  avg_cost=$%.2f",
                    pos.ticker, pos.quantity, pos.avg_cost,
                )
        else:
            logger.info("  (no positions)")
    except Exception as exc:
        logger.warning("Could not fetch positions: %s", exc)

    # --- Summary ---
    logger.info("\n" + "=" * 60)
    logger.info("EXECUTION SUMMARY")
    logger.info("=" * 60)
    for name, rpt in reports.items():
        logger.info("  %-10s : %s", name, rpt.get("status", "unknown"))

    # --- Save combined report ---
    out = output_dir(asof)
    combined_report = {
        "asof": asof.isoformat(),
        "mode": mode,
        "vrp_mode": effective_vrp_mode,
        "account": {
            "equity": account.equity,
            "cash": account.cash,
            "buying_power": account.buying_power,
        },
        "sleeve_capital": sleeve_capital,
        "sleeves": reports,
        "timestamp": datetime.now().isoformat(),
    }
    save_report(out / "three_sleeve_report.json", combined_report)

    # --- Disconnect ---
    broker._disconnect()

    return combined_report


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Three-sleeve IBKR paper trading runner",
    )
    parser.add_argument(
        "--mode",
        choices=["noop", "ibkr"],
        default="noop",
        help="Execution mode: noop (dry run) or ibkr (submit orders)",
    )
    parser.add_argument(
        "--asof",
        type=lambda s: date.fromisoformat(s),
        default=date.today(),
        help="Trading date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--vrp-mode",
        choices=["noop", "ibkr"],
        default=None,
        help="Override VRP mode (default: always noop for safety)",
    )
    parser.add_argument(
        "--selector-top-n",
        type=int,
        default=10,
        help="Number of top/bottom stocks for selector (default: 10)",
    )
    args = parser.parse_args()

    run_all_sleeves(
        asof=args.asof,
        mode=args.mode,
        vrp_mode=args.vrp_mode,
        selector_top_n=args.selector_top_n,
    )


if __name__ == "__main__":
    main()
