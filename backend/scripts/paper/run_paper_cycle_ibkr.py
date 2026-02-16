"""3-phase IBKR paper-trading cycle for CO→OC reversal futures.

Usage::

    python backend/scripts/paper/run_paper_cycle_ibkr.py \\
        --config backend/configs/cooc_reversal_futures.yaml \\
        --inputs data_cache/canonical_futures_daily.parquet \\
        --asof 2026-02-14 \\
        --phase open \\
        --mode noop
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sleeves.cooc_reversal_futures.config import COOCReversalConfig
from sleeves.cooc_reversal_futures.contract_master import CONTRACT_MASTER
from sleeves.cooc_reversal_futures.roll import active_contract_for_day, roll_week_flag
from sleeves.cooc_reversal_futures.sleeve import COOCReversalFuturesSleeve
from sleeves.cooc_reversal_futures.signal_mode import (
    SignalMode,
    heuristic_predictions,
    model_predictions,
    resolve_signal_mode,
)

from algaie.data.options.vrp_features import VolRegime
from algaie.trading.broker_ibkr import IBKRLiveBroker
from algaie.trading.orders import OrderIntent
from algaie.trading.paper_guards_futures import (
    GuardResult,
    PaperGuardConfig,
    apply_paper_guards,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------


def _load_config(config_path: str) -> dict:
    """Load YAML config."""
    import yaml  # type: ignore[import-untyped]

    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_sleeve_config(cfg: dict) -> COOCReversalConfig:
    """Build sleeve config from YAML section."""
    sc = cfg.get("sleeve", {})
    return COOCReversalConfig(
        universe=tuple(sc.get("universe", sc.get("universe_roots", ["ES", "NQ", "RTY", "YM"]))),
        lookback=sc.get("lookback", 20),
        gross_target=sc.get("gross_target", 0.8),
        caution_scale=sc.get("caution_scale", 0.5),
        net_cap=sc.get("net_cap", 0.1),
        max_contracts_per_instrument=sc.get("max_contracts_per_instrument", 20),
    )


def _build_guard_config(cfg: dict) -> PaperGuardConfig:
    """Build guard config from YAML paper section."""
    pc = cfg.get("paper", {})
    return PaperGuardConfig(
        max_orders_per_day=pc.get("max_orders_per_day", 20),
        max_contracts_per_order=pc.get("max_contracts_per_order", 5),
        max_contracts_per_instrument=pc.get("max_contracts_per_instrument", 10),
        max_gross_notional=pc.get("max_gross_notional", 5_000_000.0),
        roll_window_block=pc.get("roll_window_block", True),
    )


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _output_dir(cfg: dict, asof: date, phase: str) -> Path:
    base = Path(cfg.get("output", {}).get("base_dir", "data_lake/futures/paper"))
    out = base / asof.isoformat() / phase
    out.mkdir(parents=True, exist_ok=True)
    return out


def _save_intents(intents: List[OrderIntent], path: Path) -> None:
    records = [
        {
            "asof": str(i.asof),
            "ticker": i.ticker,
            "quantity": i.quantity,
            "side": i.side,
            "reason": i.reason,
            "limit_price": i.limit_price,
            "client_order_id": i.client_order_id,
        }
        for i in intents
    ]
    path.write_text(json.dumps(records, indent=2), encoding="utf-8")


def _save_orders_parquet(orders: list, path: Path) -> None:
    if not orders:
        pd.DataFrame().to_parquet(path)
        return
    records = [
        {
            "asof": str(o.asof),
            "ticker": o.ticker,
            "quantity": o.quantity,
            "side": o.side,
            "status": o.status,
            "fill_price": o.fill_price,
            "broker_order_id": o.broker_order_id,
            "client_order_id": o.client_order_id,
        }
        for o in orders
    ]
    pd.DataFrame(records).to_parquet(path, index=False)


# ---------------------------------------------------------------------------
# Phase: OPEN
# ---------------------------------------------------------------------------


def phase_open(
    cfg: dict,
    inputs_path: str,
    asof: date,
    mode: str,
) -> None:
    """Build signals, create entry order intents, apply guards, submit."""
    logger.info("═══ PHASE OPEN ═══ asof=%s mode=%s", asof, mode)

    sleeve_cfg = _build_sleeve_config(cfg)
    guard_cfg = _build_guard_config(cfg)
    roots = list(sleeve_cfg.universe)
    out = _output_dir(cfg, asof, "open")

    # -----------------------------------------------------------------------
    # Load inputs
    # -----------------------------------------------------------------------
    df = pd.read_parquet(inputs_path)
    logger.info("Loaded %d rows from %s", len(df), inputs_path)

    # -----------------------------------------------------------------------
    # Active contracts + roll window check
    # -----------------------------------------------------------------------
    active_contracts: Dict[str, str] = {}
    for root in roots:
        spec = CONTRACT_MASTER[root]
        active_contracts[root] = active_contract_for_day(root, asof, spec)
    is_roll = roll_week_flag(asof)
    logger.info("Active contracts: %s  roll_window=%s", active_contracts, is_roll)

    # -----------------------------------------------------------------------
    # Signal mode resolution
    # -----------------------------------------------------------------------
    pack_dir = cfg.get("model", {}).get("pack_dir")
    signal_mode_override = cfg.get("signal", {}).get("mode", "auto")

    if signal_mode_override == "heuristic":
        signal_mode = SignalMode.HEURISTIC
    elif signal_mode_override == "model":
        signal_mode = resolve_signal_mode(pack_dir, require_model_sanity=True)
        if signal_mode == SignalMode.HEURISTIC:
            logger.warning("Requested model mode but pack failed sanity → falling back to heuristic")
    else:  # auto
        signal_mode = resolve_signal_mode(pack_dir, require_model_sanity=True)

    logger.info("Signal mode: %s", signal_mode.value)

    # -----------------------------------------------------------------------
    # Generate predictions
    # -----------------------------------------------------------------------
    if signal_mode == SignalMode.HEURISTIC:
        pred_mu = heuristic_predictions(df, roots)
    else:
        pred_mu = model_predictions(pack_dir, df, roots)

    # Sigma estimate: use rolling std from data
    pred_sigma: Dict[str, float] = {}
    for root in roots:
        subset = df[df["root"] == root]
        if "rolling_std_ret_co" in subset.columns and not subset.empty:
            pred_sigma[root] = float(subset["rolling_std_ret_co"].iloc[-1])
        elif "ret_co" in subset.columns and len(subset) >= 5:
            pred_sigma[root] = float(subset["ret_co"].tail(20).std())
        else:
            pred_sigma[root] = 0.01  # fallback

    logger.info("Predictions (mu): %s", {k: f"{v:.6f}" for k, v in pred_mu.items()})
    logger.info("Predictions (sigma): %s", {k: f"{v:.6f}" for k, v in pred_sigma.items()})

    # -----------------------------------------------------------------------
    # Build daily orders via sleeve
    # -----------------------------------------------------------------------
    sleeve = COOCReversalFuturesSleeve(config=sleeve_cfg)
    prices = {}
    for root in roots:
        subset = df[df["root"] == root]
        if not subset.empty and "close" in subset.columns:
            prices[root] = float(subset["close"].iloc[-1])
        else:
            logger.warning("No close price for %s — using 0", root)
            prices[root] = 0.0

    capital = cfg.get("paper", {}).get("capital", 1_000_000.0)

    # Determine regime (default NORMAL_CARRY for paper)
    regime_str = cfg.get("paper", {}).get("regime_override", "NORMAL_CARRY")
    regime = VolRegime(regime_str.lower().replace("_", "_"))
    # VolRegime values: normal_carry, caution, crash_risk
    try:
        regime = VolRegime(regime_str.lower())
    except ValueError:
        regime = VolRegime.NORMAL_CARRY

    result = sleeve.build_daily_orders(
        date_t=asof,
        pred_mu=pred_mu,
        pred_sigma=pred_sigma,
        prices=prices,
        capital=capital,
        regime=regime,
    )

    logger.info("Sleeve result: weights=%s contracts=%s", result["weights"], result["contracts"])

    # -----------------------------------------------------------------------
    # Convert to OrderIntents (using active_contract tickers)
    # -----------------------------------------------------------------------
    intents: List[OrderIntent] = []
    for root, qty in result["contracts"].items():
        if qty == 0:
            continue
        side = "buy" if qty > 0 else "sell"
        ticker = active_contracts.get(root, root)
        intents.append(
            OrderIntent(
                asof=asof,
                ticker=ticker,
                quantity=abs(qty),
                side=side,
                reason=f"COOC open {signal_mode.value}",
                client_order_id=f"cooc_{asof.isoformat()}_{root}_open",
            )
        )

    logger.info("Generated %d order intents", len(intents))

    # -----------------------------------------------------------------------
    # Apply paper guards
    # -----------------------------------------------------------------------
    multipliers = {r: CONTRACT_MASTER[r].multiplier for r in roots}
    guard_result = apply_paper_guards(
        intents=intents,
        config=guard_cfg,
        regime=regime_str.upper(),
        multipliers=multipliers,
        reference_prices=prices,
        is_flatten=False,
        is_roll_window=is_roll,
    )

    logger.info("Guard result: passed=%s violations=%s", guard_result.passed, guard_result.violations)

    # -----------------------------------------------------------------------
    # Persist intents + guard report
    # -----------------------------------------------------------------------
    _save_intents(intents, out / "intents_open.json")
    pd.DataFrame([i.__dict__ for i in guard_result.filtered_intents]).to_parquet(
        out / "order_intents_open.parquet", index=False
    ) if guard_result.filtered_intents else pd.DataFrame().to_parquet(
        out / "order_intents_open.parquet", index=False
    )
    (out / "guard_report_open.json").write_text(
        json.dumps(guard_result.to_dict(), indent=2), encoding="utf-8"
    )

    # -----------------------------------------------------------------------
    # Submit or noop
    # -----------------------------------------------------------------------
    if mode == "noop":
        logger.info("NOOP mode — no orders submitted to broker")
        _save_orders_parquet([], out / "orders_open.parquet")
    elif mode == "ibkr":
        if not guard_result.filtered_intents:
            logger.warning("No intents after guards — nothing to submit")
            _save_orders_parquet([], out / "orders_open.parquet")
        else:
            broker = IBKRLiveBroker.from_env()
            orders = broker.submit_orders(guard_result.filtered_intents)
            _save_orders_parquet(orders, out / "orders_open.parquet")
            logger.info("Submitted %d orders to IBKR", len(orders))
            broker._disconnect()
    else:
        raise ValueError(f"Unknown mode: {mode}")

    logger.info("Phase OPEN complete → artifacts in %s", out)


# ---------------------------------------------------------------------------
# Phase: CLOSE
# ---------------------------------------------------------------------------


def phase_close(
    cfg: dict,
    asof: date,
    mode: str,
) -> None:
    """Flatten all sleeve positions at end of day."""
    logger.info("═══ PHASE CLOSE ═══ asof=%s mode=%s", asof, mode)

    sleeve_cfg = _build_sleeve_config(cfg)
    guard_cfg = _build_guard_config(cfg)
    out = _output_dir(cfg, asof, "close")
    sleeve = COOCReversalFuturesSleeve(config=sleeve_cfg)

    if mode == "noop":
        # In noop mode, check if we have open intents from earlier
        open_dir = _output_dir(cfg, asof, "open")
        intents_path = open_dir / "order_intents_open.parquet"
        if intents_path.exists():
            df = pd.read_parquet(intents_path)
            if not df.empty:
                positions = {row["ticker"]: int(row["quantity"] * (1 if row["side"] == "buy" else -1))
                             for _, row in df.iterrows()}
            else:
                positions = {}
        else:
            positions = {}
        logger.info("NOOP mode — simulated positions: %s", positions)
    elif mode == "ibkr":
        broker = IBKRLiveBroker.from_env()
        broker_positions = broker.get_positions()
        # Filter to sleeve instruments only
        sleeve_roots = set(sleeve_cfg.universe)
        positions = {}
        for pos in broker_positions:
            # Check if this position belongs to our sleeve
            for root in sleeve_roots:
                if pos.ticker.startswith(root):
                    positions[pos.ticker] = int(pos.quantity)
                    break
        logger.info("IBKR positions for sleeve: %s", positions)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Build flatten intents
    flatten_result = sleeve.force_eod_flatten(positions)
    intents: List[OrderIntent] = []
    for fo in flatten_result:
        # flatten_orders returns FuturesOrder(symbol, qty, side)
        ticker = fo.symbol
        qty = abs(fo.qty)
        side = fo.side.lower()
        if qty == 0:
            continue
        intents.append(
            OrderIntent(
                asof=asof,
                ticker=ticker,
                quantity=qty,
                side=side,
                reason="COOC EOD flatten",
                client_order_id=f"cooc_{asof.isoformat()}_{ticker}_close",
            )
        )

    logger.info("Flatten intents: %d", len(intents))

    # Minimal guards (never block flatten)
    guard_result = apply_paper_guards(
        intents=intents,
        config=guard_cfg,
        is_flatten=True,
    )

    # Persist
    _save_intents(intents, out / "intents_close.json")

    if mode == "noop":
        _save_orders_parquet([], out / "orders_close.parquet")
    elif mode == "ibkr":
        if intents:
            orders = broker.submit_orders(guard_result.filtered_intents)
            _save_orders_parquet(orders, out / "orders_close.parquet")
            logger.info("Submitted %d flatten orders", len(orders))
            broker._disconnect()
        else:
            _save_orders_parquet([], out / "orders_close.parquet")

    logger.info("Phase CLOSE complete → artifacts in %s", out)


# ---------------------------------------------------------------------------
# Phase: RECONCILE
# ---------------------------------------------------------------------------


def phase_reconcile(
    cfg: dict,
    asof: date,
    mode: str,
) -> None:
    """Reconcile fills with intents and report."""
    logger.info("═══ PHASE RECONCILE ═══ asof=%s mode=%s", asof, mode)

    out = _output_dir(cfg, asof, "reconcile")

    # Load open/close intents
    open_dir = _output_dir(cfg, asof, "open")
    close_dir = _output_dir(cfg, asof, "close")

    open_intents = _load_json_safe(open_dir / "intents_open.json")
    close_intents = _load_json_safe(close_dir / "intents_close.json")
    open_orders = _load_parquet_safe(open_dir / "orders_open.parquet")
    close_orders = _load_parquet_safe(close_dir / "orders_close.parquet")

    if mode == "ibkr":
        from backend.app.execution.reconcile_futures import reconcile_day

        broker = IBKRLiveBroker.from_env()

        # Get fills for today
        fills = broker.get_fills()
        positions = broker.get_positions()
        broker._disconnect()

        # Persist fills
        fill_records = [
            {
                "asof": str(f.asof),
                "ticker": f.ticker,
                "quantity": f.quantity,
                "price": f.price,
                "side": f.side,
                "order_id": f.order_id,
                "commission": f.commission,
                "execution_time": str(f.execution_time) if f.execution_time else None,
            }
            for f in fills
        ]
        pd.DataFrame(fill_records).to_parquet(out / f"fills_{asof.isoformat()}.parquet", index=False)

        # Run reconciliation
        report = reconcile_day(
            asof=asof,
            open_intents=open_intents,
            close_intents=close_intents,
            open_orders=open_orders,
            close_orders=close_orders,
            fills=fill_records,
            positions=[{"ticker": p.ticker, "quantity": p.quantity, "avg_cost": p.avg_cost} for p in positions],
        )

        (out / f"reconciliation_{asof.isoformat()}.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        logger.info("Reconciliation complete: %s", json.dumps(report, indent=2)[:500])

    elif mode == "noop":
        # Noop reconciliation: just summarize what we have
        report = {
            "asof": asof.isoformat(),
            "mode": "noop",
            "open_intents_count": len(open_intents) if open_intents else 0,
            "close_intents_count": len(close_intents) if close_intents else 0,
            "open_orders_count": len(open_orders) if open_orders is not None else 0,
            "close_orders_count": len(close_orders) if close_orders is not None else 0,
            "note": "noop mode — no broker fills to reconcile",
        }
        (out / f"reconciliation_{asof.isoformat()}.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        pd.DataFrame().to_parquet(out / f"fills_{asof.isoformat()}.parquet", index=False)

    logger.info("Phase RECONCILE complete → artifacts in %s", out)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_json_safe(path: Path) -> Optional[list]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def _load_parquet_safe(path: Path) -> Optional[pd.DataFrame]:
    if path.exists():
        df = pd.read_parquet(path)
        return df if not df.empty else None
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="IBKR paper-trading cycle for CO→OC reversal futures"
    )
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--inputs", default=None, help="Path to canonical futures daily parquet")
    parser.add_argument("--asof", required=True, help="YYYY-MM-DD")
    parser.add_argument("--phase", required=True, choices=["open", "close", "reconcile"])
    parser.add_argument("--mode", required=True, choices=["ibkr", "noop"])

    args = parser.parse_args()
    cfg = _load_config(args.config)
    asof = date.fromisoformat(args.asof)

    if args.phase == "open":
        if not args.inputs:
            raise ValueError("--inputs required for open phase")
        phase_open(cfg, args.inputs, asof, args.mode)
    elif args.phase == "close":
        phase_close(cfg, asof, args.mode)
    elif args.phase == "reconcile":
        phase_reconcile(cfg, asof, args.mode)


if __name__ == "__main__":
    main()
