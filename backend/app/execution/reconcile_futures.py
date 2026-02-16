"""Futures reconciliation for IBKR paper trading.

Joins open/close intents with broker orders and fills to produce
a machine-readable reconciliation report.
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def reconcile_day(
    asof: date,
    open_intents: Optional[List[dict]],
    close_intents: Optional[List[dict]],
    open_orders: Optional[Any],  # DataFrame or None
    close_orders: Optional[Any],  # DataFrame or None
    fills: List[dict],
    positions: List[dict],
) -> Dict[str, Any]:
    """Reconcile a day's trading activity.

    Parameters
    ----------
    asof
        Trading date.
    open_intents, close_intents
        Lists of intent dicts from phase open/close.
    open_orders, close_orders
        DataFrames of broker order responses (may be None or empty).
    fills
        List of fill dicts with ticker/qty/price/side/order_id/commission.
    positions
        List of end-of-day position dicts with ticker/quantity/avg_cost.

    Returns
    -------
    dict
        Reconciliation report.
    """
    n_open_intents = len(open_intents) if open_intents else 0
    n_close_intents = len(close_intents) if close_intents else 0
    n_fills = len(fills)

    # -----------------------------------------------------------------------
    # Fill coverage
    # -----------------------------------------------------------------------
    open_intent_tickers = {i["ticker"] for i in (open_intents or [])}
    close_intent_tickers = {i["ticker"] for i in (close_intents or [])}
    filled_tickers = {f["ticker"] for f in fills}

    open_fill_coverage = (
        len(open_intent_tickers & filled_tickers) / len(open_intent_tickers)
        if open_intent_tickers
        else 1.0
    )
    close_fill_coverage = (
        len(close_intent_tickers & filled_tickers) / len(close_intent_tickers)
        if close_intent_tickers
        else 1.0
    )

    # -----------------------------------------------------------------------
    # Partial fill detection
    # -----------------------------------------------------------------------
    partial_fills: List[dict] = []
    for intent in (open_intents or []) + (close_intents or []):
        ticker = intent["ticker"]
        intended_qty = abs(intent["quantity"])
        filled_qty = sum(
            abs(f["quantity"])
            for f in fills
            if f["ticker"] == ticker and f["side"] == intent["side"]
        )
        if 0 < filled_qty < intended_qty:
            partial_fills.append({
                "ticker": ticker,
                "intended_qty": intended_qty,
                "filled_qty": filled_qty,
                "shortfall": intended_qty - filled_qty,
            })

    # -----------------------------------------------------------------------
    # Residual positions (should be zero for intraday strategy)
    # -----------------------------------------------------------------------
    residual_positions = [
        p for p in positions if abs(p.get("quantity", 0)) > 0
    ]

    # -----------------------------------------------------------------------
    # Contract mismatch detection
    # -----------------------------------------------------------------------
    contract_mismatches: List[dict] = []
    all_intent_tickers = open_intent_tickers | close_intent_tickers
    for f in fills:
        if f["ticker"] not in all_intent_tickers:
            # Check if it matches a root from our intents
            is_root_match = any(
                f["ticker"].startswith(t[:2]) for t in all_intent_tickers
            )
            if is_root_match:
                contract_mismatches.append({
                    "fill_ticker": f["ticker"],
                    "intended_tickers": sorted(all_intent_tickers),
                    "note": "filled contract differs from intended",
                })

    # -----------------------------------------------------------------------
    # Commission summary
    # -----------------------------------------------------------------------
    total_commission = sum(f.get("commission", 0.0) for f in fills)

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------
    report: Dict[str, Any] = {
        "asof": asof.isoformat(),
        "summary": {
            "open_intents": n_open_intents,
            "close_intents": n_close_intents,
            "total_fills": n_fills,
            "open_fill_coverage": round(open_fill_coverage, 4),
            "close_fill_coverage": round(close_fill_coverage, 4),
            "partial_fills": len(partial_fills),
            "residual_positions": len(residual_positions),
            "contract_mismatches": len(contract_mismatches),
            "total_commission": round(total_commission, 4),
        },
        "details": {
            "partial_fills": partial_fills,
            "residual_positions": residual_positions,
            "contract_mismatches": contract_mismatches,
        },
        "status": "CLEAN" if (
            not partial_fills
            and not residual_positions
            and not contract_mismatches
        ) else "ISSUES_FOUND",
    }

    logger.info(
        "Reconciliation: fills=%d, coverage=%.1f%%/%.1f%%, partials=%d, residuals=%d, mismatches=%d",
        n_fills,
        open_fill_coverage * 100,
        close_fill_coverage * 100,
        len(partial_fills),
        len(residual_positions),
        len(contract_mismatches),
    )

    return report
