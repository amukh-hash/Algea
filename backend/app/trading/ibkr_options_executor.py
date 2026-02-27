"""IBKR options executor — converts DerivativesPosition to order intents and combo orders.

Provides:
    position_to_order_intent(pos) → dict   (orchestrator-compatible target format)
    submit_combo_order(ib, pos, ...)        (IBKR BAG order submission)
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def position_to_order_intent(pos) -> Dict[str, Any]:  # DerivativesPosition
    """Convert a ``DerivativesPosition`` into an orchestrator-compatible order intent.

    The output dict follows the same conventions as equity sleeve targets so
    the orchestrator router and UI can render them uniformly.

    Parameters
    ----------
    pos : DerivativesPosition
        A derivatives position from ``VRPStrategy.predict()``.

    Returns
    -------
    dict
        Order intent with keys: symbol, side, type, structure, legs, metadata.

    Raises
    ------
    ValueError
        If the position does not have exactly 2 legs (put credit spread).
    """
    if len(pos.legs) != 2:
        raise ValueError(
            f"Expected exactly 2 legs for put credit spread, got {len(pos.legs)}"
        )

    short_leg = None
    long_leg = None
    for leg in pos.legs:
        if leg.qty < 0:
            short_leg = leg
        elif leg.qty > 0:
            long_leg = leg

    if short_leg is None or long_leg is None:
        raise ValueError(
            f"Cannot identify short/long legs in position {pos.position_id}"
        )

    return {
        "symbol": pos.underlying,
        "side": "sell",  # net credit spread → sell-side intent
        "type": "COMBO",
        "structure": pos.structure_type.value,
        "qty": 1,
        "expiry": pos.expiry.isoformat(),
        "limit_price": round(pos.premium_collected, 2),
        "legs": [
            {
                "option_type": short_leg.option_type,
                "strike": short_leg.strike,
                "qty": short_leg.qty,
                "side": short_leg.side,
                "entry_mid": short_leg.entry_price_mid,
            },
            {
                "option_type": long_leg.option_type,
                "strike": long_leg.strike,
                "qty": long_leg.qty,
                "side": long_leg.side,
                "entry_mid": long_leg.entry_price_mid,
            },
        ],
        "metadata": {
            "position_id": pos.position_id,
            "premium_collected": pos.premium_collected,
            "max_loss": pos.max_loss,
            "multiplier": pos.multiplier,
            "delta": pos.delta,
            "theta": pos.theta,
            "vega": pos.vega,
            "gamma": pos.gamma,
            "risk_budget_used": pos.risk_budget_used,
            "strategy_tags": pos.strategy_tags,
        },
    }


def submit_combo_order(
    ib,  # ib_insync.IB
    pos,  # DerivativesPosition
    *,
    account_id: str,
    limit_price: Optional[float] = None,
) -> Dict[str, Any]:
    """Submit a put credit spread as an IBKR BAG (combo) order.

    Parameters
    ----------
    ib : ib_insync.IB
        Connected IB instance.
    pos : DerivativesPosition
        Spread position from ``VRPStrategy.predict()``.
    account_id : str
        IBKR account ID for order routing.
    limit_price : float, optional
        Override limit price (default: use strategy-derived net credit).

    Returns
    -------
    dict
        Submission result with keys: status, order_id, trade_status, message.
    """
    from ib_insync import (  # type: ignore[import-untyped]
        Contract,
        ComboLeg,
        LimitOrder,
        Option,
    )

    if not ib.isConnected():
        raise ConnectionError("IB Gateway is not connected")

    # Validate: exactly 2 legs, same expiry, same underlying
    if len(pos.legs) != 2:
        raise ValueError(f"Expected 2 legs, got {len(pos.legs)}")

    short_leg = None
    long_leg = None
    for leg in pos.legs:
        if leg.qty < 0:
            short_leg = leg
        elif leg.qty > 0:
            long_leg = leg

    if short_leg is None or long_leg is None:
        raise ValueError("Cannot identify short/long legs in position")

    # Format expiry for IBKR
    expiry_str = pos.expiry.strftime("%Y%m%d")

    # Build and qualify individual option contracts
    short_right = "P" if short_leg.option_type.lower() == "put" else "C"
    long_right = "P" if long_leg.option_type.lower() == "put" else "C"

    short_opt = Option(
        pos.underlying, expiry_str, short_leg.strike, short_right, "SMART"
    )
    long_opt = Option(
        pos.underlying, expiry_str, long_leg.strike, long_right, "SMART"
    )

    qualified = ib.qualifyContracts(short_opt, long_opt)
    if len(qualified) != 2:
        return {
            "status": "error",
            "message": f"Failed to qualify option contracts (got {len(qualified)})",
            "order_id": None,
            "trade_status": None,
        }

    q_short, q_long = qualified[0], qualified[1]

    if q_short.conId == 0 or q_long.conId == 0:
        return {
            "status": "error",
            "message": (
                f"Qualification returned invalid conId: "
                f"short={q_short.conId}, long={q_long.conId}"
            ),
            "order_id": None,
            "trade_status": None,
        }

    logger.info(
        "Qualified combo legs: %s K=%.1f (conId=%d) / %s K=%.1f (conId=%d)",
        short_right, short_leg.strike, q_short.conId,
        long_right, long_leg.strike, q_long.conId,
    )

    # Build combo BAG contract
    combo = Contract()
    combo.symbol = pos.underlying
    combo.secType = "BAG"
    combo.exchange = "SMART"
    combo.currency = "USD"
    combo.comboLegs = [
        ComboLeg(
            conId=q_short.conId,
            ratio=1,
            action="SELL",
            exchange="SMART",
        ),
        ComboLeg(
            conId=q_long.conId,
            ratio=1,
            action="BUY",
            exchange="SMART",
        ),
    ]

    # Determine limit price
    credit = limit_price if limit_price is not None else pos.premium_collected
    credit_rounded = round(credit, 2)

    # Build limit order (SELL the combo to collect credit)
    order = LimitOrder("SELL", 1, credit_rounded)
    if account_id:
        order.account = account_id

    logger.info(
        "Submitting combo: %s %s %s K=%.1f/%.1f exp=%s credit=%.2f",
        "SELL", pos.underlying, pos.structure_type.value,
        short_leg.strike, long_leg.strike, expiry_str, credit_rounded,
    )

    try:
        trade = ib.placeOrder(combo, order)
        ib.sleep(0.5)  # Allow order status to propagate

        trade_status = (
            trade.orderStatus.status if trade.orderStatus else "submitted"
        )
        order_id = str(trade.order.orderId) if trade.order else None

        logger.info(
            "Combo order placed: orderId=%s status=%s",
            order_id, trade_status,
        )

        return {
            "status": "ok",
            "order_id": order_id,
            "trade_status": trade_status,
            "message": (
                f"Put credit spread submitted: "
                f"{pos.underlying} K={short_leg.strike}/{long_leg.strike} "
                f"exp={expiry_str} credit={credit_rounded:.2f}"
            ),
        }

    except Exception as exc:
        logger.error("Combo order submission failed: %s", exc, exc_info=True)
        return {
            "status": "error",
            "order_id": None,
            "trade_status": None,
            "message": f"Order submission failed: {exc}",
        }
