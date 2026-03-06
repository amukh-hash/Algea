"""Phase-aware risk gateway and idempotent order routing.

Enforces unified Cash + Notional exposure limit across all sleeves,
DTE-0 assignment prevention (timezone-aware), and idempotent UPSERT of
order intents with delta-qty-based UUID routing.

Resolves F1 (core bypass), F3 (DTE-0 assignment), and F4 (idempotent routing).
"""
from __future__ import annotations

import logging
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from backend.app.core.schemas import ExecutionPhase, TargetIntent

logger = logging.getLogger(__name__)

EASTERN_TZ = ZoneInfo("America/New_York")


def validate_and_store_intents(
    db_path: Path | str,
    intents: list[TargetIntent],
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Validate risk limits and atomically persist intents.

    Runs inside ``EXCLUSIVE`` transaction isolation — either all intents
    pass risk checks and are stored, or zero are.

    Parameters
    ----------
    db_path : Path | str
        SQLite database path (``state.sqlite3``).
    intents : list[TargetIntent]
        Sleeve-generated position intents.
    now : datetime, optional
        Override for current time (for testing).  Defaults to
        ``datetime.now(EASTERN_TZ)``.

    Returns
    -------
    dict
        ``{"status": "ok", "exposure": {...}, "n_stored": int}``

    Raises
    ------
    RuntimeError
        On risk breach, paused state, or DTE-0 assignment risk.
    """
    now_est = now or datetime.now(EASTERN_TZ)

    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        conn.execute("BEGIN EXCLUSIVE")

        # ── F2: Check durable pause state ────────────────────────────
        row = conn.execute(
            "SELECT gross_exposure_cap, is_paused FROM app_control_state WHERE id=1"
        ).fetchone()
        if row is None:
            conn.execute("ROLLBACK")
            raise RuntimeError("app_control_state not initialized — run migrations")

        cap = float(row["gross_exposure_cap"])
        if bool(row["is_paused"]):
            conn.execute("ROLLBACK")
            raise RuntimeError("System is paused. Rejecting intent ingestion.")

        # ── F1: Unified exposure (Cash + Notional) ───────────────────
        cash_exp = sum(
            abs(i.target_weight)
            for i in intents
            if i.asset_class != "FUTURE"
        )
        not_exp = sum(
            abs(i.target_weight) * i.multiplier
            for i in intents
            if i.asset_class == "FUTURE"
        )
        total_exposure = cash_exp + not_exp

        if total_exposure > cap:
            conn.execute("ROLLBACK")
            raise RuntimeError(
                f"RISK BREACH: Total {total_exposure:.2f} > Cap {cap}"
            )

        # ── F3: Strict Timezone-Aware DTE-0 Flattening Rule ─────────
        for intent in intents:
            if (
                intent.asset_class == "OPTION"
                and intent.dte == 0
                and now_est.hour >= 15
                and intent.target_weight != 0
            ):
                conn.execute("ROLLBACK")
                raise RuntimeError(
                    f"FATAL: DTE-0 Assignment Risk on {intent.symbol}"
                )

        # ── F4: Idempotent UPSERT on logical PK ─────────────────────
        conn.executemany(
            """
            INSERT INTO order_intents
                (asof_date, execution_phase, sleeve, symbol,
                 target_weight, asset_class, multiplier)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(asof_date, execution_phase, sleeve, symbol)
            DO UPDATE SET
                target_weight = excluded.target_weight,
                status = 'PENDING'
            """,
            [
                (
                    i.asof_date,
                    i.execution_phase.value,
                    i.sleeve,
                    i.symbol,
                    i.target_weight,
                    i.asset_class,
                    i.multiplier,
                )
                for i in intents
            ],
        )
        conn.execute("COMMIT")

        logger.info(
            "Risk gateway accepted %d intents — exposure %.4f / %.4f cap",
            len(intents), total_exposure, cap,
        )
        return {
            "status": "ok",
            "n_stored": len(intents),
            "exposure": {
                "cash": round(cash_exp, 6),
                "notional": round(not_exp, 6),
                "total": round(total_exposure, 6),
                "cap": cap,
            },
        }
    except Exception:
        try:
            conn.execute("ROLLBACK")
        except Exception:
            pass
        raise
    finally:
        conn.close()


def route_phase_orders(
    db_path: Path | str,
    broker: Any,
    phase: ExecutionPhase,
    asof_date: str,
) -> dict[str, Any]:
    """Route pending intents for a specific execution phase.

    Pre-fetches live broker state (equity, positions) and computes
    discrete ``delta_qty`` for each symbol.  Deterministic UUID5 order
    IDs are derived from ``(asof_date, phase, symbol, delta_qty)`` —
    retries with the same delta produce identical broker refs (safely
    deduplicated); retries that scale a position produce a NEW delta
    and thus a NEW valid order.

    Parameters
    ----------
    db_path : Path | str
        SQLite database path.
    broker
        Broker protocol instance with ``get_account_equity()``,
        ``get_positions()``, ``get_price()``, ``place_order()``.
    phase : ExecutionPhase
        Which phase to route (e.g. ``AUCTION_OPEN``).
    asof_date : str
        Trading date (``YYYY-MM-DD``).

    Returns
    -------
    dict with ``status``, ``routed_count``, ``skipped_count``.
    """
    # Pre-fetch live state to calculate exact delta quantities
    equity = broker.get_account_equity()
    positions = broker.get_positions()  # Dict[symbol, current_qty]

    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    routed = 0
    skipped = 0

    try:
        conn.execute("BEGIN IMMEDIATE")

        pending = conn.execute(
            "SELECT * FROM order_intents "
            "WHERE status='PENDING' AND execution_phase=? AND asof_date=?",
            (phase.value, asof_date),
        ).fetchall()

        for row in pending:
            symbol = row["symbol"]
            target_weight = float(row["target_weight"])
            sleeve = row["sleeve"]
            multiplier = float(row["multiplier"])

            # Reconcile target weight into discrete share/contract delta
            price = broker.get_price(symbol)
            target_notional = target_weight * equity
            target_qty = int(target_notional / (price * multiplier))
            current_qty = positions.get(symbol, 0)
            delta_qty = target_qty - current_qty

            if delta_qty == 0:
                conn.execute(
                    "UPDATE order_intents SET status='SKIPPED' "
                    "WHERE asof_date=? AND execution_phase=? AND sleeve=? AND symbol=?",
                    (asof_date, phase.value, sleeve, symbol),
                )
                skipped += 1
                continue

            # F4: Deterministic UUID tied to the mutation intent (delta_qty)
            seed = f"{asof_date}:{phase.value}:{symbol}:{delta_qty}"
            broker_id = str(uuid.uuid5(uuid.NAMESPACE_OID, seed))

            try:
                broker.place_order(
                    client_order_id=broker_id,
                    symbol=symbol,
                    qty=delta_qty,
                )
                conn.execute(
                    "UPDATE order_intents SET status='ROUTED' "
                    "WHERE asof_date=? AND execution_phase=? AND sleeve=? AND symbol=?",
                    (asof_date, phase.value, sleeve, symbol),
                )
                routed += 1
            except Exception as exc:
                logger.error("Failed to route %s/%s: %s", symbol, sleeve, exc)
                conn.execute(
                    "UPDATE order_intents SET status='REJECTED' "
                    "WHERE asof_date=? AND execution_phase=? AND sleeve=? AND symbol=?",
                    (asof_date, phase.value, sleeve, symbol),
                )

        conn.execute("COMMIT")
    except Exception:
        try:
            conn.execute("ROLLBACK")
        except Exception:
            pass
        raise
    finally:
        conn.close()

    logger.info(
        "Phase %s routing: %d routed, %d skipped", phase.value, routed, skipped
    )
    return {"status": "ok", "routed_count": routed, "skipped_count": skipped}
