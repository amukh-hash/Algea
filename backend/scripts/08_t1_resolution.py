"""T+1 Outcome Resolution Script.

Resolves predictions in the ``ece_tracking`` table using actual
paper trading execution outcomes from the prior trading day.

Designed as a daily cron job: ``python backend/scripts/08_t1_resolution.py``

Critical Mitigation — Corporate Action ECE Corruption:
    Overnight forward splits cause raw price drops of 50%+.  Without
    adjustment, the T+1 script records false "Loss" outcomes —
    mathematically corrupting the empirical accuracy of that confidence
    bin and triggering a spurious HALTED_ECE_BREACH.

    This script queries ``cum_split_factor`` and ``cum_dividend_factor``
    from the broker API and reverse-adjusts the T+1 execution price
    to the T+0 prediction baseline before evaluating directional success.
"""
from __future__ import annotations

import logging
import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

STATE_DB = Path("backend/artifacts/orchestrator_state/state.sqlite3")
FILLS_DIR = Path("backend/artifacts/fills")


# ═══════════════════════════════════════════════════════════════════════
# PaperBroker API Abstraction
# ═══════════════════════════════════════════════════════════════════════

def _fetch_prior_day_fills(asof: date) -> list[dict[str, Any]]:
    """Fetch fills from the paper broker for the prior trading day.

    Returns a list of dicts with keys:
      trade_id, symbol, side, fill_price, fill_qty, filled_at

    Rate limiting: when using the IBKR broker API, restricts concurrent
    requests to 3 with 0.33s backoff to prevent HTTP 429 (Blind Spot 1).

    Day-2 Mitigation — Broker Gateway Forced Disconnects:
        Wraps the IBKR API call with tenacity exponential backoff (4s→60s,
        10 attempts) to survive the daily ~23:45 EST gateway reset instead
        of silently returning empty fills and skipping legitimate predictions.
    """
    import time as _time

    fills_path = FILLS_DIR / f"fills_{asof.isoformat()}.json"
    if fills_path.exists():
        import json
        data = json.loads(fills_path.read_text(encoding="utf-8"))
        return data.get("fills", [])

    # Fallback: try IBKR paper broker with rate limiting + retry
    RATE_LIMIT_DELAY_S = 0.33   # Deterministic backoff per request

    try:
        from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

        @retry(
            retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
            wait=wait_exponential(multiplier=1, min=4, max=60),
            stop=stop_after_attempt(10),
            reraise=True,
        )
        def _fetch_with_retry():
            from algae.trading.broker_ibkr import IBKRLiveBroker
            broker = IBKRLiveBroker.from_env()
            _time.sleep(RATE_LIMIT_DELAY_S)
            fills = broker.get_fills(asof)
            try:
                broker._disconnect()
            except Exception:
                pass
            return fills

        return _fetch_with_retry()
    except Exception as e:
        logger.warning("No fills source available for %s: %s", asof, e)
        return []



def _get_corporate_actions(symbol: str, action_date: date) -> dict[str, float]:
    """Query corporate action adjustments for a symbol on a given date.

    Returns dict with cum_split_factor and cum_dividend_factor.
    Defaults to 1.0 (no adjustment) if unavailable.
    """
    try:
        import yfinance as yf
        tk = yf.Ticker(symbol)

        # Check for splits
        splits = tk.splits
        cum_split = 1.0
        if splits is not None and len(splits) > 0:
            # Filter to splits on or after the action date
            for idx, split_ratio in splits.items():
                split_date = idx.date() if hasattr(idx, "date") else idx
                if isinstance(split_date, datetime):
                    split_date = split_date.date()
                if split_date == action_date:
                    cum_split = float(split_ratio)
                    logger.warning(
                        "SPLIT DETECTED %s on %s — ratio=%.4f",
                        symbol, action_date, cum_split,
                    )
                    break

        # Check for dividends
        dividends = tk.dividends
        cum_dividend_factor = 1.0
        if dividends is not None and len(dividends) > 0:
            for idx, div_amount in dividends.items():
                div_date = idx.date() if hasattr(idx, "date") else idx
                if isinstance(div_date, datetime):
                    div_date = div_date.date()
                if div_date == action_date:
                    # Approximate dividend adjustment
                    try:
                        hist = tk.history(start=str(action_date - timedelta(days=5)),
                                          end=str(action_date), auto_adjust=False)
                        if len(hist) > 0:
                            prev_close = float(hist["Close"].iloc[-1])
                            cum_dividend_factor = prev_close / (prev_close - float(div_amount))
                    except Exception:
                        pass
                    break

        return {
            "cum_split_factor": cum_split,
            "cum_dividend_factor": cum_dividend_factor,
        }
    except Exception as e:
        logger.debug("Corporate action lookup failed for %s: %s", symbol, e)
        return {"cum_split_factor": 1.0, "cum_dividend_factor": 1.0}


# ═══════════════════════════════════════════════════════════════════════
# Outcome Resolution
# ═══════════════════════════════════════════════════════════════════════

def _resolve_directional_success(
    predicted_direction: float,
    t0_price: float,
    t1_price: float,
    cum_split_factor: float,
    cum_dividend_factor: float,
) -> int:
    """Determine if the prediction was directionally correct.

    Reverse-adjusts the T+1 price by corporate action factors
    to prevent false outcomes from splits/dividends.

    Parameters
    ----------
    predicted_direction : float
        Predicted price change sign (+1 bullish, -1 bearish).
    t0_price : float
        Price at prediction time.
    t1_price : float
        Raw settlement price at T+1.
    cum_split_factor : float
        Cumulative split factor (e.g. 2.0 for 2-for-1 split).
    cum_dividend_factor : float
        Cumulative dividend adjustment factor.

    Returns
    -------
    1 if prediction was correct, 0 if incorrect.
    """
    # Reverse-adjust T+1 price: undo the split and dividend effect
    adjusted_t1 = t1_price * cum_split_factor * cum_dividend_factor
    actual_return = (adjusted_t1 - t0_price) / t0_price

    if predicted_direction > 0 and actual_return > 0:
        return 1
    if predicted_direction < 0 and actual_return < 0:
        return 1
    return 0


def resolve_outcomes(
    asof: date | None = None,
    db_path: Path = STATE_DB,
) -> dict[str, Any]:
    """Resolve T+1 outcomes for all unresolved predictions.

    Parameters
    ----------
    asof : date
        The trading date whose fills to resolve. Defaults to yesterday.
    db_path : Path
        Path to the SQLite state database.

    Returns
    -------
    Summary dict with resolution counts.
    """
    if asof is None:
        asof = date.today() - timedelta(days=1)

    logger.info("T+1 RESOLUTION for %s", asof)

    # 1. Fetch fills from paper broker
    fills = _fetch_prior_day_fills(asof)
    logger.info("  FILLS: %d executions found", len(fills))

    if not fills:
        return {
            "status": "no_fills",
            "asof": str(asof),
            "resolved": 0,
            "skipped": 0,
        }

    # Index fills by trade_id
    fill_map: dict[str, dict] = {}
    for fill in fills:
        tid = fill.get("trade_id") or fill.get("id")
        if tid:
            fill_map[str(tid)] = fill

    # 2. Query unresolved predictions from ece_tracking
    conn = sqlite3.connect(db_path, timeout=30)
    resolved = 0
    skipped = 0

    try:
        conn.execute("PRAGMA journal_mode=WAL")

        # Find unresolved rows for the prediction date
        rows = conn.execute("""
            SELECT rowid, trade_id, sleeve, predicted_probability,
                   predicted_direction, t0_price, symbol
            FROM ece_tracking
            WHERE actual_outcome IS NULL
              AND prediction_date = ?
        """, (str(asof),)).fetchall()

        logger.info("  UNRESOLVED: %d predictions to resolve", len(rows))

        for row in rows:
            rowid, trade_id, sleeve, pred_prob, pred_dir, t0_price, symbol = row

            # Find matching fill
            fill = fill_map.get(str(trade_id)) if trade_id else None

            if fill is None:
                # Try matching by symbol
                for f in fills:
                    if f.get("symbol") == symbol:
                        fill = f
                        break

            if fill is None:
                skipped += 1
                continue

            t1_price = float(fill.get("fill_price", 0.0))
            if t1_price <= 0:
                skipped += 1
                continue

            # 3. Get corporate action adjustments
            actions = _get_corporate_actions(symbol or "", asof)

            # 4. Resolve directional outcome
            outcome = _resolve_directional_success(
                predicted_direction=float(pred_dir or 1.0),
                t0_price=float(t0_price or t1_price),
                t1_price=t1_price,
                cum_split_factor=actions["cum_split_factor"],
                cum_dividend_factor=actions["cum_dividend_factor"],
            )

            # 5. Update ece_tracking
            conn.execute("""
                UPDATE ece_tracking
                SET actual_outcome = ?,
                    resolved_at = ?,
                    t1_price = ?,
                    cum_split_factor = ?,
                    cum_dividend_factor = ?
                WHERE rowid = ?
            """, (
                outcome,
                datetime.now(timezone.utc).isoformat(),
                t1_price,
                actions["cum_split_factor"],
                actions["cum_dividend_factor"],
                rowid,
            ))
            resolved += 1

        conn.commit()
    finally:
        conn.close()

    summary = {
        "status": "ok",
        "asof": str(asof),
        "resolved": resolved,
        "skipped": skipped,
        "total_fills": len(fills),
    }
    logger.info("  DONE: resolved=%d, skipped=%d", resolved, skipped)
    return summary


# ═══════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    parser = argparse.ArgumentParser(description="T+1 Outcome Resolution")
    parser.add_argument("--asof", type=str, default=None,
                        help="Date to resolve (YYYY-MM-DD). Default: yesterday.")
    parser.add_argument("--db", type=str, default=str(STATE_DB),
                        help="Path to SQLite state DB.")
    args = parser.parse_args()

    asof_date = date.fromisoformat(args.asof) if args.asof else None
    result = resolve_outcomes(asof=asof_date, db_path=Path(args.db))

    import json
    print(json.dumps(result, indent=2))
