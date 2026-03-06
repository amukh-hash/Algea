"""ECE (Expected Calibration Error) Tracker & Circuit Breaker.

Tracks per-trade predicted probabilities and actual outcomes in the
``ece_tracking`` SQLite table.  Computes calibration error and triggers
``HALTED_ECE_BREACH`` when high-confidence bins exceed the threshold.

ECE Formula:
    ECE = Σ (N_bin / N_total) × |Empirical_Acc - Stated_Conf|

Halt Condition:
    ECE > 0.10 for high-confidence bins (≥0.80) with N_bin > 50
"""
from __future__ import annotations

import logging
import sqlite3
import uuid
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

STATE_DB = Path("backend/artifacts/orchestrator_state/state.sqlite3")


# ═══════════════════════════════════════════════════════════════════════
# Trade Recording
# ═══════════════════════════════════════════════════════════════════════

def record_prediction(
    sleeve: str,
    predicted_probability: float,
    confidence_bin: str,
    trade_id: str | None = None,
    db_path: Path = STATE_DB,
) -> str:
    """Record a new prediction into the ECE tracking table.

    Parameters
    ----------
    sleeve : str
        Originating sleeve identifier.
    predicted_probability : float
        Model self-reported confidence in [0, 1].
    confidence_bin : str
        Canonical bin string (e.g. ``'0.80-0.90'``).
    trade_id : str or None
        Unique trade identifier; auto-generated if None.
    db_path : Path
        Path to SQLite database.

    Returns
    -------
    str — the trade_id.
    """
    tid = trade_id or str(uuid.uuid4())

    conn = sqlite3.connect(db_path, timeout=30)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            INSERT INTO ece_tracking (trade_id, sleeve, confidence_bin, predicted_probability)
            VALUES (?, ?, ?, ?)
        """, (tid, sleeve, confidence_bin, predicted_probability))
        conn.commit()
    finally:
        conn.close()

    logger.debug("ECE_REC %s — sleeve=%s, prob=%.3f, bin=%s", tid[:8], sleeve, predicted_probability, confidence_bin)
    return tid


def resolve_outcome(
    trade_id: str,
    actual_outcome: int,
    db_path: Path = STATE_DB,
) -> None:
    """Resolve the T+1 actual outcome for a pending prediction.

    Parameters
    ----------
    trade_id : str
        The trade_id to update.
    actual_outcome : int
        1 (win), 0 (loss).
    """
    if actual_outcome not in (0, 1):
        raise ValueError(f"actual_outcome must be 0 or 1, got {actual_outcome}")

    conn = sqlite3.connect(db_path, timeout=30)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            UPDATE ece_tracking SET actual_outcome = ? WHERE trade_id = ?
        """, (actual_outcome, trade_id))
        conn.commit()
    finally:
        conn.close()


# ═══════════════════════════════════════════════════════════════════════
# ECE Computation
# ═══════════════════════════════════════════════════════════════════════

def compute_ece(
    sleeve: str | None = None,
    high_confidence_only: bool = False,
    min_bin_samples: int = 50,
    db_path: Path = STATE_DB,
) -> dict:
    """Compute ECE from resolved predictions in the database.

    ECE = Σ (N_bin / N_total) × |Empirical_Acc - Stated_Conf|

    Parameters
    ----------
    sleeve : str or None
        Filter by sleeve; None for global ECE.
    high_confidence_only : bool
        If True, only compute ECE for bins ≥ 0.80.
    min_bin_samples : int
        Minimum samples per bin for inclusion in ECE calculation.
    db_path : Path
        Path to SQLite database.

    Returns
    -------
    dict with ``ece_score``, ``per_bin``, ``n_total``, ``is_breached``.
    """
    conn = sqlite3.connect(db_path, timeout=30)
    try:
        # Query resolved predictions grouped by bin
        where_clause = "WHERE actual_outcome IS NOT NULL"
        params: list = []

        if sleeve:
            where_clause += " AND sleeve = ?"
            params.append(sleeve)

        if high_confidence_only:
            where_clause += " AND confidence_bin IN ('0.80-0.90', '0.90-1.00')"

        query = f"""
            SELECT
                confidence_bin,
                COUNT(*) as n_samples,
                AVG(CAST(actual_outcome AS FLOAT)) as empirical_accuracy,
                AVG(predicted_probability) as stated_confidence
            FROM ece_tracking
            {where_clause}
            GROUP BY confidence_bin
            HAVING COUNT(*) >= ?
        """
        params.append(min_bin_samples)

        rows = conn.execute(query, params).fetchall()

        # Get total resolved count
        total_query = f"""
            SELECT COUNT(*) FROM ece_tracking {where_clause}
        """
        n_total = conn.execute(total_query, params[:-1]).fetchone()[0]

    finally:
        conn.close()

    if not rows or n_total == 0:
        return {
            "ece_score": 0.0,
            "per_bin": {},
            "n_total": n_total,
            "is_breached": False,
            "insufficient_data": True,
        }

    # Compute ECE
    ece = 0.0
    per_bin: dict[str, dict] = {}

    for confidence_bin, n_samples, emp_acc, stated_conf in rows:
        bin_weight = n_samples / n_total
        bin_error = abs(emp_acc - stated_conf)
        ece += bin_weight * bin_error

        per_bin[confidence_bin] = {
            "n_samples": n_samples,
            "empirical_accuracy": round(emp_acc, 4),
            "stated_confidence": round(stated_conf, 4),
            "calibration_error": round(bin_error, 4),
            "weight": round(bin_weight, 4),
        }

    # Check breach condition
    is_breached = ece > 0.10

    if is_breached:
        logger.warning(
            "🚨 ECE BREACH — ECE=%.4f > 0.10, sleeve=%s, N=%d",
            ece, sleeve or "GLOBAL", n_total,
        )

    return {
        "ece_score": round(ece, 6),
        "per_bin": per_bin,
        "n_total": n_total,
        "is_breached": is_breached,
        "insufficient_data": False,
    }


# ═══════════════════════════════════════════════════════════════════════
# Circuit Breaker
# ═══════════════════════════════════════════════════════════════════════

def check_ece(
    sleeve: str | None = None,
    threshold: float = 0.10,
    min_samples: int = 50,
    db_path: Path = STATE_DB,
) -> dict:
    """Run ECE calibration check for high-confidence bins.

    If ECE exceeds threshold for high-confidence bins (≥0.80) with
    sufficient sample count, returns ``is_breached=True`` which should
    trigger ``HALTED_ECE_BREACH`` in the DAG state machine.

    Parameters
    ----------
    sleeve : str or None
        Sleeve to check; None for global.
    threshold : float
        ECE breach threshold.
    min_samples : int
        Minimum samples required per bin.

    Returns
    -------
    dict with check results.
    """
    # ── Early guard: bypass if insufficient resolved data ────────────
    # Prevents ZeroDivisionError when ece_tracking is empty or has
    # fewer resolved predictions than the minimum sample threshold.
    import sqlite3 as _sqlite3
    _conn = _sqlite3.connect(db_path, timeout=30)
    try:
        _where = "WHERE actual_outcome IS NOT NULL"
        _params: list = []
        if sleeve:
            _where += " AND sleeve = ?"
            _params.append(sleeve)
        _count = _conn.execute(
            f"SELECT COUNT(*) FROM ece_tracking {_where}", _params
        ).fetchone()[0]
    except Exception:
        _count = 0
    finally:
        _conn.close()

    if _count < min_samples:
        logger.info(
            "ECE BYPASS — %d resolved samples < %d minimum for %s, skipping",
            _count, min_samples, sleeve or "GLOBAL",
        )
        return {
            "sleeve": sleeve or "GLOBAL",
            "high_confidence_ece": 0.0,
            "global_ece": 0.0,
            "threshold": threshold,
            "is_breached": False,
            "high_confidence_bins": {},
            "n_total": _count,
            "trigger_state": "OK",
            "insufficient_data": True,
        }

    # Check high-confidence bins specifically
    hc_result = compute_ece(
        sleeve=sleeve,
        high_confidence_only=True,
        min_bin_samples=min_samples,
        db_path=db_path,
    )

    # Also compute global ECE for reference
    global_result = compute_ece(
        sleeve=sleeve,
        high_confidence_only=False,
        min_bin_samples=min_samples,
        db_path=db_path,
    )

    is_breached = hc_result["ece_score"] > threshold and not hc_result.get("insufficient_data", True)

    result = {
        "sleeve": sleeve or "GLOBAL",
        "high_confidence_ece": hc_result["ece_score"],
        "global_ece": global_result["ece_score"],
        "threshold": threshold,
        "is_breached": is_breached,
        "high_confidence_bins": hc_result["per_bin"],
        "n_total": hc_result["n_total"],
        "trigger_state": "HALTED_ECE_BREACH" if is_breached else "OK",
    }

    if is_breached:
        logger.warning(
            "🚨 ECE CIRCUIT BREAKER — HC_ECE=%.4f > %.2f → HALTED_ECE_BREACH",
            hc_result["ece_score"], threshold,
        )
    else:
        logger.info(
            "ECE OK — HC_ECE=%.4f, Global=%.4f (threshold=%.2f, N=%d)",
            hc_result["ece_score"], global_result["ece_score"],
            threshold, hc_result["n_total"],
        )

    return result
