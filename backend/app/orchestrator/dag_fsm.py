"""DAG Finite State Machine for the orchestrator.

Defines explicit state transitions with circuit breakers at
drift and ECE validation gates.

State Flow:
  PENDING → INGESTING → VALIDATING_DRIFT → INFERRING →
  VALIDATING_ECE → EXECUTING → COMPLETED

Halt States (circuit breakers):
  HALTED_DRIFT — concept drift detected
  HALTED_ECE_BREACH — calibration error exceeded
  HALTED_OOM — CUDA OOM
  CRASHED — unrecoverable error

Idempotency: if a HALTED_* state is reached, target allocations
default to 0.0 and flatten intents are emitted.
"""
from __future__ import annotations

import enum
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

STATE_DB = Path("backend/artifacts/orchestrator_state/state.sqlite3")


class DAGState(str, enum.Enum):
    """Finite state machine states for the DAG orchestrator."""
    PENDING = "PENDING"
    INGESTING = "INGESTING"
    VALIDATING_DRIFT = "VALIDATING_DRIFT"
    INFERRING = "INFERRING"
    VALIDATING_ECE = "VALIDATING_ECE"
    EXECUTING = "EXECUTING"
    COMPLETED = "COMPLETED"

    # Halt states (circuit breakers)
    HALTED_DRIFT = "HALTED_DRIFT"
    HALTED_ECE_BREACH = "HALTED_ECE_BREACH"
    HALTED_OOM = "HALTED_OOM"
    CRASHED = "CRASHED"


# Valid transitions: from_state → set of allowed to_states
VALID_TRANSITIONS: dict[DAGState, set[DAGState]] = {
    DAGState.PENDING: {DAGState.INGESTING, DAGState.CRASHED},
    DAGState.INGESTING: {DAGState.VALIDATING_DRIFT, DAGState.CRASHED},
    DAGState.VALIDATING_DRIFT: {DAGState.INFERRING, DAGState.HALTED_DRIFT, DAGState.CRASHED},
    DAGState.INFERRING: {DAGState.VALIDATING_ECE, DAGState.HALTED_OOM, DAGState.CRASHED},
    DAGState.VALIDATING_ECE: {DAGState.EXECUTING, DAGState.HALTED_ECE_BREACH, DAGState.CRASHED},
    DAGState.EXECUTING: {DAGState.COMPLETED, DAGState.CRASHED},
    # Halt states are terminal — can only restart via a new run_id
    DAGState.COMPLETED: set(),
    DAGState.HALTED_DRIFT: set(),
    DAGState.HALTED_ECE_BREACH: set(),
    DAGState.HALTED_OOM: set(),
    DAGState.CRASHED: set(),
}

# States that short-circuit the pipeline → emit flatten intents
HALT_STATES = {
    DAGState.HALTED_DRIFT,
    DAGState.HALTED_ECE_BREACH,
    DAGState.HALTED_OOM,
    DAGState.CRASHED,
}


class DAGStateMachine:
    """Manages DAG execution state transitions with idempotent halting.

    Parameters
    ----------
    run_id : str
        Unique identifier for this DAG execution run.
    db_path : Path
        Path to the SQLite database for state persistence.
    """

    def __init__(self, run_id: str, db_path: Path = STATE_DB):
        self.run_id = run_id
        self.db_path = db_path
        self._state = DAGState.PENDING
        self._halt_reason: str | None = None

        # Persist initial state
        self._persist_state()

    @property
    def state(self) -> DAGState:
        return self._state

    @property
    def is_halted(self) -> bool:
        return self._state in HALT_STATES

    @property
    def halt_reason(self) -> str | None:
        return self._halt_reason

    def transition(self, to_state: DAGState, reason: str | None = None) -> bool:
        """Attempt a state transition.

        Parameters
        ----------
        to_state : DAGState
            Target state.
        reason : str or None
            Reason for the transition (especially for halt states).

        Returns
        -------
        bool — True if transition succeeded.

        Raises
        ------
        InvalidTransitionError if the transition is not allowed.
        """
        allowed = VALID_TRANSITIONS.get(self._state, set())
        if to_state not in allowed:
            raise InvalidTransitionError(
                f"Cannot transition from {self._state.value} to {to_state.value}. "
                f"Allowed: {[s.value for s in allowed]}"
            )

        old_state = self._state
        self._state = to_state
        self._halt_reason = reason

        logger.info(
            "DAG [%s] %s → %s%s",
            self.run_id[:8], old_state.value, to_state.value,
            f" (reason: {reason})" if reason else "",
        )

        self._persist_state()
        return True

    def halt(self, halt_state: DAGState, reason: str) -> None:
        """Trigger a circuit breaker halt.

        This is a convenience method that transitions to a halt state
        and logs the reason.
        """
        if halt_state not in HALT_STATES:
            raise ValueError(f"{halt_state} is not a halt state")

        try:
            self.transition(halt_state, reason=reason)
        except InvalidTransitionError:
            # Force halt even if not in allowed transitions
            logger.warning(
                "FORCE HALT [%s] %s → %s: %s",
                self.run_id[:8], self._state.value, halt_state.value, reason,
            )
            self._state = halt_state
            self._halt_reason = reason
            self._persist_state()

    def _persist_state(self) -> None:
        """Write current state to SQLite."""
        conn = sqlite3.connect(self.db_path, timeout=30)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute("""
                INSERT OR REPLACE INTO dag_state (run_id, current_state, halt_reason)
                VALUES (?, ?, ?)
            """, (self.run_id, self._state.value, self._halt_reason))
            conn.commit()
        except sqlite3.OperationalError as e:
            if "no such table" in str(e):
                logger.warning("dag_state table not found — run 00_init_state.py first")
            else:
                raise
        finally:
            conn.close()

    @classmethod
    def load(cls, run_id: str, db_path: Path = STATE_DB) -> "DAGStateMachine":
        """Load an existing run's state from the database."""
        conn = sqlite3.connect(db_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=5000")
        try:
            row = conn.execute(
                "SELECT current_state, halt_reason FROM dag_state WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        finally:
            conn.close()

        fsm = cls.__new__(cls)
        fsm.run_id = run_id
        fsm.db_path = db_path

        if row:
            fsm._state = DAGState(row[0])
            fsm._halt_reason = row[1]
        else:
            fsm._state = DAGState.PENDING
            fsm._halt_reason = None

        return fsm


class InvalidTransitionError(Exception):
    """Raised when an invalid FSM state transition is attempted."""


# ═══════════════════════════════════════════════════════════════════════
# Orchestrated Pipeline Runner
# ═══════════════════════════════════════════════════════════════════════

def run_dag_pipeline(
    run_id: str,
    ingest_fn: Callable[[], dict],
    drift_check_fn: Callable[[], dict],
    inference_fn: Callable[[], dict],
    ece_check_fn: Callable[[], dict],
    execute_fn: Callable[[dict], dict],
    db_path: Path = STATE_DB,
) -> dict[str, Any]:
    """Execute the full DAG pipeline with circuit breakers.

    Parameters
    ----------
    run_id : str
        Unique run identifier.
    ingest_fn : callable
        Data refresh function.
    drift_check_fn : callable
        MMD concept drift check. Must return ``{"is_drifted": bool}``.
    inference_fn : callable
        Model inference function. Must return ``{"predictions": ...}``.
    ece_check_fn : callable
        ECE calibration check. Must return ``{"is_breached": bool}``.
    execute_fn : callable
        Order execution function.

    Returns
    -------
    dict with run results.
    """
    fsm = DAGStateMachine(run_id, db_path)
    results: dict[str, Any] = {"run_id": run_id}

    try:
        # Step 1: Ingest
        fsm.transition(DAGState.INGESTING)
        results["ingest"] = ingest_fn()

        # Step 2: Drift validation
        fsm.transition(DAGState.VALIDATING_DRIFT)
        drift_result = drift_check_fn()
        results["drift"] = drift_result

        if drift_result.get("is_drifted", False):
            fsm.halt(DAGState.HALTED_DRIFT, "Concept drift detected")
            results["halt"] = "HALTED_DRIFT"
            results["flatten_intents"] = True
            return results

        # Step 3: Inference
        fsm.transition(DAGState.INFERRING)
        results["inference"] = inference_fn()

        # Step 4: ECE validation
        fsm.transition(DAGState.VALIDATING_ECE)
        ece_result = ece_check_fn()
        results["ece"] = ece_result

        if ece_result.get("is_breached", False):
            fsm.halt(DAGState.HALTED_ECE_BREACH, "ECE breach in high-confidence bins")
            results["halt"] = "HALTED_ECE_BREACH"
            results["flatten_intents"] = True
            return results

        # Step 5: Execute
        fsm.transition(DAGState.EXECUTING)
        results["execution"] = execute_fn(results["inference"])

        # Step 6: Complete
        fsm.transition(DAGState.COMPLETED)
        results["status"] = "COMPLETED"

    except torch.cuda.OutOfMemoryError:
        fsm.halt(DAGState.HALTED_OOM, "CUDA OOM during pipeline")
        results["halt"] = "HALTED_OOM"
        results["flatten_intents"] = True
    except Exception as e:
        fsm.halt(DAGState.CRASHED, str(e))
        results["halt"] = "CRASHED"
        results["error"] = str(e)
        results["flatten_intents"] = True

    return results


# Avoid import error when torch isn't loaded at module level
try:
    import torch
except ImportError:
    pass
