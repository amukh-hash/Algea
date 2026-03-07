from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

from algae.execution.interfaces import ExecutionDecision

logger = logging.getLogger(__name__)


@dataclass
class PaperExecutor:
    decisions: List[ExecutionDecision]

    def execute(self, decision: ExecutionDecision) -> None:
        self.decisions.append(decision)


@dataclass
class NoopExecutor:
    """No-op executor — only valid in NOOP mode.

    If invoked while RuntimeMode != NOOP, logs a CRITICAL alert and raises.
    """

    def execute(self, _decision: ExecutionDecision) -> None:
        from backend.app.core.runtime_mode import RuntimeMode

        # Check current runtime mode from environment
        import os
        current_mode = os.getenv("ALGAE_RUNTIME_MODE", "noop")
        if current_mode != RuntimeMode.NOOP.value:
            logger.critical(
                "NoopExecutor invoked in mode '%s' — this should NEVER happen "
                "outside of NOOP mode. Aborting execution.",
                current_mode,
            )
            raise RuntimeError(
                f"NoopExecutor cannot be used in mode '{current_mode}'. "
                f"Only RuntimeMode.NOOP is allowed."
            )
        # Noop: silently discard the decision
