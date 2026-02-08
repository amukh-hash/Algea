from typing import Optional, List
from backend.app.options.execution.executor import OptionsExecutor
from backend.app.options.types import OptionsDecision, OptionsPosition
import logging

logger = logging.getLogger(__name__)

class NoopExecutor(OptionsExecutor):
    def execute(self, decision: OptionsDecision) -> Optional[OptionsPosition]:
        logger.info(f"[NOOP] Would execute: {decision}")
        return None
        
    def get_positions(self) -> List[OptionsPosition]:
        return []
