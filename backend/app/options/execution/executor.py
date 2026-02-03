from abc import ABC, abstractmethod
from typing import Optional
from backend.app.options.types import OptionsDecision, OptionsPosition

class OptionsExecutor(ABC):
    @abstractmethod
    def execute(self, decision: OptionsDecision) -> Optional[OptionsPosition]:
        """
        Executes the decision. Returns the resulting position (if open) or None (if close/noop).
        """
        pass

    @abstractmethod
    def get_positions(self) -> list[OptionsPosition]:
        pass
