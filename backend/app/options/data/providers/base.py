from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, List
from backend.app.options.data.types import IVSnapshot

class IVProvider(ABC):
    @abstractmethod
    def get_iv(self, ticker: str, timestamp: datetime, dte: int) -> Optional[IVSnapshot]:
        """Fetch IV snapshot for a specific ticker/time/dte."""
        pass

    @abstractmethod
    def get_iv_history(self, ticker: str, start_date: datetime, end_date: datetime, dte: int) -> List[IVSnapshot]:
        """Fetch range of IV snapshots."""
        pass
