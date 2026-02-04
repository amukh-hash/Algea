import random
import math
from datetime import datetime, timedelta
from typing import Optional, List
from backend.app.core.config import OPTIONS_SEED
from backend.app.options.data.types import IVSnapshot
from backend.app.options.data.providers.base import IVProvider

class MockIVProvider(IVProvider):
    def __init__(self, seed: int = OPTIONS_SEED):
        self.seed = seed
        self.base_iv = 0.20 # 20%
        self.vol_of_vol = 0.05
        
    def _generate_synthetic_iv(self, ticker: str, timestamp: datetime, dte: int) -> IVSnapshot:
        # Deterministic generation based on timestamp and ticker
        # We combine seed + ticker_hash + timestamp_int
        ts_int = int(timestamp.timestamp())
        ticker_val = sum(ord(c) for c in ticker)
        
        # Local random instance to avoid polluting global state
        rng = random.Random(self.seed + ts_int + ticker_val)
        
        # Synthetic regime: daily seasonality + random walk
        day_of_year = timestamp.timetuple().tm_yday
        seasonal = 0.05 * math.sin(2 * math.pi * day_of_year / 365.0)
        
        noise = rng.gauss(0, self.vol_of_vol)
        
        # IV tends to be higher for lower DTE (inverse term structure often in stress, but let's assume flat-ish + noise)
        term_structure = 0.0
        if dte < 7:
            term_structure = 0.05 # Higher short term
        
        iv_val = max(0.05, self.base_iv + seasonal + term_structure + noise)
        
        # IV Rank: roughly where iv_val sits in [0.10, 0.40]
        iv_rank = max(0.0, min(1.0, (iv_val - 0.10) / 0.30))
        iv_percentile = iv_rank # simplified
        
        return IVSnapshot(
            ticker=ticker,
            timestamp=timestamp,
            dte=dte,
            atm_iv=round(iv_val, 4),
            iv_rank=round(iv_rank, 2),
            iv_percentile=round(iv_percentile, 2),
            metadata={"source": "mock", "seed": self.seed}
        )

    def get_iv(self, ticker: str, timestamp: datetime, dte: int) -> Optional[IVSnapshot]:
        return self._generate_synthetic_iv(ticker, timestamp, dte)

    def get_iv_history(self, ticker: str, start_date: datetime, end_date: datetime, dte: int) -> List[IVSnapshot]:
        snapshots = []
        curr = start_date
        while curr <= end_date:
            # Assume daily close or hourly? Let's do daily for history
            if curr.weekday() < 5: # Mon-Fri
                snapshots.append(self._generate_synthetic_iv(ticker, curr, dte))
            curr += timedelta(days=1)
        return snapshots
