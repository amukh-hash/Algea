from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Optional, List
import random
from backend.app.options.data.types import OptionChainSnapshot, OptionRow
from backend.app.core.config import OPTIONS_SEED

class ChainStore(ABC):
    @abstractmethod
    def get_chain(self, ticker: str, timestamp: datetime, expiry: str, underlying_price: float = None) -> Optional[OptionChainSnapshot]:
        pass

class MockChainStore(ChainStore):
    def __init__(self, seed: int = OPTIONS_SEED):
        self.seed = seed

    def get_chain(self, ticker: str, timestamp: datetime, expiry: str, underlying_price: float = 100.0) -> Optional[OptionChainSnapshot]:
        if underlying_price is None:
            underlying_price = 100.0 # Fallback

        # Deterministic generation
        ts_int = int(timestamp.timestamp())
        rng = random.Random(self.seed + ts_int)

        dte = (datetime.strptime(expiry, "%Y-%m-%d") - timestamp).days
        if dte < 0:
            return None

        # Generate strikes +/- 10%
        center = round(underlying_price, 0)
        strikes = [center + i for i in range(-10, 11, 1)] # 1.0 increments

        rows = []
        for k in strikes:
            # Simple pricing model (very rough)
            dist = (k - underlying_price) / underlying_price

            # Put
            iv = 0.2 + 0.1 * abs(dist) # Skew
            # Intrinsic
            intrinsic_put = max(0, k - underlying_price)
            time_value = underlying_price * 0.05 * (dte/365)**0.5
            put_price = intrinsic_put + time_value * math.exp(-20 * abs(dist)) # decay away from money

            rows.append(OptionRow(
                strike=k,
                option_type="put",
                bid=round(put_price * 0.98, 2),
                ask=round(put_price * 1.02, 2),
                iv=round(iv, 3),
                delta=None, # implement BS later if needed
                oi=rng.randint(100, 5000),
                volume=rng.randint(10, 1000)
            ))

            # Call
            intrinsic_call = max(0, underlying_price - k)
            call_price = intrinsic_call + time_value * math.exp(-20 * abs(dist))

            rows.append(OptionRow(
                strike=k,
                option_type="call",
                bid=round(call_price * 0.98, 2),
                ask=round(call_price * 1.02, 2),
                iv=round(iv, 3), # simplified symm skew
                delta=None,
                oi=rng.randint(100, 5000),
                volume=rng.randint(10, 1000)
            ))

        return OptionChainSnapshot(
            ticker=ticker,
            timestamp=timestamp,
            expiry=expiry,
            dte=dte,
            rows=rows
        )

import math
