from typing import Optional
import polars as pl
from datetime import date
from backend.app.core import artifacts
from backend.app.models.signal_types import ChronosPriors

class TeacherORunner:
    """
    Loads Chronos-2 priors for Options Overlay.
    Does NOT run inference (that happens in nightly cycle).
    Just provides data access.
    """
    def __init__(self):
        self.priors_cache = {} # date -> dataframe

    def get_priors(self, ticker: str, as_of_date: date) -> Optional[ChronosPriors]:
        """
        Retrieves priors for a specific ticker and date.
        """
        date_str = str(as_of_date)
        
        # Load Cache if needed
        if date_str not in self.priors_cache:
            path = artifacts.resolve_priors_path(date_str, "v1")
            if not path:
                return None
            try:
                self.priors_cache[date_str] = pl.read_parquet(path)
            except Exception:
                return None
                
        df = self.priors_cache[date_str]
        
        # Filter (Polars)
        row = df.filter(pl.col("symbol") == ticker)
        if len(row) == 0:
            return None
            
        # unpack
        # Assuming schema: drift, vol_forecast, tail_risk, trend_conf
        try:
            return ChronosPriors(
                drift=row["drift"][0],
                vol_forecast=row["vol_forecast"][0],
                tail_risk=row["tail_risk"][0],
                trend_conf=row["trend_conf"][0]
            )
        except Exception:
            # Schema mismatch?
            return None
