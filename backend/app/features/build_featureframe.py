import os
import pandas as pd
import numpy as np
import logging
from typing import Dict, List
from backend.app.ops import pathmap, artifact_registry
from backend.app.features import schemas, validators
from backend.app.data import calendar, security_master
from backend.app.data.ingest import ohlcv_daily as ingest_daily

logger = logging.getLogger(__name__)

def build_featureframe(start_date, end_date, feature_spec: Dict, code_version: str = "v1") -> pd.DataFrame:
    trading_days = calendar.get_trading_days(start_date, end_date)
    if not trading_days: raise ValueError("No trading days found")
    symbols = feature_spec.get("symbols")
    if not symbols:
        master = security_master.load_security_master()
        symbols = master["symbol"].dropna().unique().tolist()
    df = pd.DataFrame() # Stub
    return df
