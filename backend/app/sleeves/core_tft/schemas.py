"""Data contracts for TFT Core Reversal sleeve.

Defines the inference context that the CPU-bound data ingest job
serializes to ``tft_features.json`` for consumption by the GPU worker.

The observed_past_seq shape [184, 3] represents:
  - 184 bars: 18:00 → 09:20 EST (15.33 hours × 12 bars/hr at 5-min resolution)
  - 3 features: [log_return, volume_normalized, vwap_normalized]
"""
from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class TFTInferenceContext(BaseModel):
    """Immutable feature payload for a single TFT inference pass.

    All data must reflect the market state as of **exactly 09:20:00 EST**.
    Using the 09:30 RTH open introduces catastrophic lookahead bias.
    """

    asof_date: str

    # Shape: [184, 3] -> 18:00 to 09:20 (15.33 hours * 12 bars/hr = 184 bars)
    # Features: [log_return, volume_normalized, vwap_normalized]
    observed_past_seq: List[List[float]] = Field(
        ..., min_length=184, max_length=184,
    )

    # Static Categoricals
    day_of_week: int = Field(..., ge=0, le=4)
    is_opex: int = Field(..., ge=0, le=1)
    macro_event_id: int = Field(..., ge=0)  # 0=None, 1=CPI, 2=FOMC, 3=NFP, etc.

    # Observed Covariates (Strict 09:20 EST snapshot)
    gap_proxy_pct: float      # (09:20_LTP - prev_16:00_Close) / prev_16:00_Close
    nikkei_pct: float          # Default 0.0 on API failure
    eurostoxx_pct: float       # Default 0.0 on API failure
    zn_drift_bps: float        # 10Y Note overnight drift
    vix_spot: float            # Proxy via /VX continuous futures, not CBOE Spot
