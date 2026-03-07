"""CPU-bound data ingest job for TFT Core Reversal sleeve.

Fires parallel async requests with strict 3-second timeouts to fetch
exogenous signals (Nikkei, EuroStoxx, /ZN).  Fails closed to 0.0 on
any API error to prevent inference crashes.

This job runs on CPU (no GPU allocation) and writes the serialized
``TFTInferenceContext`` to ``tft_features.json`` in the artifact
directory for consumption by the GPU worker plugin.

**Critical Invariant**: All data reflects the market state as of
exactly 09:20:00 EST.  The 09:30 RTH open is *not* available at
inference time and must never appear in the feature set.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os

logger = logging.getLogger(__name__)


async def fetch_exogenous_safe(broker) -> dict:
    """Fire parallel requests with strict 3-second timeouts.

    Fails closed to 0.0 — no signal is safer than a stale or
    errored signal that could bias the TFT inference.

    Parameters
    ----------
    broker
        Broker adapter with ``get_overnight_return_async(symbol)`` method.

    Returns
    -------
    dict with ``nikkei_pct``, ``eurostoxx_pct``, ``zn_drift_bps``.
    """

    async def _fetch(symbol: str):
        try:
            return await asyncio.wait_for(
                broker.get_overnight_return_async(symbol), timeout=3.0,
            )
        except (asyncio.TimeoutError, Exception):
            logger.warning(
                "Exogenous fetch failed for %s — imputing 0.0", symbol,
            )
            return 0.0

    results = await asyncio.gather(
        _fetch("N225"),      # Nikkei 225
        _fetch("STOXX50E"),  # EuroStoxx 50
        _fetch("ZN"),        # 10Y Treasury Note
    )
    return {
        "nikkei_pct": results[0],
        "eurostoxx_pct": results[1],
        "zn_drift_bps": results[2] * 10000,
    }


def execute(context: dict, model_cache: dict) -> None:
    """Entry point called by the DAG loader / GPU supervisor.

    Steps
    -----
    1. Fetch /ES 5-min bars up to exactly 09:20:00 EST.
    2. Query pandas_market_calendars for OpEx / Macro event flags.
    3. Build ``TFTInferenceContext``.
    4. Serialize to ``context["artifact_dir"]/tft_features.json``.

    Parameters
    ----------
    context : dict
        Contains ``artifact_dir``, ``asof_date``, and orchestrator metadata.
    model_cache : dict
        Unused for CPU-bound ingest — included for plugin protocol compliance.
    """
    # Placeholder implementation — production version will:
    # 1. Fetch data up to exactly 09:20:00 EST
    # 2. Query pandas_market_calendars for OpEx/Macro flags
    # 3. Build TFTInferenceContext
    # 4. Serialize to context["artifact_dir"]/tft_features.json

    # QUARANTINE: This job produces zeroed-out stub data.
    # Block it from running in the production DAG unless explicitly opted in.
    if os.getenv("ALGAE_ALLOW_STUB_INGEST", "0") != "1":
        raise RuntimeError(
            "data_ingest_core is a stub producing zeroed-out features. "
            "Set ALGAE_ALLOW_STUB_INGEST=1 to allow, or implement the real pipeline."
        )

    artifact_dir = context.get("artifact_dir", ".")
    os.makedirs(artifact_dir, exist_ok=True)

    out_path = os.path.join(artifact_dir, "tft_features.json")
    logger.info(
        "Data ingest for core_tft: asof=%s, artifact_dir=%s",
        context.get("asof_date", "unknown"),
        artifact_dir,
    )

    # Production: Replace with real data pipeline
    # For now, write a stub payload so the downstream plugin can validate I/O
    stub_payload = {
        "asof_date": context.get("asof_date", "1970-01-01"),
        "observed_past_seq": [[0.0, 0.0, 0.0]] * 184,
        "day_of_week": 0,
        "is_opex": 0,
        "macro_event_id": 0,
        "gap_proxy_pct": 0.0,
        "nikkei_pct": 0.0,
        "eurostoxx_pct": 0.0,
        "zn_drift_bps": 0.0,
        "vix_spot": 0.0,
    }
    with open(out_path, "w") as f:
        json.dump(stub_payload, f)

    logger.info("TFT features written to %s", out_path)
