"""IBKR historical data provider for the pipeline's bronze ingestion layer.

Extends :class:`FuturesDataProvider` to fetch daily bars directly from IBKR
Gateway/TWS via :class:`IbkrClient.historical_bars`.  Uses roll-segment
approach with ``includeExpired=True`` to access expired contract data.
Results are cached to disk with sha256 checksums for deterministic replay.
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from algae.trading.ibkr_client import IbkrClient
from algae.trading.ibkr_contracts import build_future_contract

from .ingest import FuturesDataProvider
from ..contract_master import CONTRACT_MASTER
from ..roll import active_contract_for_day

logger = logging.getLogger(__name__)


class IBKRHistoricalDataProvider(FuturesDataProvider):
    """Fetch daily RTH bars from IBKR and cache to disk.

    Parameters
    ----------
    client
        Connected :class:`IbkrClient` (or mock for testing).
    what_to_show
        IBKR bar type: ``"TRADES"``, ``"MIDPOINT"``, etc.
    use_rth
        If True (default), only regular-trading-hours bars (09:30–16:00).
    cache_dir
        Directory for caching parquet + sidecar metadata.
    """

    def __init__(
        self,
        client: IbkrClient,
        *,
        what_to_show: str = "TRADES",
        use_rth: bool = True,
        cache_dir: str | Path | None = None,
    ) -> None:
        self._client = client
        self.what_to_show = what_to_show
        self.use_rth = use_rth
        self.cache_dir = Path(cache_dir) if cache_dir else None

    # -- FuturesDataProvider interface ----------------------------------------

    def fetch_daily_bars(
        self, root: str, start: date, end: date,
    ) -> pd.DataFrame:
        """Fetch daily bars for *root* from IBKR, using roll-based segmentation.

        Uses ``includeExpired=True`` so that expired contracts can be
        qualified and their historical data retrieved.  Each roll segment
        is fetched separately and concatenated into a continuous series.

        Returns
        -------
        pd.DataFrame
            Columns: ``timestamp, open, high, low, close, volume``
            (timestamp is UTC-aware).
        """
        # --- Check cache first ---
        if self.cache_dir is not None:
            cached = self._load_cached(root, start, end)
            if cached is not None:
                return cached

        # --- Download segment by segment ---
        spec = CONTRACT_MASTER[root]
        all_bars: List[pd.DataFrame] = []

        # Build a list of (segment_start, segment_end, active_contract_symbol)
        segments = self._build_roll_segments(root, start, end, spec)
        logger.info(
            "IBKR hist: %s has %d roll segments over %s→%s",
            root, len(segments), start, end,
        )

        for seg_start, seg_end, contract_sym in segments:
            logger.info("  Segment %s→%s : %s", seg_start, seg_end, contract_sym)
            from algae.trading.ibkr_contracts import parse_active_contract_symbol

            root_parsed, expiry = parse_active_contract_symbol(contract_sym)

            # Build contract with includeExpired=True for historical data
            contract = build_future_contract(
                root_parsed, expiry, include_expired=True,
            )

            # Qualify (includeExpired allows expired contracts)
            try:
                qualified = self._client.qualify_contracts(contract)
                if not qualified or qualified[0].conId == 0:
                    logger.warning(
                        "Failed to qualify %s — skipping segment", contract_sym,
                    )
                    continue
            except Exception as e:
                logger.warning(
                    "Qualification error for %s: %s — skipping", contract_sym, e,
                )
                continue

            # Calculate duration in days
            n_days = (seg_end - seg_start).days + 1
            duration = f"{n_days} D"

            # End datetime for IBKR (inclusive end = end + 1 day at midnight)
            end_dt = (seg_end + timedelta(days=1)).strftime("%Y%m%d %H:%M:%S")

            try:
                bars = self._client.historical_bars(
                    contract=qualified[0],
                    duration=duration,
                    bar_size="1 day",
                    what_to_show=self.what_to_show,
                    use_rth=self.use_rth,
                    end_dt=end_dt,
                )
            except Exception as e:
                logger.warning(
                    "Error fetching bars for %s %s→%s: %s",
                    contract_sym, seg_start, seg_end, e,
                )
                continue

            if bars.empty:
                logger.warning(
                    "No bars for %s %s→%s", contract_sym, seg_start, seg_end,
                )
                continue

            # Filter to requested range
            bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
            mask = (
                (bars["timestamp"].dt.date >= seg_start)
                & (bars["timestamp"].dt.date <= seg_end)
            )
            all_bars.append(bars.loc[mask].copy())

            # Rate limit courtesy — 0.5s between requests
            self._client.sleep(0.5)

        if not all_bars:
            logger.warning(
                "No IBKR data returned for %s between %s and %s",
                root, start, end,
            )
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )

        combined = pd.concat(all_bars, ignore_index=True)
        combined = combined.sort_values("timestamp").drop_duplicates(
            subset=["timestamp"], keep="last"
        ).reset_index(drop=True)

        # Enforce schema columns
        for col in ("open", "high", "low", "close"):
            combined[col] = pd.to_numeric(combined[col], errors="coerce")
        combined["volume"] = (
            pd.to_numeric(combined["volume"], errors="coerce").fillna(0).astype(int)
        )

        logger.info(
            "IBKR hist: %s → %d total bars (%s → %s)",
            root, len(combined),
            combined["timestamp"].min().date(),
            combined["timestamp"].max().date(),
        )

        # --- Cache ---
        if self.cache_dir is not None:
            self._save_cache(combined, root, start, end)

        return combined

    # -- Roll segmentation ---------------------------------------------------

    @staticmethod
    def _build_roll_segments(
        root: str,
        start: date,
        end: date,
        spec: Any,
    ) -> List[tuple[date, date, str]]:
        """Build date segments where the active contract is constant.

        Returns list of ``(segment_start, segment_end, active_contract_symbol)``.
        """
        segments: List[tuple[date, date, str]] = []
        current_day = start
        current_contract = active_contract_for_day(root, current_day, spec)
        seg_start = current_day

        while current_day <= end:
            contract = active_contract_for_day(root, current_day, spec)
            if contract != current_contract:
                # Close previous segment
                segments.append(
                    (seg_start, current_day - timedelta(days=1), current_contract)
                )
                current_contract = contract
                seg_start = current_day
            current_day += timedelta(days=1)

        # Close last segment
        segments.append((seg_start, end, current_contract))
        return segments

    # -- Caching -------------------------------------------------------------

    def _cache_key(self, root: str, start: date, end: date) -> str:
        return f"{root}_{start.isoformat()}_{end.isoformat()}_rth{self.use_rth}"

    def _cache_path(self, root: str, start: date, end: date) -> Path:
        assert self.cache_dir is not None
        cache_subdir = self.cache_dir / "ibkr_hist"
        cache_subdir.mkdir(parents=True, exist_ok=True)
        key = self._cache_key(root, start, end)
        return cache_subdir / f"{key}.parquet"

    def _load_cached(
        self, root: str, start: date, end: date,
    ) -> Optional[pd.DataFrame]:
        path = self._cache_path(root, start, end)
        if path.exists():
            logger.debug("Cache hit: %s", path)
            df = pd.read_parquet(path)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            return df
        return None

    def _save_cache(
        self, df: pd.DataFrame, root: str, start: date, end: date,
    ) -> None:
        path = self._cache_path(root, start, end)

        # Write parquet
        df.to_parquet(path, index=False)

        # Write sidecar
        buf = io.BytesIO()
        df.to_parquet(buf, index=False, engine="pyarrow")
        checksum = hashlib.sha256(buf.getvalue()).hexdigest()

        meta = {
            "root": root,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "use_rth": self.use_rth,
            "what_to_show": self.what_to_show,
            "row_count": len(df),
            "sha256": checksum,
            "retrieval_ts": pd.Timestamp.now("UTC").isoformat(),
        }
        meta_path = path.with_suffix(".json")
        meta_path.write_text(
            json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8"
        )
        logger.info(
            "Cached %d bars → %s (sha256=%s…)", len(df), path, checksum[:12],
        )
