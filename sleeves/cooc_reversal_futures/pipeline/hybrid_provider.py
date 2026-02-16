"""Hybrid IBKR + Yahoo Finance data provider.

Attempts IBKR first for each root; if IBKR returns empty (expired-contract
qualification failure), falls back to YFinance.  Each root's actual source
is tracked in ``source_map`` so the validation layer can report coverage.

For promotion purposes the pipeline treats *any* IBKR-sourced root as
promotion-grade data; roots falling back to yfinance are flagged
RESEARCH_ONLY per root (but the overall run is still considered IBKR-grade
if the majority of roots are IBKR-sourced).
"""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from .ingest import FuturesDataProvider

logger = logging.getLogger(__name__)


class HybridDataProvider(FuturesDataProvider):
    """Try IBKR first, fall back to YFinance per root.

    Parameters
    ----------
    ibkr_provider
        An initialized ``IBKRHistoricalDataProvider`` (already connected).
    yfinance_provider
        An initialized ``YFinanceDataProvider``.
    """

    def __init__(
        self,
        ibkr_provider: FuturesDataProvider,
        yfinance_provider: FuturesDataProvider,
    ) -> None:
        self._ibkr = ibkr_provider
        self._yfinance = yfinance_provider
        # Track which source provided data for each root
        self.source_map: Dict[str, str] = {}

    def fetch_daily_bars(
        self, root: str, start: date, end: date,
    ) -> pd.DataFrame:
        """Fetch from IBKR; fall back to YFinance if IBKR returns empty."""
        # Try IBKR first
        try:
            df = self._ibkr.fetch_daily_bars(root, start, end)
            if not df.empty and len(df) >= 100:  # require minimum coverage
                self.source_map[root] = "ibkr_hist"
                logger.info(
                    "Root %s: %d rows from IBKR", root, len(df),
                )
                return df
            else:
                logger.warning(
                    "Root %s: IBKR returned only %d rows — falling back to yfinance",
                    root, len(df) if not df.empty else 0,
                )
        except Exception as e:
            logger.warning(
                "Root %s: IBKR fetch failed (%s) — falling back to yfinance",
                root, e,
            )

        # Fall back to YFinance
        try:
            df = self._yfinance.fetch_daily_bars(root, start, end)
            self.source_map[root] = "yfinance"
            logger.info(
                "Root %s: %d rows from yfinance (fallback)", root, len(df),
            )
            return df
        except Exception as e:
            logger.error("Root %s: both IBKR and yfinance failed: %s", root, e)
            self.source_map[root] = "failed"
            return pd.DataFrame()

    @property
    def ibkr_roots(self) -> list[str]:
        """Roots successfully sourced from IBKR."""
        return [r for r, s in self.source_map.items() if s == "ibkr_hist"]

    @property
    def yfinance_roots(self) -> list[str]:
        """Roots that fell back to yfinance."""
        return [r for r, s in self.source_map.items() if s == "yfinance"]

    @property
    def ibkr_coverage_ratio(self) -> float:
        """Fraction of roots sourced from IBKR."""
        total = len(self.source_map) or 1
        return len(self.ibkr_roots) / total

    def summary(self) -> str:
        """Human-readable summary of data sources used."""
        ibkr = sorted(self.ibkr_roots)
        yf = sorted(self.yfinance_roots)
        return (
            f"Hybrid: {len(ibkr)} IBKR roots ({', '.join(ibkr)}), "
            f"{len(yf)} yfinance roots ({', '.join(yf)})"
        )
