"""
IV surface builder — extracts ATM IV, term structure proxies, and skew
from an option chain DataFrame.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from algea.data.options.greeks_engine import bs_delta


class IVSurfaceBuilder:
    """Pure-function design: operates on validated chain DataFrames."""

    # ------------------------------------------------------------------
    # ATM IV
    # ------------------------------------------------------------------

    @staticmethod
    def atm_iv(chain: pd.DataFrame, underlying_price: Optional[float] = None) -> float:
        """Return ATM implied vol (average of nearest put + call IVs).

        Selects the strike closest to *underlying_price* and averages
        the call and put IV at that strike across the nearest expiry.
        """
        if chain.empty:
            return np.nan
        price = underlying_price or float(chain["underlying_price"].iloc[0])

        # Use nearest expiry with DTE >= 7
        valid = chain[chain["dte"] >= 7]
        if valid.empty:
            valid = chain
        nearest_exp = valid["expiry"].min()
        exp_chain = valid[valid["expiry"] == nearest_exp]

        # Nearest strike to spot
        exp_chain = exp_chain.copy()
        exp_chain["_dist"] = (exp_chain["strike"] - price).abs()
        atm_strike = exp_chain.loc[exp_chain["_dist"].idxmin(), "strike"]

        atm_rows = exp_chain[exp_chain["strike"] == atm_strike]
        return float(atm_rows["implied_vol"].mean())

    # ------------------------------------------------------------------
    # Term structure slope (short - long IV proxy)
    # ------------------------------------------------------------------

    @staticmethod
    def term_slope(
        chain: pd.DataFrame,
        short_dte_target: int = 30,
        long_dte_target: int = 60,
    ) -> float:
        """IV30 - IV60 proxy using ATM IV at nearest expiries to targets."""
        if chain.empty:
            return np.nan
        price = float(chain["underlying_price"].iloc[0])

        def _atm_iv_for_dte(target: int) -> float:
            c = chain.copy()
            c["_dte_dist"] = (c["dte"] - target).abs()
            best_exp = c.loc[c["_dte_dist"].idxmin(), "expiry"]
            exp_c = c[c["expiry"] == best_exp]
            exp_c = exp_c.copy()
            exp_c["_dist"] = (exp_c["strike"] - price).abs()
            atm_k = exp_c.loc[exp_c["_dist"].idxmin(), "strike"]
            return float(exp_c[exp_c["strike"] == atm_k]["implied_vol"].mean())

        iv_short = _atm_iv_for_dte(short_dte_target)
        iv_long = _atm_iv_for_dte(long_dte_target)
        return iv_short - iv_long

    # ------------------------------------------------------------------
    # 25-delta skew
    # ------------------------------------------------------------------

    @staticmethod
    def skew_25d(chain: pd.DataFrame) -> float:
        """IV(25-delta put) - IV(25-delta call) using nearest delta match.

        Uses approximate BS delta to find strikes closest to ±25 delta.
        """
        if chain.empty:
            return np.nan

        price = float(chain["underlying_price"].iloc[0])

        # Use nearest expiry in 20–50 DTE range, fallback to nearest
        dte_mask = (chain["dte"] >= 20) & (chain["dte"] <= 50)
        subset = chain[dte_mask] if dte_mask.any() else chain
        nearest_exp = subset["expiry"].min()
        exp_chain = subset[subset["expiry"] == nearest_exp].copy()

        T = float(exp_chain["dte"].iloc[0]) / 365.0
        r = float(exp_chain["risk_free_rate"].iloc[0])
        q = float(exp_chain["dividend_yield"].iloc[0])

        puts = exp_chain[exp_chain["option_type"].str.lower() == "put"].copy()
        calls = exp_chain[exp_chain["option_type"].str.lower() == "call"].copy()

        if puts.empty or calls.empty:
            return np.nan

        # Compute deltas
        puts["_delta"] = bs_delta(
            price, puts["strike"].to_numpy(), T, r,
            puts["implied_vol"].to_numpy(), "put", q,
        )
        calls["_delta"] = bs_delta(
            price, calls["strike"].to_numpy(), T, r,
            calls["implied_vol"].to_numpy(), "call", q,
        )

        # 25-delta put: delta closest to -0.25
        puts["_d_dist"] = (puts["_delta"] + 0.25).abs()
        put_row = puts.loc[puts["_d_dist"].idxmin()]
        iv_put_25 = float(put_row["implied_vol"])

        # 25-delta call: delta closest to +0.25
        calls["_d_dist"] = (calls["_delta"] - 0.25).abs()
        call_row = calls.loc[calls["_d_dist"].idxmin()]
        iv_call_25 = float(call_row["implied_vol"])

        return iv_put_25 - iv_call_25

    # ------------------------------------------------------------------
    # Build summary snapshot
    # ------------------------------------------------------------------

    @classmethod
    def build_snapshot(cls, chain: pd.DataFrame) -> Dict[str, float]:
        """Return a dict of surface-derived features."""
        return {
            "atm_iv": cls.atm_iv(chain),
            "term_slope": cls.term_slope(chain),
            "skew_25d": cls.skew_25d(chain),
        }
