"""
Options IV Surface Pipeline — SPY volatility grid construction.

Fetches live SPY options chains, computes implied volatility, and
interpolates into a fixed spatial-temporal grid for the
SpatialTemporalTransformer model.

Output: ``[Temporal=5, Spatial=25]`` grid of IV values arranged by
tenor (DTE bucket) and delta bucket.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.stats import norm

logger = logging.getLogger(__name__)

# Grid dimensions
NUM_TENORS = 5        # DTE buckets: ~7, 14, 30, 60, 90 days
NUM_DELTA_BUCKETS = 25  # From 0.02 to 0.98 delta


# ═══════════════════════════════════════════════════════════════════════════
# Black-Scholes IV Solver
# ═══════════════════════════════════════════════════════════════════════════

def _bs_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes call price."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def _bs_put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes put price."""
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float = 0.05,
    option_type: str = "call",
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float:
    """Newton-Raphson IV solver.

    Parameters
    ----------
    market_price : float
        Observed option price.
    S, K, T, r : float
        Spot, strike, time-to-expiry (years), risk-free rate.
    option_type : str
        "call" or "put".
    tol, max_iter : float, int
        Convergence parameters.

    Returns
    -------
    float
        Implied volatility. Returns NaN if solver fails.
    """
    if T <= 0 or market_price <= 0:
        return float("nan")

    intrinsic = max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
    if market_price < intrinsic:
        return float("nan")

    sigma = 0.3  # Initial guess
    price_fn = _bs_call_price if option_type == "call" else _bs_put_price

    for _ in range(max_iter):
        price = price_fn(S, K, T, r, sigma)
        # Vega: dP/dσ
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        vega = S * norm.pdf(d1) * np.sqrt(T)

        if abs(vega) < 1e-12:
            break

        sigma_new = sigma - (price - market_price) / vega

        if sigma_new <= 0:
            sigma_new = sigma * 0.5

        if abs(sigma_new - sigma) < tol:
            return sigma_new

        sigma = sigma_new

    return sigma if sigma > 0 else float("nan")


def compute_delta(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call",
) -> float:
    """Black-Scholes delta."""
    if T <= 0 or sigma <= 0:
        return float("nan")
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    if option_type == "call":
        return float(norm.cdf(d1))
    return float(norm.cdf(d1) - 1)


# ═══════════════════════════════════════════════════════════════════════════
# Grid Interpolation
# ═══════════════════════════════════════════════════════════════════════════

def interpolate_to_grid(
    iv_surface: pd.DataFrame,
    num_tenors: int = NUM_TENORS,
    num_deltas: int = NUM_DELTA_BUCKETS,
) -> np.ndarray:
    """Interpolate irregular IV surface into a fixed grid.

    Parameters
    ----------
    iv_surface : pd.DataFrame
        Must have columns: ``dte``, ``delta``, ``iv``.
    num_tenors : int
        Number of DTE buckets.
    num_deltas : int
        Number of delta buckets.

    Returns
    -------
    np.ndarray
        Shape ``[num_tenors, num_deltas]`` — the fixed IV grid.
    """
    if iv_surface.empty:
        raise ValueError("IV surface DataFrame is empty")

    # Target grid points
    tenor_targets = np.array([7, 14, 30, 60, 90], dtype=float)[:num_tenors]
    delta_targets = np.linspace(0.02, 0.98, num_deltas)

    grid = np.full((num_tenors, num_deltas), np.nan)

    # Group by DTE, fit cubic spline across delta
    dte_groups = iv_surface.groupby("dte")
    available_dtes = sorted(dte_groups.groups.keys())

    if len(available_dtes) < 2:
        raise ValueError(
            f"Need at least 2 DTE groups for interpolation, got {len(available_dtes)}"
        )

    # For each tenor target, find nearest DTE group and interpolate across delta
    for t_idx, target_dte in enumerate(tenor_targets):
        # Find closest available DTE
        closest_dte = min(available_dtes, key=lambda x: abs(x - target_dte))
        group = dte_groups.get_group(closest_dte)

        if len(group) < 3:
            continue

        # Sort by delta and fit cubic spline
        group_sorted = group.sort_values("delta")
        deltas = group_sorted["delta"].values
        ivs = group_sorted["iv"].values

        # Remove NaN
        valid = ~np.isnan(ivs)
        if valid.sum() < 3:
            continue

        try:
            # Deduplicate deltas (average IV for same delta)
            dedup = group_sorted.groupby("delta", as_index=False)["iv"].mean()
            dedup = dedup.sort_values("delta")
            d_vals = dedup["delta"].values
            iv_vals = dedup["iv"].values
            valid = ~np.isnan(iv_vals)
            if valid.sum() < 3:
                continue
            cs = CubicSpline(d_vals[valid], iv_vals[valid], extrapolate=True)
            grid[t_idx, :] = cs(delta_targets)
        except Exception as e:
            logger.warning("Spline failed for DTE=%d: %s", closest_dte, e)

    # Fill remaining NaN with column mean
    col_means = np.nanmean(grid, axis=0)
    for j in range(num_deltas):
        mask = np.isnan(grid[:, j])
        grid[mask, j] = col_means[j] if not np.isnan(col_means[j]) else 0.2

    return np.clip(grid, 0.01, 2.0).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# Options Chain Ingestion (IBKR / yfinance)
# ═══════════════════════════════════════════════════════════════════════════

def fetch_spy_options_chain(
    spot_price: Optional[float] = None,
) -> pd.DataFrame:
    """Fetch SPY options chain and compute IV + delta.

    Uses yfinance as the default data source.  For production, replace
    with IBKR reqSecDefOptParams + reqMktData.

    Returns
    -------
    pd.DataFrame
        Columns: ``strike``, ``dte``, ``delta``, ``iv``, ``type``, ``mid_price``.
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance required: pip install yfinance")

    spy = yf.Ticker("SPY")

    if spot_price is None:
        hist = spy.history(period="1d")
        if hist.empty:
            raise RuntimeError("Cannot fetch SPY spot price")
        spot_price = float(hist["Close"].iloc[-1])

    expirations = spy.options
    if not expirations:
        raise RuntimeError("No SPY options expirations available")

    # Take next 6 expirations
    exp_subset = expirations[:6]
    r = 0.05  # Risk-free rate assumption
    today = datetime.now()

    records = []
    for exp_str in exp_subset:
        exp_date = datetime.strptime(exp_str, "%Y-%m-%d")
        dte = (exp_date - today).days
        if dte <= 0:
            continue
        T = dte / 365.0

        try:
            chain = spy.option_chain(exp_str)
        except Exception as e:
            logger.warning("Failed to fetch chain for %s: %s", exp_str, e)
            continue

        for opt_type, df in [("call", chain.calls), ("put", chain.puts)]:
            for _, row in df.iterrows():
                K = float(row.get("strike", 0))
                bid = float(row.get("bid", 0))
                ask = float(row.get("ask", 0))
                mid = (bid + ask) / 2.0

                if mid <= 0 or K <= 0:
                    continue

                iv = implied_volatility(mid, spot_price, K, T, r, opt_type)
                delta = compute_delta(spot_price, K, T, r, iv if not np.isnan(iv) else 0.3, opt_type)

                records.append({
                    "strike": K,
                    "dte": dte,
                    "delta": abs(delta),
                    "iv": iv,
                    "type": opt_type,
                    "mid_price": mid,
                })

    result = pd.DataFrame(records)
    result = result.dropna(subset=["iv", "delta"])
    logger.info("Fetched %d valid option quotes across %d expirations", len(result), len(exp_subset))
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Full Pipeline
# ═══════════════════════════════════════════════════════════════════════════

def run_options_pipeline(
    output_dir: str = "data_lake/vol_surface",
) -> dict:
    """End-to-end options IV surface pipeline.

    1. Fetch SPY options chain
    2. Compute IV via Newton-Raphson
    3. Interpolate to fixed [5×25] grid
    4. Save to disk

    Returns
    -------
    dict
        Pipeline summary.
    """
    logger.info("Starting options IV surface pipeline")

    # 1. Fetch + compute IV
    iv_surface = fetch_spy_options_chain()

    # 2. Interpolate
    grid = interpolate_to_grid(iv_surface)

    # 3. Save
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    grid_path = out_path / "iv_grid.npy"
    np.save(grid_path, grid)

    surface_path = out_path / "iv_surface_raw.parquet"
    iv_surface.to_parquet(surface_path)

    summary = {
        "status": "ok",
        "grid_shape": list(grid.shape),
        "num_options": len(iv_surface),
        "grid_path": str(grid_path),
        "surface_path": str(surface_path),
    }
    logger.info("Options pipeline complete: %s", summary)
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = run_options_pipeline()
    print(result)
