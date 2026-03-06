"""
Vectorised Black-Scholes greeks engine (European approximation).

Suitable for index ETF options (SPY/QQQ/IWM) in v1.
All functions accept scalar or numpy array inputs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats  # type: ignore[import-untyped]


# ---------------------------------------------------------------------------
# Core Black-Scholes helpers
# ---------------------------------------------------------------------------

def _d1(
    S: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    r: np.ndarray,
    q: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    """Compute d1 in Black-Scholes formula."""
    with np.errstate(divide="ignore", invalid="ignore"):
        num = np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T
        den = sigma * np.sqrt(T)
        return np.where(den > 0, num / den, 0.0)


def _d2(
    S: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    r: np.ndarray,
    q: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    return _d1(S, K, T, r, q, sigma) - sigma * np.sqrt(T)


# ---------------------------------------------------------------------------
# Price
# ---------------------------------------------------------------------------

def bs_price(
    S: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    r: float | np.ndarray,
    sigma: float | np.ndarray,
    option_type: str = "call",
    q: float | np.ndarray = 0.0,
) -> np.ndarray:
    """Black-Scholes option price.

    Parameters
    ----------
    S : spot price
    K : strike
    T : time to expiry in years
    r : risk-free rate (annualised)
    sigma : implied volatility (annualised)
    option_type : 'call' or 'put'
    q : continuous dividend yield
    """
    S, K, T, r, sigma, q = (np.asarray(x, dtype=float) for x in (S, K, T, r, sigma, q))
    d1 = _d1(S, K, T, r, q, sigma)
    d2 = _d2(S, K, T, r, q, sigma)

    if option_type == "call":
        price = S * np.exp(-q * T) * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
    elif option_type == "put":
        price = K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * np.exp(-q * T) * stats.norm.cdf(-d1)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")
    return np.maximum(price, 0.0)


# ---------------------------------------------------------------------------
# Greeks
# ---------------------------------------------------------------------------

def bs_delta(
    S: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    r: float | np.ndarray,
    sigma: float | np.ndarray,
    option_type: str = "call",
    q: float | np.ndarray = 0.0,
) -> np.ndarray:
    """Black-Scholes delta."""
    S, K, T, r, sigma, q = (np.asarray(x, dtype=float) for x in (S, K, T, r, sigma, q))
    d1 = _d1(S, K, T, r, q, sigma)
    if option_type == "call":
        return np.exp(-q * T) * stats.norm.cdf(d1)
    elif option_type == "put":
        return np.exp(-q * T) * (stats.norm.cdf(d1) - 1.0)
    raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")


def bs_gamma(
    S: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    r: float | np.ndarray,
    sigma: float | np.ndarray,
    q: float | np.ndarray = 0.0,
) -> np.ndarray:
    """Black-Scholes gamma (same for calls and puts)."""
    S, K, T, r, sigma, q = (np.asarray(x, dtype=float) for x in (S, K, T, r, sigma, q))
    d1 = _d1(S, K, T, r, q, sigma)
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = S * sigma * np.sqrt(T)
        gamma = np.exp(-q * T) * stats.norm.pdf(d1) / denom
        return np.where(denom > 0, gamma, 0.0)


def bs_vega(
    S: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    r: float | np.ndarray,
    sigma: float | np.ndarray,
    q: float | np.ndarray = 0.0,
) -> np.ndarray:
    """Black-Scholes vega (per 1% move in vol → divide by 100 for per-point).

    Returns vega per 1 vol point (i.e. multiply by 0.01 for a 1% vol move).
    Convention: returned as price sensitivity per unit vol change.
    """
    S, K, T, r, sigma, q = (np.asarray(x, dtype=float) for x in (S, K, T, r, sigma, q))
    d1 = _d1(S, K, T, r, q, sigma)
    return S * np.exp(-q * T) * stats.norm.pdf(d1) * np.sqrt(T)


def bs_theta(
    S: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    r: float | np.ndarray,
    sigma: float | np.ndarray,
    option_type: str = "call",
    q: float | np.ndarray = 0.0,
) -> np.ndarray:
    """Black-Scholes theta (per calendar day, negative for long positions)."""
    S, K, T, r, sigma, q = (np.asarray(x, dtype=float) for x in (S, K, T, r, sigma, q))
    d1 = _d1(S, K, T, r, q, sigma)
    d2 = _d2(S, K, T, r, q, sigma)

    with np.errstate(divide="ignore", invalid="ignore"):
        sqrt_T = np.sqrt(T)
        term1 = -(S * np.exp(-q * T) * stats.norm.pdf(d1) * sigma) / (2.0 * sqrt_T)

    if option_type == "call":
        term2 = -r * K * np.exp(-r * T) * stats.norm.cdf(d2)
        term3 = q * S * np.exp(-q * T) * stats.norm.cdf(d1)
        theta_annual = term1 + term2 + term3
    elif option_type == "put":
        term2 = r * K * np.exp(-r * T) * stats.norm.cdf(-d2)
        term3 = -q * S * np.exp(-q * T) * stats.norm.cdf(-d1)
        theta_annual = term1 + term2 + term3
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

    # Convert from per-year to per-calendar-day
    return theta_annual / 365.0


# ---------------------------------------------------------------------------
# Compute full greeks frame for a chain DataFrame
# ---------------------------------------------------------------------------

def compute_greeks_frame(chain: pd.DataFrame) -> pd.DataFrame:
    """Add delta, gamma, vega, theta columns to an option chain DataFrame.

    Requires columns: underlying_price, strike, dte, implied_vol,
    risk_free_rate, dividend_yield, option_type.
    """
    df = chain.copy()

    S = df["underlying_price"].to_numpy(dtype=float)
    K = df["strike"].to_numpy(dtype=float)
    T = df["dte"].to_numpy(dtype=float) / 365.0
    r = df["risk_free_rate"].to_numpy(dtype=float)
    sigma = df["implied_vol"].to_numpy(dtype=float)
    q = df["dividend_yield"].to_numpy(dtype=float)

    # Separate calls and puts
    is_call = (df["option_type"].str.lower() == "call").to_numpy()

    delta = np.zeros(len(df))
    theta = np.zeros(len(df))

    if is_call.any():
        idx = is_call
        delta[idx] = bs_delta(S[idx], K[idx], T[idx], r[idx], sigma[idx], "call", q[idx])
        theta[idx] = bs_theta(S[idx], K[idx], T[idx], r[idx], sigma[idx], "call", q[idx])

    if (~is_call).any():
        idx = ~is_call
        delta[idx] = bs_delta(S[idx], K[idx], T[idx], r[idx], sigma[idx], "put", q[idx])
        theta[idx] = bs_theta(S[idx], K[idx], T[idx], r[idx], sigma[idx], "put", q[idx])

    gamma = bs_gamma(S, K, T, r, sigma, q)
    vega = bs_vega(S, K, T, r, sigma, q)

    df["delta"] = delta
    df["gamma"] = gamma
    df["vega"] = vega
    df["theta"] = theta
    return df
