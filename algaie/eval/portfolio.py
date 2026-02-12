"""
Portfolio construction, cost modeling, and volatility scaling.

Provides:
- ``build_portfolio``: top-K stock selection with multiple weighting modes
- ``compute_portfolio_returns``: gross/net returns with turnover-based costs
- ``compute_portfolio_metrics``: Sharpe, drawdown, CAGR, turnover, vol scaling
- ``build_equity_curve``: cumulative returns DataFrame

All functions operate on scored DataFrames produced by the selector pipeline.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Portfolio construction
# ═══════════════════════════════════════════════════════════════════════════

def build_portfolio(
    scored_df: pd.DataFrame,
    score_col: str = "score_final",
    target_col: str = "y_ret",
    k: int = 50,
    weighting: str = "equal",
    max_weight: float = 0.05,
    max_sector_weight: float = 0.25,
    sector_col: Optional[str] = None,
    market_neutral: bool = False,
    softmax_temp: float = 1.0,
    rebalance_period: int = 1,
) -> pd.DataFrame:
    """Build a rebalanced portfolio from scored cross-sections.

    Parameters
    ----------
    scored_df : DataFrame
        Must have: date, symbol, score_col, target_col.
    score_col : str
        Column with model scores for ranking.
    target_col : str
        Column with forward returns (for PnL computation).
    k : int
        Number of stocks in long (and short if market_neutral) leg.
    weighting : str
        ``"equal"`` — 1/K uniform.
        ``"score_proportional"`` — weight proportional to score.
        ``"softmax"`` — softmax-normalized scores with temperature.
    max_weight : float
        Maximum weight per stock (exposure cap, default 5%).
    max_sector_weight : float
        Maximum aggregate weight per sector (default 25%).
    sector_col : str or None
        Column name for sector. If None, sector cap is skipped.
    market_neutral : bool
        If True, long top-K + short bottom-K (dollar neutral).
    softmax_temp : float
        Temperature for softmax weighting mode.
    rebalance_period : int
        Rebalance every N-th date (default 1 = every date).
        Set to target_horizon (e.g. 10) for non-overlapping windows.

    Returns
    -------
    DataFrame with columns: date, symbol, weight, score, y_ret, side
        ``side`` is ``"long"`` or ``"short"``.
    """
    if score_col not in scored_df.columns:
        raise ValueError(f"score_col '{score_col}' not in DataFrame")

    # Get sorted unique dates and sample every N-th for non-overlapping windows
    all_dates = sorted(scored_df["date"].unique())
    if rebalance_period > 1:
        rebalance_dates = set(all_dates[::rebalance_period])
        logger.info(
            f"  Rebalance period={rebalance_period}: {len(rebalance_dates)}/{len(all_dates)} dates selected"
        )
    else:
        rebalance_dates = set(all_dates)

    results = []

    for dt, group in scored_df.groupby("date"):
        if dt not in rebalance_dates:
            continue

        group = group.dropna(subset=[score_col]).copy()
        if len(group) < k:
            continue

        # Rank by score descending
        ranked = group.sort_values(score_col, ascending=False)

        # Long leg: top K
        long_leg = ranked.head(k).copy()
        long_leg["side"] = "long"
        long_leg["weight"] = _compute_weights(
            long_leg[score_col].values, weighting, softmax_temp,
        )

        legs = [long_leg]

        # Short leg (market neutral)
        if market_neutral:
            short_leg = ranked.tail(k).copy()
            short_leg["side"] = "short"
            short_leg["weight"] = -_compute_weights(
                -short_leg[score_col].values, weighting, softmax_temp,
            )
            legs.append(short_leg)

        port = pd.concat(legs, ignore_index=True)

        # Apply exposure caps
        port["weight"] = _apply_exposure_caps(
            port, max_weight, max_sector_weight, sector_col,
        )

        port = port[["date", "symbol", "weight", score_col, target_col, "side"]].copy()
        port = port.rename(columns={score_col: "score", target_col: "y_ret"})
        results.append(port)

    if not results:
        return pd.DataFrame(columns=["date", "symbol", "weight", "score", "y_ret", "side"])

    return pd.concat(results, ignore_index=True)


def _compute_weights(scores: np.ndarray, weighting: str, temp: float) -> np.ndarray:
    """Compute portfolio weights from scores (for one leg)."""
    n = len(scores)
    if n == 0:
        return np.array([])

    if weighting == "equal":
        w = np.ones(n) / n
    elif weighting == "score_proportional":
        shifted = scores - scores.min() + 1e-8  # ensure positive
        w = shifted / shifted.sum()
    elif weighting == "softmax":
        logits = scores / max(temp, 1e-6)
        logits = logits - logits.max()  # numerical stability
        exp_logits = np.exp(logits)
        w = exp_logits / exp_logits.sum()
    else:
        raise ValueError(f"Unknown weighting mode: {weighting}")

    return w


def _apply_exposure_caps(
    port: pd.DataFrame,
    max_weight: float,
    max_sector_weight: float,
    sector_col: Optional[str],
) -> np.ndarray:
    """Apply per-stock and per-sector weight caps, then renormalize."""
    w = port["weight"].values.copy()
    signs = np.sign(w)
    abs_w = np.abs(w)

    # Per-stock cap
    clipped = np.minimum(abs_w, max_weight)

    # Per-sector cap (redistribute excess proportionally)
    if sector_col and sector_col in port.columns:
        sectors = port[sector_col].values
        for sector in np.unique(sectors):
            mask = sectors == sector
            sector_total = clipped[mask].sum()
            if sector_total > max_sector_weight:
                scale = max_sector_weight / sector_total
                clipped[mask] *= scale

    # Renormalize long and short legs separately
    long_mask = signs > 0
    short_mask = signs < 0

    if long_mask.any():
        clipped[long_mask] /= clipped[long_mask].sum()
    if short_mask.any():
        clipped[short_mask] /= clipped[short_mask].sum()

    return signs * clipped


# ═══════════════════════════════════════════════════════════════════════════
# Returns computation (with cost model)
# ═══════════════════════════════════════════════════════════════════════════

def compute_portfolio_returns(
    portfolio_df: pd.DataFrame,
    cost_bps: float = 10.0,
    slippage_multiplier: float = 1.0,
) -> pd.DataFrame:
    """Compute daily gross/net portfolio returns with turnover-based costs.

    Parameters
    ----------
    portfolio_df : DataFrame
        Output of ``build_portfolio``: date, symbol, weight, y_ret.
    cost_bps : float
        Round-trip cost in basis points (default 10 bps).
    slippage_multiplier : float
        Additional slippage proportional to turnover (default 1.0).

    Returns
    -------
    DataFrame with columns: date, gross_ret, turnover, cost, net_ret
        One row per date.
    """
    if portfolio_df.empty:
        return pd.DataFrame(columns=["date", "gross_ret", "turnover", "cost", "net_ret"])

    cost_frac = cost_bps / 10_000.0
    dates = sorted(portfolio_df["date"].unique())
    records = []

    prev_weights = {}  # symbol → weight

    for dt in dates:
        day = portfolio_df[portfolio_df["date"] == dt]
        curr_weights = dict(zip(day["symbol"], day["weight"]))
        curr_returns = dict(zip(day["symbol"], day["y_ret"]))

        # Gross return: weighted sum of forward returns
        gross_ret = sum(
            curr_weights.get(s, 0) * curr_returns.get(s, 0)
            for s in curr_weights
        )

        # Turnover: sum of absolute weight changes
        all_symbols = set(prev_weights) | set(curr_weights)
        turnover = sum(
            abs(curr_weights.get(s, 0) - prev_weights.get(s, 0))
            for s in all_symbols
        )

        # Cost = turnover * cost_bps * (1 + slippage)
        cost = turnover * cost_frac * (1.0 + slippage_multiplier * 0.5)

        net_ret = gross_ret - cost

        records.append({
            "date": dt,
            "gross_ret": gross_ret,
            "turnover": turnover,
            "cost": cost,
            "net_ret": net_ret,
        })

        prev_weights = curr_weights

    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════════════
# Portfolio metrics (Sharpe, drawdown, CAGR, vol scaling)
# ═══════════════════════════════════════════════════════════════════════════

def compute_portfolio_metrics(
    returns_df: pd.DataFrame,
    target_vol: Optional[float] = None,
    vol_lookback: int = 20,
    vol_clamp: Tuple[float, float] = (0.5, 2.0),
    trading_days_per_year: int = 252,
    periods_per_year: Optional[int] = None,
) -> Dict[str, float]:
    """Compute portfolio-level performance metrics.

    Parameters
    ----------
    returns_df : DataFrame
        Must have: date, gross_ret, net_ret, turnover, cost.
    target_vol : float or None
        If set, apply volatility scaling (annualized target, e.g. 0.15).
    vol_lookback : int
        Rolling window for realized vol (default 20 days).
    vol_clamp : (float, float)
        Min/max clamp for vol scaling factor.
    trading_days_per_year : int
        Annualization factor.

    periods_per_year : int or None
        Number of rebalance periods per year. If None, defaults to
        trading_days_per_year (daily rebalance). For 10d non-overlapping
        rebalance, set to 252/10 = 25.

    Returns
    -------
    dict with keys: ann_return_gross, ann_return_net, ann_vol_gross, ann_vol_net,
        gross_sharpe, net_sharpe, max_drawdown, cagr_gross, cagr_net,
        avg_turnover, ann_turnover, avg_holding_period,
        vol_scaled_sharpe (if target_vol set),
        vol_scaling_effect (if target_vol set).
    """
    if returns_df.empty or len(returns_df) < 2:
        return {k: 0.0 for k in [
            "ann_return_gross", "ann_return_net", "ann_vol_gross", "ann_vol_net",
            "gross_sharpe", "net_sharpe", "max_drawdown",
            "cagr_gross", "cagr_net", "avg_turnover", "ann_turnover",
        ]}

    # Use periods_per_year for annualization (handles non-daily rebalance)
    ppy = periods_per_year if periods_per_year is not None else trading_days_per_year

    n = len(returns_df)
    gross = returns_df["gross_ret"].values
    net = returns_df["net_ret"].values
    turnover = returns_df["turnover"].values

    # Annualized return and vol
    ann_ret_g = float(np.mean(gross) * ppy)
    ann_ret_n = float(np.mean(net) * ppy)
    ann_vol_g = float(np.std(gross, ddof=1) * np.sqrt(ppy))
    ann_vol_n = float(np.std(net, ddof=1) * np.sqrt(ppy))

    # Sharpe
    gross_sharpe = ann_ret_g / max(ann_vol_g, 1e-8)
    net_sharpe = ann_ret_n / max(ann_vol_n, 1e-8)

    # Max drawdown (from cumulative net returns)
    cum_net = np.cumsum(net)
    running_max = np.maximum.accumulate(cum_net)
    drawdowns = cum_net - running_max
    max_dd = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0

    # CAGR
    years = n / ppy
    total_gross = float(np.sum(gross))
    total_net = float(np.sum(net))
    cagr_g = (np.exp(total_gross) ** (1 / max(years, 0.01))) - 1 if years > 0 else 0.0
    cagr_n = (np.exp(total_net) ** (1 / max(years, 0.01))) - 1 if years > 0 else 0.0

    # Turnover
    avg_turnover = float(np.mean(turnover))
    ann_turnover = avg_turnover * ppy

    # Average holding period (approximate: 2 / avg_turnover if turnover > 0)
    avg_hold = 2.0 / max(avg_turnover, 1e-8) if avg_turnover > 0.01 else float("inf")

    result = {
        "ann_return_gross": round(ann_ret_g, 6),
        "ann_return_net": round(ann_ret_n, 6),
        "ann_vol_gross": round(ann_vol_g, 6),
        "ann_vol_net": round(ann_vol_n, 6),
        "gross_sharpe": round(gross_sharpe, 4),
        "net_sharpe": round(net_sharpe, 4),
        "max_drawdown": round(max_dd, 6),
        "cagr_gross": round(float(cagr_g), 6),
        "cagr_net": round(float(cagr_n), 6),
        "avg_turnover": round(avg_turnover, 6),
        "ann_turnover": round(ann_turnover, 4),
        "avg_holding_period": round(min(avg_hold, 999.0), 1),
    }

    result["periods_per_year"] = ppy
    result["rebalance_period_days"] = round(trading_days_per_year / ppy, 1)

    # Volatility scaling
    if target_vol is not None and target_vol > 0 and n > vol_lookback:
        scaled_net = _apply_vol_scaling(
            net, target_vol, vol_lookback, vol_clamp, ppy,
        )
        ann_ret_vs = float(np.mean(scaled_net) * ppy)
        ann_vol_vs = float(np.std(scaled_net, ddof=1) * np.sqrt(ppy))
        vs_sharpe = ann_ret_vs / max(ann_vol_vs, 1e-8)

        result["vol_scaled_return"] = round(ann_ret_vs, 6)
        result["vol_scaled_vol"] = round(ann_vol_vs, 6)
        result["vol_scaled_sharpe"] = round(vs_sharpe, 4)
        result["vol_scaling_effect"] = round(vs_sharpe - net_sharpe, 4)

    return result


def _apply_vol_scaling(
    returns: np.ndarray,
    target_vol: float,
    lookback: int,
    clamp: Tuple[float, float],
    trading_days: int,
) -> np.ndarray:
    """Apply target-vol scaling to a return stream."""
    scaled = np.copy(returns)
    for i in range(lookback, len(returns)):
        window = returns[i - lookback:i]
        realized = float(np.std(window, ddof=1) * np.sqrt(trading_days))
        if realized < 1e-8:
            scale = 1.0
        else:
            scale = target_vol / realized
        scale = np.clip(scale, clamp[0], clamp[1])
        scaled[i] = returns[i] * scale
    return scaled


# ═══════════════════════════════════════════════════════════════════════════
# Equity curve
# ═══════════════════════════════════════════════════════════════════════════

def build_equity_curve(returns_df: pd.DataFrame) -> pd.DataFrame:
    """Build cumulative equity curve from portfolio returns.

    Returns DataFrame with: date, cum_gross, cum_net.
    """
    if returns_df.empty:
        return pd.DataFrame(columns=["date", "cum_gross", "cum_net"])

    df = returns_df[["date", "gross_ret", "net_ret"]].copy()
    df["cum_gross"] = np.exp(np.cumsum(df["gross_ret"].values)) - 1
    df["cum_net"] = np.exp(np.cumsum(df["net_ret"].values)) - 1
    return df[["date", "cum_gross", "cum_net"]]


# ═══════════════════════════════════════════════════════════════════════════
# Regime breakdown
# ═══════════════════════════════════════════════════════════════════════════

def compute_regime_breakdown(
    scored_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    cs_col: str = "z_cs_tail_30_std",
    periods_per_year: int = 252,
) -> Dict[str, Dict]:
    """Break down portfolio performance by CS regime terciles.

    Returns dict with keys: "low_stress", "mid_stress", "high_stress",
    each containing: n_dates, mean_gross_ret, mean_net_ret, ann_sharpe.
    Also adds "top_20pct_stress" for the most extreme dates.
    """
    if scored_df.empty or returns_df.empty or cs_col not in scored_df.columns:
        return {}

    # Get per-date z_cs (take first per date since it's date-level)
    date_cs = scored_df.groupby("date")[cs_col].first().reset_index()
    merged = returns_df.merge(date_cs, on="date", how="left")

    if merged[cs_col].isna().all():
        return {}

    # Terciles
    tercile_edges = merged[cs_col].quantile([1/3, 2/3]).values
    low = merged[merged[cs_col] <= tercile_edges[0]]
    mid = merged[(merged[cs_col] > tercile_edges[0]) & (merged[cs_col] <= tercile_edges[1])]
    high = merged[merged[cs_col] > tercile_edges[1]]

    # Top 20% stress
    top_20_thresh = merged[cs_col].quantile(0.80)
    top_20 = merged[merged[cs_col] >= top_20_thresh]

    ppy = periods_per_year

    def _bucket_stats(sub: pd.DataFrame, label: str) -> Dict:
        if sub.empty:
            return {"n_dates": 0, "mean_gross_ret": 0, "mean_net_ret": 0, "ann_sharpe": 0}
        n = len(sub)
        mg = float(sub["gross_ret"].mean())
        mn = float(sub["net_ret"].mean())
        std = float(sub["net_ret"].std(ddof=1)) if n > 1 else 1e-8
        sharpe = (mn * ppy) / max(std * np.sqrt(ppy), 1e-8)
        return {
            "n_dates": n,
            "mean_gross_ret": round(mg, 6),
            "mean_net_ret": round(mn, 6),
            "ann_sharpe": round(sharpe, 4),
        }

    return {
        "low_stress": _bucket_stats(low, "low"),
        "mid_stress": _bucket_stats(mid, "mid"),
        "high_stress": _bucket_stats(high, "high"),
        "top_20pct_stress": _bucket_stats(top_20, "top_20"),
    }
