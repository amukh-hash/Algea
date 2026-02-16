"""Trade-proxy validation: cost-aware rank-based evaluation.

CHANGE LOG (2026-02-14):
  - D4: Refactored to use TradeProxyConfig with explicit score_semantics
    and baseline_semantics. Alpha normalization via to_alpha() / baseline_to_alpha().
    Score polarity: longs = highest alpha, shorts = lowest alpha.
    Returns computed in equity-based units (proxy_equity_usd).
    Deterministic tie-breaking via secondary sort on root.
  - D5: Added compute_trade_proxy_diagnostics() with skew, kurtosis, CVaR,
    zero-return fraction, insufficient-day count.

Score direction convention
--------------------------
  Under default ``score_semantics="alpha_high_long"`` (matching y = -r_oc):
    - **LONG = highest alpha** (expected higher r_oc → reversal up)
    - **SHORT = lowest alpha** (expected lower r_oc → reversal down)
  Baseline with ``baseline_semantics="r_co_meanrevert"``:
    - baseline_alpha = -r_co (gap up → short, gap down → long)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .types import TradeProxyConfig, TradeProxyReport, TradeProxyRealism

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG = TradeProxyConfig()
_DEFAULT_REALISM = TradeProxyRealism()


# ---------------------------------------------------------------------------
# Alpha normalization (D4)
# ---------------------------------------------------------------------------

def to_alpha(
    df: pd.DataFrame,
    *,
    score_col: str,
    semantics: str = "alpha_high_long",
) -> pd.Series:
    """Normalize score to alpha: higher alpha = more attractive for long.

    Parameters
    ----------
    df : DataFrame containing score_col
    score_col : column name with raw scores
    semantics :
        - "alpha_high_long": alpha = score (model trained on y = -r_oc)
        - "alpha_low_long": alpha = -score (legacy: lower score = long)
    """
    if semantics == "alpha_high_long":
        return df[score_col].copy()
    elif semantics == "alpha_low_long":
        return -df[score_col]
    else:
        raise ValueError(f"Unknown score_semantics: {semantics!r}")


def baseline_to_alpha(
    df: pd.DataFrame,
    *,
    r_co_col: str = "r_co",
    semantics: str = "r_co_meanrevert",
) -> pd.Series:
    """Compute baseline alpha from r_co.

    Parameters
    ----------
    semantics :
        - "r_co_meanrevert": alpha = -r_co (gap up → short, gap down → long)
        - "r_co_momentum": alpha = r_co (follow the gap)
    """
    if semantics == "r_co_meanrevert":
        return -df[r_co_col]
    elif semantics == "r_co_momentum":
        return df[r_co_col].copy()
    else:
        raise ValueError(f"Unknown baseline_semantics: {semantics!r}")


# ---------------------------------------------------------------------------
# Diagnostics (D5)
# ---------------------------------------------------------------------------

def compute_trade_proxy_diagnostics(
    daily_returns: pd.Series,
    n_insufficient_days: int = 0,
) -> Dict[str, Any]:
    """Compute extended diagnostics for a daily return series.

    Returns dict with: n_days, mean, vol, skew, kurtosis, worst_1pct,
    cvar_1pct, max_drawdown, zero_return_frac, n_insufficient_days.
    """
    n = len(daily_returns)
    if n == 0:
        return {
            "n_days": 0, "mean": 0.0, "vol": 0.0, "skew": 0.0,
            "kurtosis": 0.0, "worst_1pct": 0.0, "cvar_1pct": 0.0,
            "max_drawdown": 0.0, "zero_return_frac": 1.0,
            "n_zero_return_days": 0, "n_insufficient_days": n_insufficient_days,
        }

    mean = float(daily_returns.mean())
    vol = float(daily_returns.std())
    skew_val = float(daily_returns.skew()) if n >= 3 else 0.0
    kurt_val = float(daily_returns.kurtosis()) if n >= 4 else 0.0
    worst_1pct = float(daily_returns.quantile(0.01))

    # CVaR (Expected Shortfall) at 1%
    threshold = daily_returns.quantile(0.01)
    tail = daily_returns[daily_returns <= threshold]
    cvar_1pct = float(tail.mean()) if len(tail) > 0 else float(worst_1pct)

    # Max drawdown
    cumret = (1 + daily_returns).cumprod()
    peak = cumret.cummax()
    drawdown = ((cumret - peak) / peak)
    max_dd = float(drawdown.min())

    # Zero return days
    n_zero = int((daily_returns.abs() < 1e-15).sum())

    return {
        "n_days": n,
        "mean": mean,
        "vol": vol,
        "skew": skew_val,
        "kurtosis": kurt_val,
        "worst_1pct": worst_1pct,
        "cvar_1pct": cvar_1pct,
        "max_drawdown": max_dd,
        "zero_return_frac": n_zero / max(n, 1),
        "n_zero_return_days": n_zero,
        "n_insufficient_days": n_insufficient_days,
    }


# ---------------------------------------------------------------------------
# Core: single-day proxy return (D4 refactored)
# ---------------------------------------------------------------------------

def _daily_proxy_return(
    group: pd.DataFrame,
    alpha_col: str,
    ret_col: str,
    cfg: TradeProxyConfig,
    *,
    root_col: str = "root",
    realism: Optional[TradeProxyRealism] = None,
    day_hash: int = 0,
) -> float:
    """Compute single-day proxy portfolio return in equity-fraction units.

    Strategy:
      - LONG = top_k by alpha (highest alpha)
      - SHORT = bottom_k by alpha (lowest alpha)

    Returns net daily return as fraction of proxy_equity_usd.
    """
    top_k = cfg.top_k

    if len(group) < 2 * top_k:
        if not cfg.allow_insufficient_universe:
            return 0.0
        # Shrink k to fit
        top_k = max(1, len(group) // 2)

    # D4: Deterministic tie-breaking — sort by alpha then root
    secondary = root_col if root_col in group.columns else group.columns[0]
    ranked = group.sort_values([alpha_col, secondary], ascending=[True, True])

    # Longs = highest alpha (tail), Shorts = lowest alpha (head)
    shorts = ranked.head(top_k)
    longs = ranked.tail(top_k)

    # --- Shock detection ---
    instrument_shocked: set[str] = set()
    is_shock = False

    if "shock_flag" in group.columns:
        is_shock = (group["shock_flag"] == 1.0).any()
        if realism is not None and root_col in group.columns:
            if "shock_score" in group.columns:
                threshold = realism.shock_z_threshold
                for _, row in group.iterrows():
                    if row.get("shock_score", 0.0) > threshold:
                        instrument_shocked.add(row[root_col])
            else:
                for _, row in group[group["shock_flag"] == 1.0].iterrows():
                    if root_col in row.index:
                        instrument_shocked.add(row[root_col])
    elif "r_co" in group.columns:
        is_shock = (group["r_co"].abs() > group["r_co"].abs().quantile(0.9))
        is_shock = bool(is_shock.any())

    shock_slip_mult = cfg.shock_slippage_mult if is_shock else 1.0
    shock_gross_mult = cfg.shock_gross_mult

    # --- Sizing (D4: equity-based) ---
    equity_usd = cfg.proxy_equity_usd
    n_positions = 2 * top_k
    per_leg_notional_usd = equity_usd * cfg.gross_target / max(n_positions, 1)

    # --- Gross PnL ---
    long_pnl_usd = 0.0
    for _, row in longs.iterrows():
        root = row.get(root_col, "") if root_col in row.index else ""
        inst_scale = shock_gross_mult if root in instrument_shocked else 1.0
        long_pnl_usd += per_leg_notional_usd * row[ret_col] * inst_scale

    short_pnl_usd = 0.0
    for _, row in shorts.iterrows():
        root = row.get(root_col, "") if root_col in row.index else ""
        inst_scale = shock_gross_mult if root in instrument_shocked else 1.0
        short_pnl_usd += per_leg_notional_usd * (-row[ret_col]) * inst_scale

    gross_pnl_usd = long_pnl_usd + short_pnl_usd

    # --- Slippage (D4: equity-based) ---
    total_slip_usd = 0.0
    total_commission_usd = 0.0
    positions = pd.concat([longs, shorts])
    for _, row in positions.iterrows():
        root = row.get(root_col, "") if root_col in row.index else ""
        if realism is not None and root:
            slip_open = realism.slippage_open_for_root(root, cfg.slippage_bps_open)
            slip_close = realism.slippage_close_for_root(root, cfg.slippage_bps_close)
            cost = realism.cost_for_root(root, cfg.cost_per_contract)
        else:
            slip_open = cfg.slippage_bps_open
            slip_close = cfg.slippage_bps_close
            cost = cfg.cost_per_contract

        # Slippage in USD: notional * bps / 1e4
        slip_bps = (slip_open + slip_close) * shock_slip_mult
        total_slip_usd += per_leg_notional_usd * slip_bps / 1e4

        # Commission: fixed USD per contract (D4: NOT divided by notional)
        total_commission_usd += cost

    # --- Partial fill simulation (deterministic) ---
    fill_scale = cfg.fill_scale
    if realism is not None and realism.partial_fill_prob_shock > 0 and is_shock:
        import hashlib
        h = hashlib.sha256(f"{realism.partial_fill_seed}:{day_hash}".encode()).hexdigest()
        p = int(h[:8], 16) / 0xFFFFFFFF
        if p < realism.partial_fill_prob_shock:
            fill_scale *= 0.5

    # --- Net return as fraction of equity ---
    gross_return = gross_pnl_usd / equity_usd
    slip_return = total_slip_usd / equity_usd
    commission_return = total_commission_usd / equity_usd

    net_return = gross_return * fill_scale - slip_return - commission_return
    return float(net_return)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_trade_proxy(
    dataset: pd.DataFrame,
    preds: pd.Series,
    config: Optional[Dict[str, Any] | TradeProxyConfig] = None,
    output_dir: Optional[str | Path] = None,
    realism: Optional[TradeProxyRealism] = None,
    gross_schedule: Optional[pd.Series] = None,
) -> TradeProxyReport:
    """Evaluate model predictions using a trade-proxy strategy.

    Parameters
    ----------
    dataset
        Panel DataFrame with columns: ``trading_day``, ``root``, ``r_oc``
        (the label), and optionally ``close``, ``multiplier``.
    preds
        Predictions indexed identically to *dataset*.
    config
        Trade proxy config: a dict or TradeProxyConfig. Dict keys are
        mapped to TradeProxyConfig fields for backward compatibility.
    output_dir
        If provided, persist daily returns and report.
    realism
        Per-root cost/slippage overrides.
    gross_schedule
        Optional per-day gross scale (index = trading_day).  Each day's
        ``gross_target`` is multiplied by the corresponding value.
        Produced by :func:`exposure_policy.compute_gross_schedule`.

    Returns
    -------
    TradeProxyReport
    """
    # Build config
    if isinstance(config, TradeProxyConfig):
        cfg = config
    elif config is not None:
        cfg = TradeProxyConfig.from_dict(config)
    else:
        cfg = _DEFAULT_CONFIG

    top_k = cfg.top_k

    df = dataset.copy()
    df["pred"] = preds.values if hasattr(preds, "values") else preds

    # --- D4: Alpha normalization ---
    df["model_alpha"] = to_alpha(df, score_col="pred", semantics=cfg.score_semantics)

    # --- D4: Baseline alpha ---
    r_co_col = "r_co" if "r_co" in df.columns else "ret_co"
    if r_co_col in df.columns:
        df["baseline_alpha"] = baseline_to_alpha(
            df, r_co_col=r_co_col, semantics=cfg.baseline_semantics,
        )
        # Legacy: keep baseline_score as r_co for backward compat
        df["baseline_score"] = df[r_co_col]
    elif "signal" in df.columns:
        # Legacy: signal was -ret_co
        df["baseline_alpha"] = df["signal"]  # -ret_co = mean-revert alpha
        df["baseline_score"] = -df["signal"]
    else:
        df["baseline_alpha"] = 0.0
        df["baseline_score"] = 0.0

    ret_col = "r_oc" if "r_oc" in df.columns else ("ret_oc" if "ret_oc" in df.columns else "y")

    # Ensure trading_day column
    if "trading_day" not in df.columns and df.index.names[0] == "trading_day":
        df = df.reset_index()

    # Ensure root column for per-root costs
    root_col = "root" if "root" in df.columns else "instrument"
    _realism = realism or _DEFAULT_REALISM

    n_insufficient = 0

    # --- R4: gross_schedule support ---
    _gross_scales: Dict[Any, float] = {}  # day → scale (populated below)
    if gross_schedule is not None:
        for day_key, scale_val in gross_schedule.items():
            _gross_scales[day_key] = float(scale_val)

    # --- Compute daily returns for model ---
    def _day_return(g: pd.DataFrame, alpha_col: str) -> float:
        nonlocal n_insufficient
        if len(g) < 2 * top_k:
            n_insufficient += 1
        day_val = g["trading_day"].iloc[0] if "trading_day" in g.columns else 0
        day_hash = hash(str(day_val))

        # Apply gross_schedule scaling if provided
        day_cfg = cfg
        if _gross_scales:
            scale = _gross_scales.get(day_val, 1.0)
            from dataclasses import replace as _dc_replace
            day_cfg = _dc_replace(cfg, gross_target=cfg.gross_target * scale)

        return _daily_proxy_return(
            g, alpha_col, ret_col, day_cfg,
            root_col=root_col,
            realism=_realism,
            day_hash=day_hash,
        )

    model_daily = df.groupby("trading_day").apply(
        lambda g: _day_return(g, "model_alpha"),
        include_groups=False,
    ).rename("model_return")

    n_insufficient_model = n_insufficient
    n_insufficient = 0

    # --- Compute daily returns for baseline ---
    baseline_daily = df.groupby("trading_day").apply(
        lambda g: _day_return(g, "baseline_alpha"),
        include_groups=False,
    ).rename("baseline_return")

    # --- Combine ---
    daily = pd.concat([model_daily, baseline_daily], axis=1).dropna()
    if daily.empty:
        return TradeProxyReport(
            sharpe_model=0.0, sharpe_baseline=0.0, hit_rate=0.0,
            max_drawdown=0.0, mean_daily_return=0.0, worst_1pct_return=0.0,
            gate_passed=False,
        )

    # --- Metrics ---
    ann_factor = np.sqrt(252)

    mean_m = daily["model_return"].mean()
    std_m = daily["model_return"].std()
    sharpe_m = (mean_m / std_m * ann_factor) if std_m > 1e-12 else 0.0

    mean_b = daily["baseline_return"].mean()
    std_b = daily["baseline_return"].std()
    sharpe_b = (mean_b / std_b * ann_factor) if std_b > 1e-12 else 0.0

    hit_rate = float((daily["model_return"] > 0).mean())

    # D5: Full diagnostics
    diag = compute_trade_proxy_diagnostics(
        daily["model_return"], n_insufficient_days=n_insufficient_model,
    )

    # --- Gate ---
    gate_passed = True
    if cfg.require_not_worse_than_baseline:
        if sharpe_m < sharpe_b - cfg.sharpe_tolerance:
            gate_passed = False
            logger.warning(
                "Trade proxy FAIL: model Sharpe %.3f < baseline Sharpe %.3f - tol %.3f",
                sharpe_m, sharpe_b, cfg.sharpe_tolerance,
            )

    # --- Persist ---
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        report_data = {
            "sharpe_model": sharpe_m,
            "sharpe_baseline": sharpe_b,
            "hit_rate": hit_rate,
            "max_drawdown": diag["max_drawdown"],
            "mean_daily_return": float(mean_m),
            "worst_1pct_return": diag["worst_1pct"],
            "gate_passed": gate_passed,
            "n_days": diag["n_days"],
            "diagnostics": diag,
            "config": cfg.to_dict(),
            "realism": _realism.to_dict(),
        }
        (out / "trade_proxy_report.json").write_text(
            json.dumps(report_data, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        daily.to_parquet(out / "trade_proxy_daily.parquet")

    # --- R4: gating diagnostics ---
    pct_crash = None
    pct_caution = None
    avg_scale = None
    if _gross_scales:
        scales = pd.Series(_gross_scales)
        n_total = len(scales)
        if n_total > 0:
            pct_crash = float((scales == 0.0).sum() / n_total)
            pct_caution = float(((scales > 0.0) & (scales < 1.0)).sum() / n_total)
            avg_scale = float(scales.mean())

    return TradeProxyReport(
        sharpe_model=sharpe_m,
        sharpe_baseline=sharpe_b,
        hit_rate=hit_rate,
        max_drawdown=diag["max_drawdown"],
        mean_daily_return=float(mean_m),
        worst_1pct_return=diag["worst_1pct"],
        gate_passed=gate_passed,
        n_days=diag["n_days"],
        vol=diag["vol"],
        skew=diag["skew"],
        kurtosis=diag["kurtosis"],
        cvar_1pct=diag["cvar_1pct"],
        n_zero_return_days=diag["n_zero_return_days"],
        n_insufficient_days=diag["n_insufficient_days"],
        pct_crash_days=pct_crash,
        pct_caution_days=pct_caution,
        avg_gross_scale=avg_scale,
    )


# ---------------------------------------------------------------------------
# F3: Tier2 market-impact model + realism ladder
# ---------------------------------------------------------------------------

def compute_tier2_impact_bps(
    n_contracts: float,
    adv: float,
    impact_cfg: "Tier2ImpactConfig | None" = None,
) -> float:
    """Compute spread+impact cost in bps for a given trade size and ADV.

    impact_bps = base_bps + k * (n_contracts / max(adv, min_adv))^p

    If impact_bps > impact_cap_bps and downscale_on_cap is True,
    returns the capped value (caller should note capacity constraint).
    """
    from sleeves.cooc_reversal_futures.pipeline.types import Tier2ImpactConfig

    if impact_cfg is None:
        impact_cfg = Tier2ImpactConfig()

    participation = n_contracts / max(adv, impact_cfg.min_adv_contracts)
    impact = impact_cfg.base_bps + impact_cfg.k * (participation ** impact_cfg.p)

    if impact > impact_cfg.impact_cap_bps and impact_cfg.downscale_on_cap:
        impact = impact_cfg.impact_cap_bps

    return float(impact)


def evaluate_realism_ladder(
    dataset: pd.DataFrame,
    preds: pd.Series,
    *,
    tier2_cfg: "Tier2ImpactConfig | None" = None,
    base_config: Optional[Dict[str, Any] | TradeProxyConfig] = None,
    adv_by_root: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Evaluate predictions under each realism tier.

    Parameters
    ----------
    adv_by_root
        Per-root average daily volume in contracts/day.
        When provided, Tier2 impact is computed per-root instead of
        using a single avg ADV=200 fallback.

    Returns a dict keyed by tier name with Sharpe ratios and diagnostics.
    """
    from sleeves.cooc_reversal_futures.pipeline.types import (
        RealismTier, Tier2ImpactConfig,
    )

    if tier2_cfg is None:
        tier2_cfg = Tier2ImpactConfig()

    results: Dict[str, Any] = {}

    # Tier 0: zero cost
    tier0_config = {
        "cost_per_contract": 0.0,
        "slippage_bps_open": 0.0,
        "slippage_bps_close": 0.0,
    }
    if isinstance(base_config, TradeProxyConfig):
        t0_dict = base_config.to_dict()
        t0_dict.update(tier0_config)
    elif base_config is not None:
        t0_dict = {**base_config, **tier0_config}
    else:
        t0_dict = tier0_config
    report_t0 = evaluate_trade_proxy(dataset, preds, config=t0_dict)
    results["TIER0_ZERO_COST"] = {
        "sharpe": report_t0.sharpe_model,
        "hit_rate": report_t0.hit_rate,
        "max_drawdown": report_t0.max_drawdown,
        "report": report_t0,
    }

    # Tier 1: simple flat cost
    report_t1 = evaluate_trade_proxy(dataset, preds, config=base_config)
    results["TIER1_SIMPLE_COST"] = {
        "sharpe": report_t1.sharpe_model,
        "hit_rate": report_t1.hit_rate,
        "max_drawdown": report_t1.max_drawdown,
        "report": report_t1,
    }

    # Tier 2: spread + market impact
    # Use per-root ADV when available, otherwise weighted-average fallback
    if adv_by_root and "root" in dataset.columns:
        roots_in_data = dataset["root"].unique()
        total_adv = 0.0
        n_roots = 0
        for r in roots_in_data:
            adv = adv_by_root.get(r, tier2_cfg.min_adv_contracts)
            total_adv += adv
            n_roots += 1
        avg_adv = total_adv / max(n_roots, 1)
    else:
        avg_adv = 200.0  # legacy fallback

    avg_impact_bps = compute_tier2_impact_bps(
        n_contracts=1.0, adv=avg_adv, impact_cfg=tier2_cfg,
    )
    tier2_slip_open = avg_impact_bps
    tier2_slip_close = avg_impact_bps

    if isinstance(base_config, TradeProxyConfig):
        t2_dict = base_config.to_dict()
    elif base_config is not None:
        t2_dict = dict(base_config)
    else:
        t2_dict = {}
    t2_dict["slippage_bps_open"] = tier2_slip_open
    t2_dict["slippage_bps_close"] = tier2_slip_close

    report_t2 = evaluate_trade_proxy(dataset, preds, config=t2_dict)
    results["TIER2_SPREAD_IMPACT"] = {
        "sharpe": report_t2.sharpe_model,
        "hit_rate": report_t2.hit_rate,
        "max_drawdown": report_t2.max_drawdown,
        "impact_bps_model": avg_impact_bps,
        "adv_used": avg_adv,
        "report": report_t2,
    }

    # Summary
    results["summary"] = {
        "tier0_sharpe": report_t0.sharpe_model,
        "tier1_sharpe": report_t1.sharpe_model,
        "tier2_sharpe": report_t2.sharpe_model,
        "cost_erosion_t0_t1": report_t0.sharpe_model - report_t1.sharpe_model,
        "cost_erosion_t1_t2": report_t1.sharpe_model - report_t2.sharpe_model,
    }

    return results


# ---------------------------------------------------------------------------
# R2: Tier2 cost decomposition
# ---------------------------------------------------------------------------

def compute_tier2_cost_decomposition(
    dataset: pd.DataFrame,
    preds: pd.Series,
    *,
    tier2_cfg: "Tier2ImpactConfig | None" = None,
    base_config: Optional[Dict[str, Any] | TradeProxyConfig] = None,
    adv_by_root: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Break out commission, slippage, and impact costs per tier.

    Returns a dict keyed by tier name, each containing:
    - commission_bps_mean, slippage_bps_mean, impact_bps_mean, total_cost_bps_mean
    - contracts_adv_distribution (root → mean contracts/ADV ratio)
    - downscale_trigger_rate (fraction of root-days hitting impact cap)
    """
    from sleeves.cooc_reversal_futures.pipeline.types import Tier2ImpactConfig

    if tier2_cfg is None:
        tier2_cfg = Tier2ImpactConfig()

    # Resolve per-root ADV
    _adv_by_root = adv_by_root or {}

    # Get base config values
    if isinstance(base_config, TradeProxyConfig):
        bc = base_config
    elif base_config is not None:
        bc = TradeProxyConfig.from_dict(base_config)
    else:
        bc = TradeProxyConfig()

    commission_bps = bc.cost_per_contract * 1e4 / bc.proxy_equity_usd if bc.proxy_equity_usd > 0 else 0.0

    # Tier 0
    t0 = {
        "commission_bps_mean": 0.0,
        "slippage_bps_mean": 0.0,
        "impact_bps_mean": 0.0,
        "total_cost_bps_mean": 0.0,
    }

    # Tier 1
    t1_slip = bc.slippage_bps_open + bc.slippage_bps_close
    t1 = {
        "commission_bps_mean": commission_bps,
        "slippage_bps_mean": t1_slip,
        "impact_bps_mean": 0.0,
        "total_cost_bps_mean": commission_bps + t1_slip,
    }

    # Tier 2: per-root impact
    impact_by_root: Dict[str, float] = {}
    contracts_adv_ratio: Dict[str, float] = {}
    cap_hits = 0
    total_root_days = 0

    if "root" in dataset.columns:
        for root in dataset["root"].unique():
            root_adv = _adv_by_root.get(root, tier2_cfg.min_adv_contracts)
            # Approximate 1 contract per root per day as base trade size
            impact = compute_tier2_impact_bps(
                n_contracts=1.0, adv=root_adv, impact_cfg=tier2_cfg,
            )
            impact_by_root[root] = impact
            contracts_adv_ratio[root] = 1.0 / max(root_adv, 1e-9)
            root_day_count = int((dataset["root"] == root).sum())

            # Check cap hits
            raw_participation = 1.0 / max(root_adv, tier2_cfg.min_adv_contracts)
            raw_impact = tier2_cfg.base_bps + tier2_cfg.k * (raw_participation ** tier2_cfg.p)
            if raw_impact > tier2_cfg.impact_cap_bps:
                cap_hits += root_day_count
            total_root_days += root_day_count

    avg_impact = float(np.mean(list(impact_by_root.values()))) if impact_by_root else 0.0
    downscale_rate = cap_hits / max(total_root_days, 1)

    t2 = {
        "commission_bps_mean": commission_bps,
        "slippage_bps_mean": t1_slip,
        "impact_bps_mean": avg_impact,
        "total_cost_bps_mean": commission_bps + t1_slip + avg_impact,
        "contracts_adv_distribution": contracts_adv_ratio,
        "downscale_trigger_rate": downscale_rate,
    }

    return {
        "TIER0_ZERO_COST": t0,
        "TIER1_SIMPLE_COST": t1,
        "TIER2_SPREAD_IMPACT": t2,
    }


# ---------------------------------------------------------------------------
# R2: Tier2 sensitivity sweep
# ---------------------------------------------------------------------------

def evaluate_tier2_sensitivity(
    dataset: pd.DataFrame,
    preds: pd.Series,
    *,
    slippage_steps: tuple[float, ...] = (1.0, 2.0, 5.0, 10.0),
    impact_k_steps: tuple[float, ...] = (0.05, 0.1, 0.2, 0.5),
    tier2_cfg: "Tier2ImpactConfig | None" = None,
    base_config: Optional[Dict[str, Any] | TradeProxyConfig] = None,
    adv_by_root: Optional[Dict[str, float]] = None,
) -> Dict[str, Dict[str, float]]:
    """Run Tier2 evaluation across stepped cost parameters.

    Returns dict keyed by step label → {sharpe, hit_rate, max_drawdown}.
    """
    from sleeves.cooc_reversal_futures.pipeline.types import Tier2ImpactConfig

    if tier2_cfg is None:
        tier2_cfg = Tier2ImpactConfig()

    results: Dict[str, Dict[str, float]] = {}

    # Sweep slippage
    for slip in slippage_steps:
        label = f"slip_{slip:.1f}bps"
        if isinstance(base_config, TradeProxyConfig):
            cfg_d = base_config.to_dict()
        elif base_config is not None:
            cfg_d = dict(base_config)
        else:
            cfg_d = {}
        cfg_d["slippage_bps_open"] = slip
        cfg_d["slippage_bps_close"] = slip
        report = evaluate_trade_proxy(dataset, preds, config=cfg_d)
        results[label] = {
            "sharpe": report.sharpe_model,
            "hit_rate": report.hit_rate,
            "max_drawdown": report.max_drawdown,
        }

    # Sweep impact k
    for k in impact_k_steps:
        label = f"impact_k_{k:.2f}"
        stepped_cfg = Tier2ImpactConfig(
            base_bps=tier2_cfg.base_bps,
            k=k,
            p=tier2_cfg.p,
            adv_window=tier2_cfg.adv_window,
            min_adv_contracts=tier2_cfg.min_adv_contracts,
            impact_cap_bps=tier2_cfg.impact_cap_bps,
            downscale_on_cap=tier2_cfg.downscale_on_cap,
        )
        ladder = evaluate_realism_ladder(
            dataset, preds,
            tier2_cfg=stepped_cfg,
            base_config=base_config,
            adv_by_root=adv_by_root,
        )
        t2 = ladder["TIER2_SPREAD_IMPACT"]
        results[label] = {
            "sharpe": t2["sharpe"],
            "hit_rate": t2["hit_rate"],
            "max_drawdown": t2["max_drawdown"],
        }

    return results


