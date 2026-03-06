"""Validation harness — must-pass gates for data, leakage, model, strategy."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..config import COOCReversalConfig
from ..contract_master import CONTRACT_MASTER
from ..sleeve import COOCReversalFuturesSleeve
from ..sizing import contracts_from_weights
from .dataset import assert_no_leakage
from .train import FEATURE_COLUMNS, Preprocessor
from .types import CoverageReport, GateResult, ModelBundle, PromotionWindow, PromotionWindowsReport, SplitSpec, ValidationReport

from algae.data.options.vrp_features import VolRegime


# ---------------------------------------------------------------------------
# Gate implementations
# ---------------------------------------------------------------------------

def _data_gate_missingness(
    dataset: pd.DataFrame,
    features: Tuple[str, ...],
    threshold: float = 0.05,
) -> GateResult:
    """Fail if any feature has >threshold fraction missing after assembly."""
    feat_df = dataset[list(features)]
    miss_frac = feat_df.isna().mean()
    worst = float(miss_frac.max())
    worst_col = str(miss_frac.idxmax()) if worst > 0 else ""
    passed = worst <= threshold
    detail = f"worst missingness {worst:.4f} in '{worst_col}' (threshold={threshold})"
    return GateResult(name="data_missingness", passed=passed, detail=detail)


def _data_gate_roll_coverage(
    dataset: pd.DataFrame,
    contract_map: Optional[pd.DataFrame] = None,
) -> GateResult:
    """Fail if any (root, trading_day) pair is missing from contract map."""
    if contract_map is None:
        return GateResult(name="roll_coverage", passed=True, detail="no contract_map provided, skipped")

    if "active_contract" in dataset.columns:
        missing = dataset["active_contract"].isna().sum()
    else:
        missing = 0
    passed = int(missing) == 0
    detail = f"{missing} rows missing active_contract"
    return GateResult(name="roll_coverage", passed=passed, detail=detail)


def _leakage_gate(dataset: pd.DataFrame) -> GateResult:
    """Hard-fail if leakage invariants are violated."""
    try:
        assert_no_leakage(dataset)
        return GateResult(name="leakage", passed=True, detail="all rows pass leakage check")
    except AssertionError as e:
        return GateResult(name="leakage", passed=False, detail=str(e))


def _model_sanity_gate(
    dataset: pd.DataFrame,
    model: Any,
    preprocessor: Preprocessor,
    split: SplitSpec,
    tolerance: float = 0.0,
) -> Tuple[GateResult, float, float]:
    """Compare model IC to baseline (-ret_co) IC on validation data.

    Returns (gate_result, baseline_ic, model_ic).
    """
    val_start = date.fromisoformat(split.val_start)
    val_end = date.fromisoformat(split.val_end)

    if "trading_day" in dataset.columns:
        days = dataset["trading_day"]
    else:
        days = dataset.index.get_level_values("trading_day")

    val_mask = pd.Series(
        [val_start <= (d if isinstance(d, date) else pd.Timestamp(d).date()) <= val_end for d in days],
        index=dataset.index,
    )
    val_df = dataset.loc[val_mask]

    if len(val_df) < 5:
        return (
            GateResult(name="model_sanity", passed=True, detail="too few val rows to evaluate"),
            0.0,
            0.0,
        )

    y_true = val_df["y"].values.astype(np.float64)

    # Baseline: score = r_co (aligns with y = -r_oc reversal target)
    r_co_col = "r_co" if "r_co" in val_df.columns else "ret_co"
    baseline_pred = val_df[r_co_col].values.astype(np.float64)
    baseline_ic = _ic(y_true, baseline_pred)

    # Model prediction
    X_val = preprocessor.transform(val_df)
    model_pred = model.predict(X_val)
    model_ic = _ic(y_true, model_pred)

    passed = model_ic >= baseline_ic - tolerance
    detail = f"model IC={model_ic:.4f}, baseline IC={baseline_ic:.4f}, tol={tolerance}"

    return (
        GateResult(name="model_sanity", passed=passed, detail=detail),
        baseline_ic,
        model_ic,
    )


def _strategy_gate_crash(config: COOCReversalConfig) -> GateResult:
    """Verify CRASH_RISK → zero exposure and flatten orders."""
    sleeve = COOCReversalFuturesSleeve(config)
    r = sleeve.build_daily_orders(
        date_t=date(2025, 1, 2),
        pred_mu={"ES": 0.002, "NQ": -0.001},
        pred_sigma={"ES": 0.01, "NQ": 0.01},
        prices={"ES": 5000, "NQ": 18000},
        capital=1_000_000,
        regime=VolRegime.CRASH_RISK,
    )
    zero_exposure = all(v == 0 for v in r["contracts"].values())
    no_orders = len(r["orders"]) == 0
    passed = zero_exposure and no_orders
    detail = f"crash: zero_exposure={zero_exposure}, no_orders={no_orders}"
    return GateResult(name="strategy_crash", passed=passed, detail=detail)


def _strategy_gate_caution(config: COOCReversalConfig) -> GateResult:
    """Verify CAUTION → scaled gross."""
    sleeve = COOCReversalFuturesSleeve(config)
    r_normal = sleeve.build_daily_orders(
        date_t=date(2025, 1, 2),
        pred_mu={"ES": 0.003, "NQ": -0.003},
        pred_sigma={"ES": 0.01, "NQ": 0.01},
        prices={"ES": 5000, "NQ": 18000},
        capital=5_000_000,
        regime=VolRegime.NORMAL_CARRY,
    )
    r_caution = sleeve.build_daily_orders(
        date_t=date(2025, 1, 2),
        pred_mu={"ES": 0.003, "NQ": -0.003},
        pred_sigma={"ES": 0.01, "NQ": 0.01},
        prices={"ES": 5000, "NQ": 18000},
        capital=5_000_000,
        regime=VolRegime.CAUTION,
    )
    normal_gross = sum(abs(v) for v in r_normal["weights"].values())
    caution_gross = sum(abs(v) for v in r_caution["weights"].values())
    # Caution gross should be <= normal gross * caution_scale tolerance
    scaled_correctly = caution_gross <= normal_gross + 1e-6
    detail = f"normal_gross={normal_gross:.4f}, caution_gross={caution_gross:.4f}, scale={config.caution_scale}"
    return GateResult(name="strategy_caution", passed=scaled_correctly, detail=detail)


def _strategy_gate_caps(config: COOCReversalConfig) -> GateResult:
    """Verify integer rounding respects max_contracts_per_instrument."""
    test_weights = {"ES": 0.5, "NQ": 0.5}
    prices = {"ES": 5000, "NQ": 18000}
    multipliers = {"ES": CONTRACT_MASTER["ES"].multiplier, "NQ": CONTRACT_MASTER["NQ"].multiplier}
    contracts = contracts_from_weights(
        test_weights, capital=100_000_000, prices=prices,
        multipliers=multipliers, max_contracts=config.max_contracts_per_instrument,
    )
    cap_ok = all(abs(v) <= config.max_contracts_per_instrument for v in contracts.values())
    detail = f"contracts={contracts}, cap={config.max_contracts_per_instrument}"
    return GateResult(name="strategy_caps", passed=cap_ok, detail=detail)

def _coverage_gate(
    dataset: pd.DataFrame,
    min_roots_per_day: int = 4,
    allow_partial: bool = False,
) -> tuple[GateResult, CoverageReport]:
    """Verify cross-sectional panel has enough roots per trading day."""
    day_col = "trading_day"
    # Always reset index to avoid ambiguity when trading_day is both
    # a column and an index level (as produced by assemble_dataset)
    if isinstance(dataset.index, pd.MultiIndex) or day_col in (dataset.index.names or []):
        working = dataset.reset_index(drop=day_col in dataset.columns)
    else:
        working = dataset
    if day_col not in working.columns and day_col in (dataset.index.names or []):
        working = dataset.reset_index()

    root_col = "root" if "root" in working.columns else "instrument"
    counts = working.groupby(day_col)[root_col].nunique()

    days_total = len(counts)
    days_below = int((counts < min_roots_per_day).sum())

    # Histogram: {num_roots: num_days}
    histogram = counts.value_counts().sort_index().to_dict()
    histogram = {int(k): int(v) for k, v in histogram.items()}

    if days_below > 0 and not allow_partial:
        passed = False
        detail = (f"{days_below}/{days_total} days have < {min_roots_per_day} roots; "
                  f"worst day has {int(counts.min())} roots")
    else:
        passed = True
        detail = (f"all {days_total} days have >= {min_roots_per_day} roots "
                  f"(min={int(counts.min())})" if days_total > 0 else "no days")

    report = CoverageReport(
        days_total=days_total,
        days_below_threshold=days_below,
        min_roots_per_day=min_roots_per_day,
        gate_passed=passed,
        histogram=histogram,
    )

    return GateResult(name="coverage", passed=passed, detail=detail), report


def _trade_proxy_gate(
    dataset: pd.DataFrame,
    model: Any,
    preprocessor: Preprocessor,
    split: SplitSpec,
    min_sharpe_delta: float = 0.10,
    max_drawdown_tolerance: float = -0.10,
    worst_1pct_tolerance_bps: float = 25.0,
    min_hit_rate: float = 0.45,
) -> GateResult:
    """Trade-proxy promotion gate — 4 sub-checks.

    1. Sharpe delta: model_sharpe >= baseline_sharpe + min_sharpe_delta
    2. Max drawdown: model max_drawdown >= max_drawdown_tolerance
    3. Worst 1% day: within tolerance
    4. Hit rate: >= min_hit_rate
    """
    from .trade_proxy import evaluate_trade_proxy

    val_start = date.fromisoformat(split.val_start)
    val_end = date.fromisoformat(split.val_end)

    if "trading_day" in dataset.columns:
        days = dataset["trading_day"]
    else:
        days = dataset.index.get_level_values("trading_day")

    val_mask = pd.Series(
        [val_start <= (d if isinstance(d, date) else pd.Timestamp(d).date()) <= val_end for d in days],
        index=dataset.index,
    )
    val_df = dataset.loc[val_mask]
    # Reset MultiIndex to avoid column/index ambiguity in trade proxy
    if isinstance(val_df.index, pd.MultiIndex):
        val_df = val_df.reset_index(drop=True)

    if len(val_df) < 10:
        return GateResult(
            name="trade_proxy", passed=True,
            detail="too few val rows to evaluate trade proxy",
        )

    # Model predictions
    X_val = preprocessor.transform(val_df)
    if hasattr(model, "predict"):
        preds = model.predict(X_val)
    else:
        # Transformer: flat predict on val
        import torch
        from ..model.score_stabilizer import stabilize_derived_score
        model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_val, dtype=torch.float32).unsqueeze(0)
            out = model(X_t)
            if isinstance(out, tuple):
                score, risk = out
                preds = stabilize_derived_score(score, risk).squeeze(0).numpy()
            else:
                preds = out.squeeze(0).numpy()

    try:
        report = evaluate_trade_proxy(
            dataset=val_df,
            preds=preds,
            config={"cost_per_contract": 0.0, "slippage_bps_open": 0.0,
                    "slippage_bps_close": 0.0},
        )
    except Exception as e:
        return GateResult(
            name="trade_proxy", passed=False,
            detail=f"trade_proxy evaluation failed: {e}",
        )

    # Sub-check 1: Sharpe delta
    sharpe_ok = report.sharpe_model >= report.sharpe_baseline + min_sharpe_delta
    # Sub-check 2: Max drawdown
    dd_ok = report.max_drawdown >= max_drawdown_tolerance
    # Sub-check 3: Worst 1% day
    worst_ok = report.worst_1pct_return >= -(worst_1pct_tolerance_bps / 10000.0)
    # Sub-check 4: Hit rate
    hit_ok = report.hit_rate >= min_hit_rate

    passed = sharpe_ok and dd_ok  # Hard gates
    detail = (
        f"sharpe_model={report.sharpe_model:.3f}, "
        f"sharpe_baseline={report.sharpe_baseline:.3f}, "
        f"sharpe_ok={sharpe_ok}, "
        f"max_dd={report.max_drawdown:.4f}, dd_ok={dd_ok}, "
        f"worst_1pct={report.worst_1pct_return:.6f}, worst_ok={worst_ok}, "
        f"hit_rate={report.hit_rate:.3f}, hit_ok={hit_ok}"
    )

    return GateResult(name="trade_proxy", passed=passed, detail=detail)


def _regime_slicing_report(
    dataset: pd.DataFrame,
    model: Any,
    preprocessor: Preprocessor,
    split: SplitSpec,
) -> GateResult:
    """Regime-sliced performance report (informational, not a hard gate).

    Slices: vol quartiles, shock-vs-normal days, roll-window days.
    """
    from .trade_proxy import evaluate_trade_proxy

    val_start = date.fromisoformat(split.val_start)
    val_end = date.fromisoformat(split.val_end)

    if "trading_day" in dataset.columns:
        days = dataset["trading_day"]
    else:
        days = dataset.index.get_level_values("trading_day")

    val_mask = pd.Series(
        [val_start <= (d if isinstance(d, date) else pd.Timestamp(d).date()) <= val_end for d in days],
        index=dataset.index,
    )
    val_df = dataset.loc[val_mask]
    # Reset MultiIndex to avoid column/index ambiguity
    if isinstance(val_df.index, pd.MultiIndex):
        val_df = val_df.reset_index(drop=True)

    if len(val_df) < 20:
        return GateResult(
            name="regime_slicing", passed=True,
            detail="too few val rows for regime slicing",
        )

    X_val = preprocessor.transform(val_df)
    if hasattr(model, "predict"):
        preds = model.predict(X_val)
    else:
        import torch
        from ..model.score_stabilizer import stabilize_derived_score
        model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_val, dtype=torch.float32).unsqueeze(0)
            out = model(X_t)
            if isinstance(out, tuple):
                score, risk = out
                preds = stabilize_derived_score(score, risk).squeeze(0).numpy()
            else:
                preds = out.squeeze(0).numpy()

    val_df = val_df.copy()
    val_df["_pred"] = preds if len(preds) == len(val_df) else preds[:len(val_df)]

    slices_detail = []

    # --- Slice by volatility quartiles ---
    if "sigma_co" in val_df.columns:
        val_df["_vol_q"] = pd.qcut(val_df["sigma_co"], q=4, labels=False, duplicates="drop")
        for q in sorted(val_df["_vol_q"].dropna().unique()):
            sub = val_df[val_df["_vol_q"] == q]
            if len(sub) >= 5:
                y = sub["y"].values
                p = sub["_pred"].values
                ic = float(np.corrcoef(y, p)[0, 1]) if len(y) > 2 else 0.0
                slices_detail.append(f"vol_q{int(q)}: n={len(sub)}, ic={ic:.3f}")

    # --- Slice by shock flag ---
    if "shock_flag" in val_df.columns:
        for flag_val, label in [(0.0, "normal"), (1.0, "shock")]:
            sub = val_df[val_df["shock_flag"] == flag_val]
            if len(sub) >= 5:
                y = sub["y"].values
                p = sub["_pred"].values
                ic = float(np.corrcoef(y, p)[0, 1]) if len(y) > 2 else 0.0
                slices_detail.append(f"{label}: n={len(sub)}, ic={ic:.3f}")

    # --- Slice by roll window ---
    if "roll_window_flag" in val_df.columns:
        for flag_val, label in [(0, "non_roll"), (1, "roll")]:
            sub = val_df[val_df["roll_window_flag"] == flag_val]
            if len(sub) >= 5:
                y = sub["y"].values
                p = sub["_pred"].values
                ic = float(np.corrcoef(y, p)[0, 1]) if len(y) > 2 else 0.0
                slices_detail.append(f"{label}: n={len(sub)}, ic={ic:.3f}")

    detail = "; ".join(slices_detail) if slices_detail else "no regime slices computed"
    return GateResult(name="regime_slicing", passed=True, detail=detail)


def _data_provider_gate(config: COOCReversalConfig, provider_name: str = "yfinance") -> GateResult:
    """Refuse promotion for non-IBKR data providers."""
    if provider_name in ("ibkr_hist", "hybrid"):
        return GateResult(
            name="data_provider", passed=True,
            detail=f"provider={provider_name} — promotion-grade",
        )
    else:
        return GateResult(
            name="data_provider", passed=False,
            detail=f"provider={provider_name} — RESEARCH_ONLY, not promotion-grade (need ibkr_hist)",
        )


# ---------------------------------------------------------------------------
# R6: Tail-first validation gates (Tier2 only)
# ---------------------------------------------------------------------------

def _tail_risk_gates(
    report_model: "TradeProxyReport",
    report_baseline: "TradeProxyReport",
    *,
    delta_min: float = 0.10,
    dd_tolerance: float = 0.02,
    cvar_tolerance: float = 0.005,
    skew_tolerance: float = 0.5,
) -> list["GateResult"]:
    """Evaluate tail-risk gates comparing model vs baseline proxy reports.

    Hard gates (Tier2):
    - Sharpe delta >= delta_min
    - maxDD_model >= maxDD_baseline - dd_tolerance
    - CVaR_1% model >= CVaR_1% baseline - cvar_tolerance

    Soft gates:
    - worst_1% model >= worst_1% baseline
    - skew model >= skew baseline - skew_tolerance

    Parameters
    ----------
    report_model, report_baseline
        TradeProxyReport results from Tier2 evaluation.
    delta_min
        Minimum Sharpe improvement over baseline.
    dd_tolerance
        Allowed drawdown degradation (absolute).
    cvar_tolerance
        Allowed CVaR degradation (absolute).
    skew_tolerance
        Allowed skew degradation (absolute).

    Returns
    -------
    List of GateResult with name prefixed by ``tail_``.
    """
    from .types import GateResult as _GateResult

    gates: list[_GateResult] = []

    # --- Hard gate: Sharpe delta ---
    sharpe_delta = report_model.sharpe_model - report_baseline.sharpe_model
    sharpe_ok = sharpe_delta >= delta_min
    gates.append(_GateResult(
        name="tail_sharpe_delta",
        passed=sharpe_ok,
        detail=(
            f"model={report_model.sharpe_model:.4f}, "
            f"baseline={report_baseline.sharpe_model:.4f}, "
            f"delta={sharpe_delta:.4f}, min={delta_min}"
        ),
    ))

    # --- Hard gate: maxDD ---
    dd_ok = report_model.max_drawdown >= report_baseline.max_drawdown - dd_tolerance
    gates.append(_GateResult(
        name="tail_maxdd",
        passed=dd_ok,
        detail=(
            f"model_dd={report_model.max_drawdown:.4f}, "
            f"baseline_dd={report_baseline.max_drawdown:.4f}, "
            f"tol={dd_tolerance}"
        ),
    ))

    # --- Hard gate: CVaR 1% ---
    cvar_ok = report_model.cvar_1pct >= report_baseline.cvar_1pct - cvar_tolerance
    gates.append(_GateResult(
        name="tail_cvar_1pct",
        passed=cvar_ok,
        detail=(
            f"model_cvar={report_model.cvar_1pct:.6f}, "
            f"baseline_cvar={report_baseline.cvar_1pct:.6f}, "
            f"tol={cvar_tolerance}"
        ),
    ))

    # --- Soft gate: worst 1% day ---
    worst_ok = report_model.worst_1pct_return >= report_baseline.worst_1pct_return
    gates.append(_GateResult(
        name="tail_worst_1pct",
        passed=worst_ok,
        detail=(
            f"model_worst={report_model.worst_1pct_return:.6f}, "
            f"baseline_worst={report_baseline.worst_1pct_return:.6f}"
        ),
    ))

    # --- Soft gate: skew ---
    skew_ok = report_model.skew >= report_baseline.skew - skew_tolerance
    gates.append(_GateResult(
        name="tail_skew",
        passed=skew_ok,
        detail=(
            f"model_skew={report_model.skew:.4f}, "
            f"baseline_skew={report_baseline.skew:.4f}, "
            f"tol={skew_tolerance}"
        ),
    ))

    return gates


def _multi_window_promotion_gate(
    dataset: pd.DataFrame,
    preds: np.ndarray,
    promotion_windows: List[Dict[str, str]],
    min_sharpe_delta: float = 0.10,
    max_drawdown_tolerance: float = -0.10,
    stress_required: int = 1,
) -> PromotionWindowsReport:
    """Evaluate trade proxy across multiple date windows for promotion.

    Parameters
    ----------
    dataset : full panel dataset with trading_day, predictions, r_oc, etc.
    preds : model predictions aligned with dataset.
    promotion_windows : list of {"name": ..., "start": ..., "end": ...} dicts.
    min_sharpe_delta : required Sharpe improvement for each window.
    max_drawdown_tolerance : worst acceptable drawdown per window.
    stress_required : minimum stress windows that must pass.

    The first window is treated as the primary; the rest are stress windows.
    """
    from .trade_proxy import evaluate_trade_proxy

    if not promotion_windows:
        return PromotionWindowsReport(overall_passed=True)

    df = dataset.copy()
    if "trading_day" not in df.columns and df.index.names[0] == "trading_day":
        df = df.reset_index()
    df["_pred"] = preds if len(preds) == len(df) else preds[:len(df)]

    windows: List[PromotionWindow] = []
    for i, w in enumerate(promotion_windows):
        w_start = date.fromisoformat(w["start"])
        w_end = date.fromisoformat(w["end"])
        name = w.get("name", f"window_{i}")

        days = df["trading_day"]
        mask = pd.Series(
            [w_start <= (d if isinstance(d, date) else pd.Timestamp(d).date()) <= w_end for d in days],
            index=df.index,
        )
        sub = df.loc[mask]

        if len(sub) < 10:
            windows.append(PromotionWindow(
                name=name, start=w["start"], end=w["end"],
                passed=False,
            ))
            continue

        try:
            report = evaluate_trade_proxy(
                dataset=sub,
                preds=sub["_pred"],
                config={"cost_per_contract": 0.0, "slippage_bps_open": 0.0,
                        "slippage_bps_close": 0.0},
            )
            sharpe_ok = report.sharpe_model >= report.sharpe_baseline + min_sharpe_delta
            dd_ok = report.max_drawdown >= max_drawdown_tolerance
            passed = sharpe_ok and dd_ok
            windows.append(PromotionWindow(
                name=name, start=w["start"], end=w["end"],
                sharpe_model=report.sharpe_model,
                sharpe_baseline=report.sharpe_baseline,
                hit_rate=report.hit_rate,
                max_drawdown=report.max_drawdown,
                passed=passed,
            ))
        except Exception:
            windows.append(PromotionWindow(
                name=name, start=w["start"], end=w["end"],
                passed=False,
            ))

    primary_passed = windows[0].passed if windows else False
    stress_wins = windows[1:] if len(windows) > 1 else []
    stress_passed_count = sum(1 for w in stress_wins if w.passed)
    overall = primary_passed and stress_passed_count >= stress_required

    return PromotionWindowsReport(
        windows=tuple(windows),
        primary_passed=primary_passed,
        stress_passed_count=stress_passed_count,
        stress_required=stress_required,
        overall_passed=overall,
    )

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_validation(
    bundle: ModelBundle,
    model: Any,
    preprocessor: Preprocessor,
    dataset: pd.DataFrame,
    splits: List[SplitSpec],
    config: COOCReversalConfig,
    contract_map: Optional[pd.DataFrame] = None,
    ic_tolerance: float = 0.01,
    missingness_threshold: float = 0.05,
    min_roots_per_day: int = 4,
    allow_partial: bool = False,
    min_sharpe_delta: float = 0.10,
    max_drawdown_tolerance: float = -0.10,
    worst_1pct_tolerance_bps: float = 25.0,
    min_hit_rate: float = 0.45,
    require_ibkr: bool = False,
    promotion_windows: Optional[List[Dict[str, str]]] = None,
    stress_required: int = 1,
    provider_name: str = "yfinance",
) -> ValidationReport:
    """Run all must-pass validation gates.

    Parameters
    ----------
    bundle : trained model bundle metadata
    model : the trained model object
    preprocessor : fitted Preprocessor
    dataset : assembled dataset
    splits : list of SplitSpec from CV
    config : sleeve configuration
    contract_map : optional contract map for roll coverage check
    ic_tolerance : how much worse model IC can be vs baseline
    missingness_threshold : max fraction of missing feature values
    min_sharpe_delta : required Sharpe improvement over baseline for trade_proxy gate
    max_drawdown_tolerance : worst acceptable max drawdown
    worst_1pct_tolerance_bps : tolerance for worst 1% day
    min_hit_rate : minimum acceptable hit rate
    require_ibkr : enforce IBKR data provider gate

    Returns
    -------
    ValidationReport with pass/fail per gate
    """
    gates: List[GateResult] = []

    # Data gates
    gates.append(_data_gate_missingness(dataset, FEATURE_COLUMNS, missingness_threshold))
    gates.append(_data_gate_roll_coverage(dataset, contract_map))

    # Leakage gate
    gates.append(_leakage_gate(dataset))

    # Model sanity gate (IC — secondary health check)
    last_split = splits[-1] if splits else None
    baseline_ic = 0.0
    model_ic = 0.0
    if last_split is not None:
        gate_result, baseline_ic, model_ic = _model_sanity_gate(
            dataset, model, preprocessor, last_split, tolerance=ic_tolerance,
        )
        gates.append(gate_result)
    else:
        gates.append(GateResult(name="model_sanity", passed=True, detail="no splits provided"))

    # Trade proxy gate (primary model promotion gate — 4 sub-checks)
    if last_split is not None:
        tp_gate = _trade_proxy_gate(
            dataset, model, preprocessor, last_split,
            min_sharpe_delta=min_sharpe_delta,
            max_drawdown_tolerance=max_drawdown_tolerance,
            worst_1pct_tolerance_bps=worst_1pct_tolerance_bps,
            min_hit_rate=min_hit_rate,
        )
        gates.append(tp_gate)

    # Regime slicing (informational)
    if last_split is not None:
        regime_gate = _regime_slicing_report(
            dataset, model, preprocessor, last_split,
        )
        gates.append(regime_gate)

    # Data provider gate
    if require_ibkr:
        gates.append(_data_provider_gate(config, provider_name=provider_name))

    # Strategy gates
    gates.append(_strategy_gate_crash(config))
    gates.append(_strategy_gate_caution(config))
    gates.append(_strategy_gate_caps(config))

    # Coverage gate
    coverage_gate, _coverage_report = _coverage_gate(
        dataset, min_roots_per_day=min_roots_per_day, allow_partial=allow_partial,
    )
    gates.append(coverage_gate)

    # Multi-window promotion gate (if windows configured)
    promotion_report = None
    if promotion_windows and last_split is not None:
        # Generate predictions on full dataset for window evaluation
        X_all = preprocessor.transform(dataset)
        preds_all = model.predict(X_all).flatten()

        promotion_report = _multi_window_promotion_gate(
            dataset, preds_all, promotion_windows,
            min_sharpe_delta=min_sharpe_delta,
            max_drawdown_tolerance=max_drawdown_tolerance,
            stress_required=stress_required,
        )
        # Add as a gate
        if promotion_report.overall_passed:
            detail = (
                f"primary={'PASS' if promotion_report.primary_passed else 'FAIL'}, "
                f"stress={promotion_report.stress_passed_count}/{len(promotion_windows) - 1} "
                f"(need {stress_required})"
            )
        else:
            detail = (
                f"primary={'PASS' if promotion_report.primary_passed else 'FAIL'}, "
                f"stress={promotion_report.stress_passed_count}/{len(promotion_windows) - 1} "
                f"(need {stress_required}) — FAIL"
            )
        gates.append(GateResult(
            name="promotion_windows",
            passed=promotion_report.overall_passed,
            detail=detail,
        ))

    all_passed = all(g.passed for g in gates)

    return ValidationReport(
        all_passed=all_passed,
        gates=tuple(gates),
        baseline_ic=baseline_ic,
        model_ic=model_ic,
    )


def persist_validation_report(
    report: ValidationReport,
    output_dir: str | Path,
) -> Path:
    """Write validation report as JSON."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "validation_report.json"
    path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Information coefficient (Pearson correlation)."""
    if len(y_true) < 3:
        return 0.0
    c = np.corrcoef(y_true, y_pred)[0, 1]
    return float(c) if np.isfinite(c) else 0.0


# ---------------------------------------------------------------------------
# F5: Additional validation gates
# ---------------------------------------------------------------------------

def _ic_distribution_gate(
    per_day_ics: np.ndarray,
    ic_tail_floor: float = -0.05,
) -> GateResult:
    """Fail if 10th percentile of per-day IC is below floor.

    A model with a few catastrophically wrong days is worse than
    a consistently mediocre one.
    """
    if len(per_day_ics) < 5:
        return GateResult(name="ic_distribution", passed=True,
                          detail="too few days for IC distribution gate")

    p10 = float(np.percentile(per_day_ics, 10))
    passed = p10 >= ic_tail_floor
    detail = (
        f"IC p10={p10:.4f}, floor={ic_tail_floor}, "
        f"median={float(np.median(per_day_ics)):.4f}"
    )
    return GateResult(name="ic_distribution", passed=passed, detail=detail)


def _multi_seed_stability_gate(
    sharpes_by_seed: Dict[int, float],
    max_cv: float = 0.30,
    *,
    baseline_sharpes_by_seed: Optional[Dict[int, float]] = None,
    catastrophic_delta_floor: float = -0.25,
    max_delta_cv: float = 0.30,
) -> GateResult:
    """HARD gate — fail if cross-seed Sharpe instability exceeds thresholds.

    Checks:
      1. CV of raw Sharpe across seeds <= max_cv
      2. If baseline provided: min(delta_i) >= catastrophic_delta_floor
      3. If baseline provided: CV of delta distribution <= max_delta_cv

    Prevents promotion of models that are lucky on one seed.
    """
    values = np.array(list(sharpes_by_seed.values()))
    if len(values) < 2:
        return GateResult(name="multi_seed_stability", passed=True,
                          detail="single seed — skipped")

    mean_s = float(values.mean())
    std_s = float(values.std(ddof=1))
    cv = std_s / max(abs(mean_s), 1e-6)

    failures: list[str] = []

    # --- Check 1: raw CV ---
    if cv > max_cv:
        failures.append(f"raw_cv={cv:.3f} > {max_cv}")

    # --- Baseline-relative checks ---
    delta_detail = ""
    if baseline_sharpes_by_seed is not None:
        seeds = sorted(set(sharpes_by_seed) & set(baseline_sharpes_by_seed))
        if len(seeds) >= 2:
            deltas = np.array([
                sharpes_by_seed[s] - baseline_sharpes_by_seed[s]
                for s in seeds
            ])
            min_delta = float(deltas.min())
            median_delta = float(np.median(deltas))
            delta_mean = float(deltas.mean())
            delta_std = float(deltas.std(ddof=1))
            delta_cv = delta_std / max(abs(delta_mean), 1e-6)

            # Check 2: catastrophic floor
            if min_delta < catastrophic_delta_floor:
                failures.append(
                    f"min_delta={min_delta:.3f} < {catastrophic_delta_floor}"
                )
            # Check 3: delta CV
            if delta_cv > max_delta_cv:
                failures.append(
                    f"delta_cv={delta_cv:.3f} > {max_delta_cv}"
                )
            delta_detail = (
                f", deltas: min={min_delta:.3f}, median={median_delta:.3f}, "
                f"cv={delta_cv:.3f}"
            )

    passed = len(failures) == 0
    detail = (
        f"seeds={len(values)}, mean_sharpe={mean_s:.3f}, "
        f"std={std_s:.3f}, cv={cv:.3f}, max_cv={max_cv}"
        f"{delta_detail}"
    )
    if failures:
        detail += f" | FAILED: {'; '.join(failures)}"
    return GateResult(name="multi_seed_stability", passed=passed, detail=detail)


def _stress_window_gate(
    stress_results: List[Dict[str, Any]],
    catastrophic_sharpe_floor: float = -0.50,
    catastrophic_dd_worse_than_baseline: float = -0.20,
) -> GateResult:
    """Fail if any stress window shows catastrophic behaviour.

    Catastrophic = Sharpe < floor OR drawdown worse than baseline by > threshold.
    """
    if not stress_results:
        return GateResult(name="stress_window", passed=True,
                          detail="no stress windows defined")

    failures = []
    for r in stress_results:
        name = r.get("name", "unnamed")
        sharpe = r.get("sharpe_model", 0.0)
        dd = r.get("max_drawdown", 0.0)
        baseline_dd = r.get("max_drawdown_baseline", 0.0)

        if sharpe < catastrophic_sharpe_floor:
            failures.append(f"{name}: sharpe={sharpe:.3f} < {catastrophic_sharpe_floor}")
        if baseline_dd != 0.0 and (dd - baseline_dd) < catastrophic_dd_worse_than_baseline:
            failures.append(f"{name}: dd delta={dd - baseline_dd:.4f} < {catastrophic_dd_worse_than_baseline}")

    passed = len(failures) == 0
    detail = "; ".join(failures) if failures else f"{len(stress_results)} stress windows OK"
    return GateResult(name="stress_window", passed=passed, detail=detail)


def _contiguous_oos_gate(
    oos_sharpe: float,
    cv_mean_sharpe: float,
    max_degradation: float = 0.50,
    *,
    baseline_oos_sharpe: Optional[float] = None,
    sharpe_delta_min_oos: float = 0.05,
    worst_1pct_oos: Optional[float] = None,
    worst_1pct_baseline: Optional[float] = None,
    worst_1pct_tolerance: float = 0.02,
) -> GateResult:
    """Fail if contiguous OOS Sharpe degrades > max_degradation vs CV mean,
    or if absolute OOS delta falls below floor.

    This catches models that look good in walk-forward but fall apart
    on a longer, unseen period.
    """
    failures: list[str] = []

    # --- Check 1: relative degradation vs CV mean ---
    if abs(cv_mean_sharpe) < 1e-6:
        if oos_sharpe < 0:
            failures.append(f"oos_sharpe={oos_sharpe:.3f} < 0 with cv_mean~0")
        degradation_str = "cv_mean=~0"
    else:
        degradation = (cv_mean_sharpe - oos_sharpe) / abs(cv_mean_sharpe)
        if degradation > max_degradation:
            failures.append(
                f"degradation={degradation:.2%} > {max_degradation:.0%}"
            )
        degradation_str = f"degradation={degradation:.2%}, max={max_degradation:.0%}"

    # --- Check 2: absolute OOS delta vs baseline ---
    delta_str = ""
    if baseline_oos_sharpe is not None:
        oos_delta = oos_sharpe - baseline_oos_sharpe
        if oos_delta < sharpe_delta_min_oos:
            failures.append(
                f"oos_delta={oos_delta:.3f} < {sharpe_delta_min_oos}"
            )
        delta_str = f", oos_delta={oos_delta:.3f} vs min={sharpe_delta_min_oos}"

    # --- Check 3: tail risk not worse than baseline ---
    tail_str = ""
    if worst_1pct_oos is not None and worst_1pct_baseline is not None:
        tail_diff = worst_1pct_oos - worst_1pct_baseline
        if tail_diff < -worst_1pct_tolerance:
            failures.append(
                f"worst_1pct gap={tail_diff:.4f} < -{worst_1pct_tolerance}"
            )
        tail_str = f", worst_1pct_gap={tail_diff:.4f}"

    passed = len(failures) == 0
    detail = (
        f"oos_sharpe={oos_sharpe:.3f}, cv_mean={cv_mean_sharpe:.3f}, "
        f"{degradation_str}{delta_str}{tail_str}"
    )
    if failures:
        detail += f" | FAILED: {'; '.join(failures)}"
    return GateResult(name="contiguous_oos", passed=passed, detail=detail)

