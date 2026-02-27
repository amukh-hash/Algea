"""
Baseline scorer and blend/gate strategy for the selector pipeline.

Provides:
- ``compute_baseline_score``: priors-only linear combiner (no model)
- ``sigmoid_gate``: soft gate weight driven by regime risk
- ``piecewise_gate``: clamped linear gate weight
- ``blend_scores``: combines model + baseline via gating

All functions operate on DataFrames or Series — no PyTorch dependency.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Baseline scorer (priors-only)
# ═══════════════════════════════════════════════════════════════════════════

def compute_baseline_score(
    df: pd.DataFrame,
    a: float = 1.0,
    lam: float = 0.5,
    mu: float = 0.5,
    drift_col: str = "z_drift_10",
    regime_col: str = "z_regime_risk",
    tail_col: str = "z_tail_risk_30",
) -> pd.Series:
    """Deterministic priors-only baseline score.

    ``score_baseline = a * z_drift_10 - lam * z_regime_risk - mu * z_tail_risk_30``

    Parameters
    ----------
    a : float
        Weight for drift signal (default 1.0).
    lam : float
        Penalty weight for regime risk / z_iqr_30 (default 0.5).
    mu : float
        Penalty weight for tail risk (default 0.5).

    Falls back to ``z_iqr_30`` if ``z_regime_risk`` is absent.
    """
    # Drift
    d = df[drift_col].fillna(0).values.astype(float) if drift_col in df.columns else 0

    # Regime risk
    if regime_col in df.columns:
        r = df[regime_col].fillna(0).values.astype(float)
    elif "z_iqr_30" in df.columns:
        r = df["z_iqr_30"].fillna(0).values.astype(float)
    else:
        r = 0

    # Tail risk
    if tail_col in df.columns:
        t = df[tail_col].fillna(0).values.astype(float)
    else:
        t = 0

    vals = a * d - lam * r - mu * t
    return pd.Series(vals, index=df.index, name="score_baseline")


# ═══════════════════════════════════════════════════════════════════════════
# Gate functions
# ═══════════════════════════════════════════════════════════════════════════

def sigmoid_gate(
    regime_risk: np.ndarray,
    g0: float = 0.0,
    g1: float = 1.0,
) -> np.ndarray:
    """Sigmoid soft gate: w = σ(g0 - g1 * z_regime_risk).

    When regime risk is high (positive z-score), w → 0 → more baseline.
    When regime risk is low (negative z-score), w → 1 → more model.
    """
    logits = g0 - g1 * regime_risk
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -20, 20)))


def piecewise_gate(
    regime_risk: np.ndarray,
    threshold: float = 0.0,
    k: float = 1.0,
) -> np.ndarray:
    """Piecewise linear gate: w = clamp(1 - k * max(0, z_regime_risk - t), 0, 1)."""
    return np.clip(1.0 - k * np.maximum(0, regime_risk - threshold), 0.0, 1.0)


# ═══════════════════════════════════════════════════════════════════════════
# Blend strategy
# ═══════════════════════════════════════════════════════════════════════════

def sanity_check_gate_monotonicity(
    w: np.ndarray,
    gate_input: np.ndarray,
    g1: float,
    label: str = "",
) -> bool:
    """Check that w is monotonically decreasing in gate_input when g1>0.

    Returns True if the check passes.  Logs a WARNING if violated.
    Intended for post-training diagnostics — do NOT call in hot inference.
    """
    if g1 <= 0 or len(w) < 10:
        return True
    rho = np.corrcoef(w, gate_input)[0, 1]
    if np.isnan(rho):
        return True  # degenerate input, skip
    if rho >= 0:
        logger.warning(
            f"GATE MONOTONICITY VIOLATION{f' ({label})' if label else ''}: "
            f"corr(w, gate_input)={rho:+.4f} should be <0 when g1={g1}. "
            f"Check gate_cs_sign."
        )
        return False
    return True


def compute_gate_input(
    df: pd.DataFrame,
    regime_col: str = "z_regime_risk",
    gate_gamma: float = 0.0,
    uncertainty_col: str = "z_regime_uncertainty",
    gate_gamma_cs: float = 1.0,
    cs_col: str = "z_cs_tail_30_std",
    gate_cs_sign: float = -1.0,
) -> np.ndarray:
    """Public API to compute raw gate_input (for diagnostics).

    Same as ``_get_gate_input`` but exposed for post-training analysis.
    """
    return _get_gate_input(
        df, regime_col, gate_gamma, uncertainty_col,
        gate_gamma_cs, cs_col, gate_cs_sign,
    )


def _get_gate_input(
    df: pd.DataFrame,
    regime_col: str = "z_regime_risk",
    gate_gamma: float = 0.0,
    uncertainty_col: str = "z_regime_uncertainty",
    gate_gamma_cs: float = 1.0,
    cs_col: str = "z_cs_tail_30_std",
    gate_cs_sign: float = -1.0,
) -> np.ndarray:
    """Build gate input: z_regime_risk + gamma_unc * uncertainty + gamma_cs * sign * cs.

    The first term (z_regime_risk) is cross-sectional (per-ticker within a date).
    The third term is date-level (same for all tickers on a date).

    ``gate_cs_sign`` controls the direction of the date-level signal.
    With the default ``-1.0``, negative z_cs values (stress months) become
    positive contributions to gate_input, which drives w lower (more baseline).
    """
    if regime_col in df.columns:
        risk = df[regime_col].fillna(0).values.astype(float)
    elif "z_iqr_30" in df.columns:
        risk = df["z_iqr_30"].fillna(0).values.astype(float)
    else:
        risk = np.zeros(len(df))

    if gate_gamma > 0 and uncertainty_col in df.columns:
        uncert = df[uncertainty_col].fillna(0).values.astype(float)
        risk = risk + gate_gamma * uncert

    if gate_gamma_cs != 0 and cs_col in df.columns:
        cs = df[cs_col].fillna(0).values.astype(float)
        risk = risk + gate_gamma_cs * gate_cs_sign * cs

    return risk


def compute_gate_weights(
    df: pd.DataFrame,
    blend_mode: str = "sigmoid",
    regime_col: str = "z_regime_risk",
    g0: float = 0.0,
    g1: float = 1.0,
    gate_threshold: float = 0.0,
    gate_k: float = 1.0,
    gate_gamma: float = 0.0,
    uncertainty_col: str = "z_regime_uncertainty",
    gate_gamma_cs: float = 1.0,
    cs_col: str = "z_cs_tail_30_std",
    gate_cs_sign: float = -1.0,
) -> np.ndarray:
    """Compute per-row gate weights (model weight w). Useful for reporting."""
    risk = _get_gate_input(df, regime_col, gate_gamma, uncertainty_col,
                           gate_gamma_cs, cs_col, gate_cs_sign)
    if blend_mode == "sigmoid":
        return sigmoid_gate(risk, g0, g1)
    elif blend_mode == "piecewise":
        return piecewise_gate(risk, gate_threshold, gate_k)
    else:
        raise ValueError(f"Unknown blend_mode: {blend_mode}")


def blend_scores(
    df: pd.DataFrame,
    model_score_col: str = "score_model",
    baseline_score_col: str = "score_baseline",
    blend_mode: str = "sigmoid",
    regime_col: str = "z_regime_risk",
    # Sigmoid params
    g0: float = 0.0,
    g1: float = 1.0,
    # Piecewise params
    gate_threshold: float = 0.0,
    gate_k: float = 1.0,
    # Uncertainty params
    gate_gamma: float = 0.0,
    uncertainty_col: str = "z_regime_uncertainty",
    # Date-level regime params
    gate_gamma_cs: float = 1.0,
    cs_col: str = "z_cs_tail_30_std",
    gate_cs_sign: float = -1.0,
) -> pd.Series:
    """Blend model and baseline scores using two-level regime-aware gating.

    ``score_final = w * score_model + (1 - w) * score_baseline``

    Gate input = ``z_regime_risk + gamma_unc * z_regime_uncertainty
                  + gamma_cs * gate_cs_sign * z_cs_tail_30_std``.

    The first two terms are per-ticker (cross-sectional adaptation).
    The third term is per-date (temporal/regime adaptation).

    Parameters
    ----------
    blend_mode : "sigmoid" | "piecewise"
    g0, g1 : sigmoid gate parameters
    gate_threshold, gate_k : piecewise gate parameters
    gate_gamma : float
        Weight for regime uncertainty in gate input (default 0).
    gate_gamma_cs : float
        Weight for date-level regime stress (default 1.0).
    gate_cs_sign : float
        Sign multiplier for date-level signal (default -1.0).
    cs_col : str
        Column for date-level regime signal.

    Returns
    -------
    pd.Series named "score_final"
    """
    model_s = df[model_score_col].values.astype(float)
    baseline_s = df[baseline_score_col].values.astype(float)

    w = compute_gate_weights(
        df, blend_mode, regime_col, g0, g1,
        gate_threshold, gate_k, gate_gamma, uncertainty_col,
        gate_gamma_cs, cs_col, gate_cs_sign,
    )

    final = w * model_s + (1.0 - w) * baseline_s
    return pd.Series(final, index=df.index, name="score_final")


# ═══════════════════════════════════════════════════════════════════════════
# Gate parameter tuning (lightweight grid search on validation)
# ═══════════════════════════════════════════════════════════════════════════

def tune_gate_params(
    df: pd.DataFrame,
    target_col: str = "y_ret",
    model_score_col: str = "score_model",
    baseline_score_col: str = "score_baseline",
    regime_col: str = "z_regime_risk",
    g0_grid: Optional[list] = None,
    g1_grid: Optional[list] = None,
    gate_gamma: float = 0.0,
    gamma_cs_grid: Optional[list] = None,
    cs_sign_grid: Optional[list] = None,
) -> dict:
    """Grid search for best sigmoid gate params on validation data.

    Searches over g0 × g1 × gamma_cs × cs_sign combinations and picks
    the set maximizing mean Spearman IC across dates.

    Returns dict with keys: best_g0, best_g1, best_gamma_cs, best_cs_sign,
    best_ic, grid_results.
    """
    from algea.eval.selector_metrics import per_date_metrics

    if g0_grid is None:
        g0_grid = [-1.0, 0.0, 1.0]
    if g1_grid is None:
        g1_grid = [0.5, 1.0, 2.0]
    if gamma_cs_grid is None:
        gamma_cs_grid = [0.0, 0.5, 1.0, 2.0]
    if cs_sign_grid is None:
        cs_sign_grid = [-1.0, 1.0]

    best_ic = -float("inf")
    best_g0, best_g1, best_gamma_cs, best_cs_sign = 0.0, 1.0, 1.0, -1.0
    grid_results = []

    for g0 in g0_grid:
        for g1 in g1_grid:
            for gcs in gamma_cs_grid:
                for csign in cs_sign_grid:
                    df["_trial_blend"] = blend_scores(
                        df, model_score_col, baseline_score_col,
                        "sigmoid", regime_col, g0=g0, g1=g1,
                        gate_gamma=gate_gamma,
                        gate_gamma_cs=gcs,
                        gate_cs_sign=csign,
                    )
                    dm = per_date_metrics(df, "_trial_blend", target_col)
                    ic = dm["ic"].mean()

                    grid_results.append({
                        "g0": g0, "g1": g1, "gamma_cs": gcs,
                        "cs_sign": csign, "ic": round(ic, 6),
                    })

                    if ic > best_ic:
                        best_ic = ic
                        best_g0 = g0
                        best_g1 = g1
                        best_gamma_cs = gcs
                        best_cs_sign = csign

    df.drop(columns=["_trial_blend"], inplace=True, errors="ignore")

    return {
        "best_g0": best_g0,
        "best_g1": best_g1,
        "best_gamma_cs": best_gamma_cs,
        "best_cs_sign": best_cs_sign,
        "best_ic": round(best_ic, 6),
        "grid_results": grid_results,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Dead feature detection utility
# ═══════════════════════════════════════════════════════════════════════════

def detect_dead_features(
    df: pd.DataFrame,
    feature_cols: list,
    std_threshold: float = 1e-6,
) -> list:
    """Identify features with zero (or near-zero) global standard deviation.

    Returns list of column names that are effectively constant.
    """
    dead = []
    for col in feature_cols:
        if col not in df.columns:
            continue
        if df[col].std() < std_threshold:
            dead.append(col)
    return dead
