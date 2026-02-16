"""Canonical alpha-polarity helpers — single source of truth.

Every component (training loss, trade proxy, validation) MUST use these
functions to convert between score / label / alpha spaces.

Polarity contract
-----------------
r_oc        = open-to-close return (intraday)
y           = -r_oc   (higher y = more bearish)
alpha_target= -y = r_oc  (higher alpha = more attractive to LONG)
derived     = score_raw / (eps + sigma)
alpha_pred  = -derived  (for y=-r_oc convention)
baseline    = -r_co     (mean-revert: gap up -> short)

Any function that touches sign logic MUST live here.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Label & alpha target
# ---------------------------------------------------------------------------

def label_y(df: pd.DataFrame, *, r_oc_col: str = "r_oc") -> pd.Series:
    """Compute label y = -r_oc."""
    return -df[r_oc_col]


def alpha_target_from_y(y: np.ndarray | pd.Series) -> np.ndarray | pd.Series:
    """Convert label y to alpha target = -y = r_oc.

    Higher alpha_target = more attractive for LONG.
    """
    return -y


# ---------------------------------------------------------------------------
# Risk / sigma helpers
# ---------------------------------------------------------------------------

def sigma_from_log_sigma(
    log_sigma_raw: "np.ndarray",
    sigma_floor: float = 1e-4,
) -> "np.ndarray":
    """Convert raw log_sigma to positive sigma = softplus(raw) + floor."""
    try:
        import torch
        if isinstance(log_sigma_raw, torch.Tensor):
            return torch.nn.functional.softplus(log_sigma_raw) + sigma_floor
    except ImportError:
        pass
    # numpy path
    return np.log1p(np.exp(log_sigma_raw)) + sigma_floor


# ---------------------------------------------------------------------------
# Derived score and alpha
# ---------------------------------------------------------------------------

def derived_score(
    score_raw: "np.ndarray",
    sigma: "np.ndarray",
    eps: float = 1e-6,
) -> "np.ndarray":
    """Risk-normalized derived score = score_raw / (eps + sigma)."""
    return score_raw / (eps + sigma)


def derived_to_alpha(
    derived: "np.ndarray",
    convention: Literal["y_neg_roc"] = "y_neg_roc",
) -> "np.ndarray":
    """Convert derived score to alpha for ranking: alpha = -derived.

    Under y=-r_oc convention, derived is positive when score_raw
    is positive (bearish), so alpha = -derived makes higher r_oc
    instruments rank higher for LONG.
    """
    if convention != "y_neg_roc":
        raise ValueError(f"Unknown convention: {convention!r}")
    return -derived


# ---------------------------------------------------------------------------
# Baseline alpha
# ---------------------------------------------------------------------------

def baseline_alpha_from_r_co(
    r_co: np.ndarray | pd.Series,
    mode: Literal["meanrevert", "momentum"] = "meanrevert",
) -> np.ndarray | pd.Series:
    """Compute baseline alpha from r_co.

    meanrevert: alpha = -r_co (gap up -> short, gap down -> long)
    momentum:   alpha =  r_co (follow the gap)
    """
    if mode == "meanrevert":
        return -r_co
    elif mode == "momentum":
        if isinstance(r_co, pd.Series):
            return r_co.copy()
        return r_co.copy() if isinstance(r_co, np.ndarray) else r_co
    raise ValueError(f"Unknown baseline mode: {mode!r}")


# ---------------------------------------------------------------------------
# Score-to-alpha (backward compat with trade_proxy semantics)
# ---------------------------------------------------------------------------

def score_to_alpha(
    score: np.ndarray | pd.Series,
    semantics: Literal["alpha_high_long", "alpha_low_long"] = "alpha_low_long",
) -> np.ndarray | pd.Series:
    """Convert raw score to alpha.

    alpha_high_long: alpha = score  (higher score = long)
    alpha_low_long:  alpha = -score (lower score = long, y=-r_oc default)
    """
    if semantics == "alpha_high_long":
        if isinstance(score, pd.Series):
            return score.copy()
        return score.copy() if isinstance(score, np.ndarray) else score
    elif semantics == "alpha_low_long":
        return -score
    raise ValueError(f"Unknown semantics: {semantics!r}")
