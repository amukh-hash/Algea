"""R5: Baseline-shrinkage alpha blending.

Blends model alpha with baseline alpha *before* proxy execution.
The proxy itself is unchanged — this module produces the blended
alpha series that gets fed in.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AlphaBlendConfig:
    """Baseline-shrinkage blending parameters."""
    enabled: bool = False
    lambda_mode: Literal["constant", "confidence_linear"] = "constant"
    lambda_const: float = 0.5
    lambda_min: float = 0.0
    lambda_max: float = 1.0
    conf_low: float = 0.0
    conf_high: float = 1.0


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def compute_lambda(conf: float, cfg: AlphaBlendConfig) -> float:
    """Map a scalar confidence to a blending weight λ.

    λ = 1 means "use model alpha only".
    λ = 0 means "use baseline alpha only".

    Parameters
    ----------
    conf
        Day-level confidence (e.g. from exposure_policy.compute_day_confidence).
    cfg
        Blend config.

    Returns
    -------
    float in [lambda_min, lambda_max].
    """
    if cfg.lambda_mode == "constant":
        return float(np.clip(cfg.lambda_const, cfg.lambda_min, cfg.lambda_max))

    # confidence_linear: linearly map [conf_low, conf_high] → [lambda_min, lambda_max]
    span = cfg.conf_high - cfg.conf_low
    if span <= 0:
        return cfg.lambda_min

    t = (conf - cfg.conf_low) / span
    t = float(np.clip(t, 0.0, 1.0))
    lam = cfg.lambda_min + t * (cfg.lambda_max - cfg.lambda_min)
    return float(lam)


def compute_lambda_series(
    conf_series: pd.Series,
    cfg: AlphaBlendConfig,
) -> pd.Series:
    """Vectorised :func:`compute_lambda` over a per-day confidence series.

    Parameters
    ----------
    conf_series
        Index = trading_day, values = confidence.
    cfg
        Blend config.

    Returns
    -------
    pd.Series
        Index = trading_day, values = λ in [lambda_min, lambda_max].
    """
    return conf_series.map(lambda c: compute_lambda(c, cfg)).rename("lambda")


def blend_alpha(
    alpha_model: pd.Series,
    alpha_baseline: pd.Series,
    lam: float | pd.Series,
) -> pd.Series:
    """Blend model and baseline alpha.

    ``alpha_final = λ * alpha_model + (1 - λ) * alpha_baseline``

    Parameters
    ----------
    alpha_model, alpha_baseline
        Same-indexed alpha series.
    lam
        Scalar or per-row lambda.  λ=1 → pure model, λ=0 → pure baseline.

    Returns
    -------
    pd.Series
        Blended alpha, same index as inputs.
    """
    return (lam * alpha_model + (1 - lam) * alpha_baseline).rename("alpha_blended")
