"""R4: Confidence-based exposure gating.

Computes per-day confidence from cross-sectional alpha dispersion and
risk, then produces a gross_schedule that scales exposure before proxy
execution.  The proxy itself stays pure — this module runs *before* it.
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
class ExposurePolicyConfig:
    """Confidence-based exposure gating parameters."""
    crash_threshold: float = 0.05
    caution_threshold: float = 0.15
    caution_scale: float = 0.5
    eps: float = 1e-6
    dispersion_method: Literal["iqr", "std"] = "iqr"


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def compute_day_confidence(
    alpha: pd.Series,
    sigma: pd.Series,
    cfg: ExposurePolicyConfig,
) -> float:
    """Compute intraday confidence from alpha dispersion and risk.

    Parameters
    ----------
    alpha
        Cross-sectional alpha predictions for one trading day.
    sigma
        Cross-sectional risk (sigma) estimates for one trading day.
    cfg
        Exposure policy config.

    Returns
    -------
    float
        Non-negative confidence score.  Higher = more dispersed alpha
        relative to risk → more opportunity → keep full gross.
    """
    valid_alpha = alpha.dropna()
    valid_sigma = sigma.dropna()

    if len(valid_alpha) < 2:
        return 0.0

    # Dispersion of alpha predictions
    if cfg.dispersion_method == "iqr":
        q75, q25 = np.percentile(valid_alpha, [75, 25])
        disp = float(q75 - q25)
    else:  # std
        disp = float(valid_alpha.std())

    # Median risk
    risk = float(valid_sigma.median()) if len(valid_sigma) > 0 else 0.0

    conf = disp / (cfg.eps + risk)
    return float(conf)


def compute_gross_scale(conf: float, cfg: ExposurePolicyConfig) -> float:
    """Map confidence to gross exposure scale.

    Returns
    -------
    float
        0.0 if conf < crash_threshold (CRASH: zero exposure)
        caution_scale if crash <= conf < caution (CAUTION: scaled)
        1.0 otherwise (NORMAL: full gross)
    """
    if conf < cfg.crash_threshold:
        return 0.0
    if conf < cfg.caution_threshold:
        return cfg.caution_scale
    return 1.0


def compute_gross_schedule(
    df: pd.DataFrame,
    alpha_col: str = "model_alpha",
    sigma_col: str = "sigma_pred",
    cfg: Optional[ExposurePolicyConfig] = None,
    day_col: str = "trading_day",
) -> pd.Series:
    """Produce per-day gross_scale_t series from a panel DataFrame.

    Parameters
    ----------
    df
        Panel with ``day_col``, ``alpha_col``, ``sigma_col`` columns.
    alpha_col
        Column containing model alpha predictions.
    sigma_col
        Column containing risk / sigma estimates.
    cfg
        Exposure policy config (uses defaults if None).
    day_col
        Trading day column name.

    Returns
    -------
    pd.Series
        Index = trading days, values = gross scale [0.0, 1.0].
    """
    if cfg is None:
        cfg = ExposurePolicyConfig()

    if sigma_col not in df.columns:
        # No sigma available → default to full gross (no gating)
        days = df[day_col].unique() if day_col in df.columns else df.index.unique()
        return pd.Series(1.0, index=days, name="gross_scale")

    records = []
    grouped = df.groupby(day_col)
    for day, group in grouped:
        alpha = group[alpha_col] if alpha_col in group.columns else pd.Series(dtype=float)
        sigma = group[sigma_col] if sigma_col in group.columns else pd.Series(dtype=float)
        conf = compute_day_confidence(alpha, sigma, cfg)
        scale = compute_gross_scale(conf, cfg)
        records.append({"day": day, "conf": conf, "gross_scale": scale})

    result = pd.DataFrame(records)
    if result.empty:
        return pd.Series(dtype=float, name="gross_scale")
    return result.set_index("day")["gross_scale"]
