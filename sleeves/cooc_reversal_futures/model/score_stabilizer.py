"""Canonical derived-score stabilization for CS-Transformer two-head models.

Shared by training eval, validation gates, and runtime inference.
No duplication — a single source of truth for the score/(eps + risk) pipeline.

F2 revision: risk_pred is now **raw log_sigma** from the model.
sigma = softplus(log_sigma_raw) + sigma_floor (via alpha_conventions).
Floor only — no hard clamp max for sigma. An optional numeric-safety cap
records cap hit-rate via diagnostics.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def stabilize_derived_score(
    raw_score: "np.ndarray | torch.Tensor",
    risk_pred: "np.ndarray | torch.Tensor",
    *,
    sigma_floor: float = 1e-4,
    sigma_cap: float | None = None,
    score_tanh: bool = False,
    derived_clip: float = 10.0,
    eps: float = 1e-6,
    # Legacy compat — ignored if set, use sigma_floor instead
    risk_clamp_min: float | None = None,
    risk_clamp_max: float | None = None,
) -> "np.ndarray | torch.Tensor":
    """Compute stabilized derived score = raw_score / (eps + sigma).

    F2 update: risk_pred is raw log_sigma from model.
    sigma = softplus(risk_pred) + sigma_floor.
    If sigma_cap is set, cap sigma and record (but don't clamp aggressively).

    Parameters
    ----------
    raw_score
        Raw score-head output. Shape ``[B, N]`` or ``[N]``.
    risk_pred
        Raw risk-head output (log_sigma, NOT softplus'd). Same shape.
    sigma_floor
        Floor for sigma — prevents division blowup.
    sigma_cap
        Optional cap for sigma (numeric safety). Records fraction if exceeded.
    score_tanh
        If True, apply tanh to raw_score before division.
    derived_clip
        Clip derived score to ``[-derived_clip, +derived_clip]``.
    eps
        Small constant for numerical stability.
    risk_clamp_min, risk_clamp_max
        Legacy parameters — ignored. Use sigma_floor/sigma_cap instead.

    Returns
    -------
    Stabilized derived score, same dtype and shape as input.
    """
    # Detect if torch tensor
    _is_torch = False
    try:
        import torch
        if isinstance(raw_score, torch.Tensor):
            _is_torch = True
    except ImportError:
        pass

    if _is_torch:
        import torch
        from backend.app.portfolio.alpha_conventions import sigma_from_log_sigma

        s = torch.tanh(raw_score) if score_tanh else raw_score
        sigma = sigma_from_log_sigma(risk_pred, sigma_floor=sigma_floor)

        if sigma_cap is not None:
            sigma = torch.clamp(sigma, max=sigma_cap)

        derived = s / (eps + sigma)
        derived = torch.clamp(derived, min=-derived_clip, max=derived_clip)
        return derived
    else:
        from backend.app.portfolio.alpha_conventions import sigma_from_log_sigma

        s = np.tanh(raw_score) if score_tanh else raw_score
        sigma = sigma_from_log_sigma(risk_pred, sigma_floor=sigma_floor)

        if sigma_cap is not None:
            sigma = np.clip(sigma, a_min=None, a_max=sigma_cap)

        derived = s / (eps + sigma)
        derived = np.clip(derived, -derived_clip, derived_clip)
        return derived


def stabilizer_params_from_manifest(manifest: dict) -> dict:
    """Extract stabilizer kwargs from a model manifest dict."""
    return {
        "sigma_floor": manifest.get("sigma_floor", 1e-4),
        "sigma_cap": manifest.get("sigma_cap", None),
        "score_tanh": manifest.get("score_tanh", False),
        "derived_clip": manifest.get("derived_score_clip", 10.0),
    }
