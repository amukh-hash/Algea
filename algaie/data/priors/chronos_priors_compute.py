"""
Canonical teacher-priors extraction — single entry point for every consumer.

All priors computation in the selector pipeline MUST go through
``compute_teacher_priors()`` defined here.  This wrapper delegates to
the verified ``infer_priors()`` function in
``algaie.models.foundation.chronos2_teacher`` — it does NOT contain
independent quantile-extraction logic.

Consumers:
    - ``build_priors_cache.py``
    - ``build_priors_frame.py``
    - ``selector_inference.py``  (``score_universe``)
    - ``test_selector_features.py``
"""
from __future__ import annotations

import logging
import math
from typing import Dict, List, Literal, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def compute_teacher_priors(
    teacher: nn.Module,
    input_tensor: torch.Tensor,
    horizon_days: int,
    strict: bool = True,
    mode: Literal["auto", "native_nll", "quantile_head"] = "auto",
) -> "List[ChronosPriors]":
    """Compute distributional priors for a batch of tickers.

    This is the **only** sanctioned priors extraction function for the
    selector pipeline.  It wraps ``infer_priors()`` and performs extra
    validation suitable for downstream selector consumption.

    Parameters
    ----------
    teacher : nn.Module
        A ``Chronos2NativeWrapper`` (NLL-trained teacher) — or any model
        supported by ``infer_priors()``.
    input_tensor : torch.Tensor
        Price context.  Accepted shapes:

        * ``[B, T, F]`` — batch of multivariate (F features each, but
          only ``[:, :, 0]`` is used for the close price).
        * ``[B, T]`` — batch of univariate close series.
        * ``[T]`` or ``[T, F]`` — single series (auto-batched).
    horizon_days : int
        Forecast horizon in **trading days**.
    strict : bool
        If True, raise on validation failures; else clamp/fix and warn.
    mode : {"auto", "native_nll", "quantile_head"}
        Inference mode forwarded to ``infer_priors()``.

    Returns
    -------
    List[ChronosPriors]
        One ``ChronosPriors`` per batch element.  Each is validated:

        * ``q10 ≤ q50 ≤ q90``   (monotonic quantiles)
        * ``dispersion ≥ 0``
        * ``prob_up ∈ [0, 1]``
        * All values finite

    Raises
    ------
    ValueError
        If ``strict=True`` and any invariant is violated.

    Notes
    -----
    All return values represent **end-of-horizon cumulative returns**::

        return_t = (predicted_price_t+H / price_t) - 1
    """
    # Lazy import to avoid pulling heavy deps at module level
    from algaie.models.foundation.chronos2_teacher import (
        ChronosPriors,
        infer_priors,
    )

    # --- Normalise input shape --------------------------------------------
    if input_tensor.ndim == 1:
        input_tensor = input_tensor.unsqueeze(0).unsqueeze(-1)  # [T] -> [1,T,1]
    elif input_tensor.ndim == 2:
        if input_tensor.shape[0] > input_tensor.shape[1]:
            # Likely [T, F] single series — unsqueeze batch
            input_tensor = input_tensor.unsqueeze(0)  # [T,F] -> [1,T,F]
        # else: [B, T] — infer_priors handles this

    # --- Delegate to verified infer_priors --------------------------------
    priors: List[ChronosPriors] = infer_priors(
        model=teacher,
        input_tensor=input_tensor,
        horizon_days=horizon_days,
        n_samples=1,  # NLL teachers are deterministic
        mode=mode,
        strict=strict,
    )

    # --- Extra validation for selector consumption ------------------------
    for i, p in enumerate(priors):
        # infer_priors already calls p.validate(strict=strict), but we do
        # a secondary check for NaN propagation here to be safe.
        for field_name in ("drift", "vol_forecast", "tail_risk", "q10",
                           "q50", "q90", "dispersion", "prob_up"):
            val = getattr(p, field_name)
            if not math.isfinite(val):
                msg = f"Priors[{i}].{field_name} = {val} is not finite"
                if strict:
                    raise ValueError(msg)
                logger.warning(msg)

    return priors


def priors_to_row(
    priors: "ChronosPriors",
    horizon_suffix: str,
) -> Dict[str, float]:
    """Flatten a ``ChronosPriors`` into a dict with horizon-suffixed keys.

    Parameters
    ----------
    priors : ChronosPriors
        Validated priors from ``compute_teacher_priors()``.
    horizon_suffix : str
        E.g. ``"10"`` or ``"30"``.

    Returns
    -------
    dict
        Example keys: ``drift_10``, ``q50_10``, ``prob_up_10``, etc.
    """
    h = horizon_suffix
    return {
        f"drift_{h}": priors.drift,
        f"vol_forecast_{h}": priors.vol_forecast,
        f"tail_risk_{h}": priors.tail_risk,
        f"prob_up_{h}": priors.prob_up,
        f"q10_{h}": priors.q10,
        f"q50_{h}": priors.q50,
        f"q90_{h}": priors.q90,
        f"dispersion_{h}": priors.dispersion,
    }
